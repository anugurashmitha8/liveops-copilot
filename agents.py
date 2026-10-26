"""Multi-agent loop over the World Model.

Collector -> Monitor -> Investigator <-> Critic -> Explainer

The Investigator explains a problem using only World Model evidence.
The Critic verifies every cited claim in code, then reviews the reasoning
with the LLM, and sends weak drafts back (up to MAX_ROUNDS).
"""

from __future__ import annotations

import argparse
import json
import os
import re

from hmem import HMem, issue_key
from issues import _model_now, find_issues
from logger import RUN_ID, log
from sources import live_snapshot, reference_data
from world_model import WorldModel

MAX_ROUNDS = 4
SEV_RANK = {"high": 0, "medium": 1, "low": 2}
TYPE_RANK = {"boarder_blocked": 0, "boarder_no_bed": 0, "unit_full": 1,
             "stale_bed": 2, "cleaning_overdue": 3, "discharge_backlog": 4,
             "feed_mismatch": 5 ,"boarder_moving": 6}
KEY_ATTRS = {
    "patient": ["status", "bed_id", "assigned_bed", "unit_id", "admit_time",
                "transport_status", "esi_acuity", "expected_discharge"],
    "bed": ["status", "status_since", "unit_id", "patient_id", "cleaning_status",
            "cleaning_assigned_to", "cleaning_created_at", "cleaning_target_minutes"],
    "staff": ["label", "role", "status", "location", "zone", "DEPARTMENT", "SHIFT"],
    "unit": ["UNIT_NAME", "BED_COUNT", "FLOOR"],
}
EID_RE = re.compile(r"\b(?:bed|patient|staff|unit|ambulance):[A-Za-z0-9&_\-]+")


# ---------- helpers ----------

def _llm(prompt, system):
    from llm import generator  # imported lazily so tests run without API keys
    return generator(prompt, system)


def _parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def ask_json(agent, system, prompt):
    for attempt in (1, 2):
        raw = _llm(prompt, system)
        try:
            return _parse_json(raw)
        except (ValueError, json.JSONDecodeError) as err:
            log(agent, "bad_json", {"attempt": attempt, "error": str(err),
                                    "raw": (raw or "")[:500]})
            prompt += "\n\nYour last reply was not valid JSON. Reply with JSON only."
    return {}


def neighborhood(wm, seeds, limit=40):
    picked = list(dict.fromkeys(e for e in seeds if e in wm.entities))
    seedset = set(picked)
    for rel in wm.relations:
        if len(picked) >= limit:
            break
        if rel["subject"] in seedset and rel["object"] not in picked:
            picked.append(rel["object"])
        elif rel["object"] in seedset and rel["subject"] not in picked:
            picked.append(rel["subject"])
    return [e for e in picked if e in wm.entities][:limit]


def compact_context(wm, eids):
    lines = []
    for eid in eids:
        ent = wm.entities[eid]
        parts = []
        for attr in KEY_ATTRS.get(ent["kind"], []):
            for src, slot in (ent["attrs"].get(attr) or {}).items():
                parts.append(f"{attr}={slot['value']!r} ({src}"
                             + (f" @{slot['as_of']}" if attr == "status" else "")
                             + ")")
        lines.append(f"{eid}: " + ("; ".join(parts) or "(no key attributes)"))
    return "\n".join(lines)


# ---------- agents ----------

def collector():
    wm = WorldModel()
    wm.ingest_live(live_snapshot())
    wm.ingest_reference(reference_data())
    log("collector", "world_model_built",
        {"entities": len(wm.entities), "relations": len(wm.relations)})
    print(f"[Collector] World Model: {len(wm.entities)} entities, "
          f"{len(wm.relations)} relations")
    return wm


def monitor(wm):
    issues = find_issues(wm)
    issues.sort(key=lambda i: (SEV_RANK.get(i["severity"], 9),
                               TYPE_RANK.get(i["type"], 9)))
    log("monitor", "issues_ranked",
        {"count": len(issues), "top": [i["summary"] for i in issues[:5]]})
    print(f"[Monitor] {len(issues)} issue(s) found")
    for n, issue in enumerate(issues[:5]):
        print(f"   {n}. [{issue['severity']}] {issue['summary']}")
    return issues


INVESTIGATOR_SYSTEM = (
    "You are the Investigator agent on a hospital operations team. Explain the "
    "problem using ONLY the evidence provided. Every factual statement must be "
    "backed by a claim that cites an entity id and attribute exactly as they "
    "appear in the evidence, with the exact value shown. You may cite the "
    "Monitor's finding as entity \"issue\", attribute \"summary\".\n"
    "Your cause must go beyond restating the finding. Say what it is blocking "
    "right now (for example an ED boarder waiting for a bed on that unit, or no "
    "free beds on the unit). If the evidence shows free beds and nobody waiting, "
    "say plainly that it is not blocking anyone yet, and why it still matters. "
    "Use memory durations when they make the point.\n"
    "If the evidence is not enough to know the cause, say so and lower your "
    "confidence. Propose at most 2 actions, each tied directly to cited evidence; "
    "fewer precise actions beat many vague ones. Reply with JSON only."
)


def investigator(issue, context, question=None, feedback=None, memory=None):
    prompt = (
        f"MONITOR FINDING:\n{issue['summary']}\n\n"
        f"FINDING EVIDENCE:\n{json.dumps(issue['evidence'], default=str)}\n\n"
        f"WORLD MODEL CONTEXT (entity: attribute=value (source)):\n{context}\n\n"
    )
    if memory:
        prompt += f"MEMORY (H-MEM, from this system's earlier checks):\n{memory}\n\n"
    if question:
        prompt += f"OPERATOR QUESTION: {question}\n\n"
    if feedback:
        prompt += ("THE CRITIC REJECTED YOUR LAST DRAFT. Fix these problems:\n- "
                   + "\n- ".join(feedback) + "\n\n")
    prompt += (
        'Return JSON: {"cause": "one sentence", '
        '"explanation": "2-4 sentences", '
        '"actions": ["at most 2 concrete actions"], '
        '"claims": [{"entity": "bed:4E-410A", "attribute": "status", '
        '"value": "dirty"}], '
        '"confidence": "high|medium|low"}'
    )
    draft = ask_json("investigator", INVESTIGATOR_SYSTEM, prompt)
    log("investigator", "draft", draft)
    return draft

def _same(a, b):
    return str(a).strip().lower() == str(b).strip().lower()


def _matches_derived(claim_value, derived_value):
    """Does a claim match a value the Monitor calculated (e.g. bed counts)?"""
    if isinstance(derived_value, dict):
        if isinstance(claim_value, str):
            try:
                claim_value = json.loads(claim_value.replace("'", '"'))
            except json.JSONDecodeError:
                return False
        if isinstance(claim_value, dict):
            return all(_same(derived_value.get(k), v) for k, v in claim_value.items())
        return False
    return _same(claim_value, derived_value)

def verify_claims(wm, draft, allowed=None, derived_evidence=None):
    """Deterministic checks: cited entities exist and values match."""
    problems = []
    claims = draft.get("claims") or []
    derived = {(e.get("entity"), e.get("attribute")): e.get("value") for e in (derived_evidence or [])}
    if not claims:
        problems.append("No claims cited. Every statement needs evidence.")
    for claim in claims:
        eid, attr = claim.get("entity"), claim.get("attribute")
        if eid in ("issue", "memory"):
            continue
        if (eid, attr) in derived and _matches_derived(claim.get("value"), derived[(eid, attr)]):
            continue
        ent = wm.entities.get(eid)
        if ent is None:
            problems.append(f"Claim cites {eid}, which does not exist in the World Model.")
            continue
        if allowed is not None and eid not in allowed:
            problems.append(f"Claim cites {eid}, which is not part of this issue's evidence.")
            continue    
        slots = ent["attrs"].get(attr)
        if not slots:
            if str(claim.get("value")).strip().lower() in ("none", "null", ""):
                continue  # claim says "not set", and it isn't set: true
            problems.append(f"{eid} has no attribute '{attr}'.")
            continue
        actual = {str(s["value"]).strip().lower() for s in slots.values()}
        if str(claim.get("value")).strip().lower() not in actual:
            problems.append(f"{eid}.{attr} is {sorted(actual)}, "
                            f"not {claim.get('value')!r}.")
    text = " ".join([str(draft.get("cause", "")), str(draft.get("explanation", ""))]
                    + [str(a) for a in draft.get("actions") or []])
    for eid in sorted(set(EID_RE.findall(text))):
        if eid not in wm.entities:
            problems.append(f"Text mentions {eid}, which does not exist.")
        elif allowed is not None and eid not in allowed:
            problems.append(f"Text mentions {eid}, which is not part of this issue's evidence.")
        elif allowed is not None and eid not in allowed:
            problems.append(f"Text mentions {eid}, which is not part of this issue's evidence.")
    return problems


CRITIC_SYSTEM = (
    "You are the Critic agent on a hospital operations team. Review another "
    "agent's explanation against the evidence. The Monitor finding is "
    "established and may be cited.\n"
    "A problem is BLOCKING only if it would mislead the bed manager: a false or "
    "unsupported fact, an invented cause, a cause that only restates the Monitor "
    "finding, a draft that contradicts itself or the evidence (for example "
    "claiming patients block admissions while the evidence shows free beds on "
    "that unit and nobody waiting), normal in-progress activity described as a "
    "failure, overstated urgency or confidence, or an action that is wrong, "
    "unsafe or chases something already resolved.\n"
    "Wording, style, or redundant but harmless actions are MINOR.\n"
    'Reply with JSON only: {"blocking": ["..."], "minor": ["..."]}'
)

DOMAIN_NOTES = (
    "\n\nDomain notes: a patient's bed_id is the bed they are in now; "
    "assigned_bed is the inpatient bed they are waiting to move to. 'boarding' "
    "means admitted from the ED and waiting in an ED bed for an inpatient bed. "
    "Bed status 'reserved' means held for an incoming patient. If an attribute "
    "is absent from the evidence, the live feed has not set it: treat it as "
    "not set, and cite it with value null. Memory notes come from this "
    "system's own earlier checks: cite them as entity \"memory\", attribute "
    "\"recall\"; durations in them are lower bounds ('at least')."
)
INVESTIGATOR_SYSTEM += DOMAIN_NOTES
CRITIC_SYSTEM += DOMAIN_NOTES


def critic(wm, issue, context, draft, allowed=None, memory=None):
    code_problems = verify_claims(wm, draft, allowed, issue.get("evidence"))
    prompt = f"MONITOR FINDING:\n{issue['summary']}\n\nEVIDENCE:\n{context}\n\n"
    if memory:
        prompt += f"MEMORY (recorded by this system; established):\n{memory}\n\n"
    prompt += (f"DRAFT TO REVIEW:\n{json.dumps(draft, default=str)}\n\n"
               f"AUTOMATED CLAIM CHECK FOUND: {code_problems or 'no problems'}")
    review = ask_json("critic", CRITIC_SYSTEM, prompt)
    blocking = list(review.get("blocking") or review.get("problems") or [])
    minor = list(review.get("minor") or [])
    problems = code_problems + blocking
    verdict = "pass" if not problems else "revise"
    result = {"verdict": verdict, "problems": problems, "minor": minor,
              "code_problems": code_problems}
    log("critic", "review", result)
    return result


EXPLAINER_SYSTEM = (
    "You are the Explainer agent. Write a short message for the hospital bed "
    "manager from an analysis the Critic has approved. Use only facts in the "
    "analysis. DO NOW must list exactly the analysis's actions: reword them for "
    "clarity, but never add, drop or change an action. Format:\n"
    "WHAT: one sentence\nWHY: one or two sentences\n"
    "DO NOW: the approved actions as bullet points\n"
    "EVIDENCE: the entity ids relied on"
)


def explainer(draft, verified):
    message = _llm(json.dumps(draft, default=str), EXPLAINER_SYSTEM)
    message = re.sub(r"<think>.*?</think>", "", message or "", flags=re.S).strip()
    if not verified:
        message = ("UNVERIFIED: the Critic did not approve this analysis.\n\n"
                   + message)
    log("explainer", "message", {"verified": verified, "message": message})
    return message

# ---------- orchestration ----------

def run(issue_index=0, question=None, key=None):
    wm = collector()
    issues = monitor(wm)
    mem = HMem()
    now = _model_now(wm)
    seen = mem.observe(wm, issues, now)
    mem.consolidate()  # lessons up to date before the agents read them
    print(f"[Memory] {len(seen['new'])} new, {len(seen['ongoing'])} ongoing, "
          f"{len(seen['resolved'])} resolved since last check")
    if not issues:
        mem.save()
        print("[Monitor] Nothing to investigate right now.")
        return None

    if key:
        issue = next((i for i in issues if issue_key(i) == key), None)
        if issue is None:
            target = key.split("|", 1)[-1]  # e.g. "unit:ICU"
            issue = next((i for i in issues
                          if issue_key(i).endswith("|" + target)), None)
            if issue:
                print(f"[Monitor] {key} has cleared; investigating "
                      f"{issue_key(issue)} instead.")
        if issue is None:
            mem.save()
            print(f"[Monitor] Issue {key} is no longer present.")
            return None
    else:
        issue = issues[min(issue_index, len(issues) - 1)]
    eids = neighborhood(wm, issue["entities"])
    context = compact_context(wm, eids)
    memory = mem.recall(issue, now)
    print(f"\n[Monitor] Investigating: {issue['summary']}")
    print("[Memory] recall:\n   " + memory.replace("\n", "\n   "))

    feedback, draft, review, rounds = None, {}, {"verdict": "revise"}, []
    all_feedback = []
    for rnd in range(1, MAX_ROUNDS + 1):
        draft = investigator(issue, context, question, feedback, memory)
        print(f"\n[Investigator] round {rnd}: {draft.get('cause', '(no cause)')}")
        review = critic(wm, issue, context, draft, set(eids), memory)
        rounds.append({"round": rnd, "draft": draft, "review": review})
        print(f"[Critic] round {rnd}: {review['verdict'].upper()}")
        for p in review["problems"][:5]:
            print(f"   - {p}")
        if review["verdict"] == "pass":
            break
        all_feedback += review["problems"]
        feedback = all_feedback

    verified = review["verdict"] == "pass"
    if verified:
        message = explainer(draft, verified)
    else:
        message = (
            f"UNVERIFIED: the agents could not agree on a cause after "
            f"{len(rounds)} rounds.\n\nMONITOR FINDING: {issue['summary']}\n\n"
            "OPEN CONCERNS FROM THE CRITIC:\n- "
            + "\n- ".join(review["problems"][:4])
        )
        log("explainer", "unverified", {"message": message})
    print(f"\n[Explainer]\n{message}")

    result = {"run_id": RUN_ID, "issue": issue, "question": question,
              "memory": memory, "rounds": rounds, "verified": verified,
              "message": message}
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/answer_{RUN_ID}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    mem.record_investigation(issue, result)
    mem.consolidate()
    mem.save()
    log("orchestrator", "done", {"verified": verified, "rounds": len(rounds)})
    return result


def watch(minutes):
    """Observe only (no LLM calls) once a minute, to build memory over time."""
    import time
    for i in range(minutes):
        wm = collector()
        issues = monitor(wm)
        mem = HMem()
        seen = mem.observe(wm, issues, _model_now(wm))
        mem.save()
        print(f"[Memory] check {i + 1}/{minutes}: {len(seen['new'])} new, "
              f"{len(seen['ongoing'])} ongoing, {len(seen['resolved'])} resolved\n")
        if i < minutes - 1:
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Ops multi-agent run")
    parser.add_argument("--issue", type=int, default=0,
                        help="which ranked issue to investigate (0 = most urgent)")
    parser.add_argument("--question", help="optional operator question")
    parser.add_argument("--watch", type=int, metavar="MINUTES",
                        help="observe only, once a minute, to build memory (no LLM)")
    args = parser.parse_args()
    if args.watch:
        watch(args.watch)
    else:
        run(args.issue, args.question)