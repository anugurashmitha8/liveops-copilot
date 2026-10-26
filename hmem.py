"""H-MEM: tiered memory for the Live Ops agents (Meterless H-MEM pattern).

Plain, grep-able files under memory/:
- working.json    open issues and tracked patient statuses, with first-seen times
- episodic.jsonl  one line per investigation or resolved issue
- semantic.json   lessons consolidated from repeated episodes
- ledger.jsonl    every memory write, with its reason

Conflict rule: a newer live observation of an entity's status replaces the
older one, and the ledger records what was replaced and why.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from logger import RUN_ID, log

LIVE = "live_api"
TRACKED_STATUSES = {"boarding", "discharge_ordered", "ready_for_discharge"}
RESOLVE_AFTER_MIN = 3  # an issue must be gone this long before it counts as resolved
GAP_MIN = 30  # no check for this long: the Copilot wasn't watching, so restart open clocks

def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _mins(later, earlier):
    return round((later - earlier).total_seconds() / 60)


def issue_key(issue):
    """Stable identity for an issue across runs: type plus its unit (or main entity)."""
    ents = issue.get("entities") or [""]
    unit = next((e for e in ents if e.startswith("unit:")), None)
    return f"{issue['type']}|{unit or ents[0]}"


class HMem:
    def __init__(self, root="memory"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.working = self._load("working.json", {"issues": {}, "status": {}})
        self.semantic = self._load("semantic.json", {"lessons": {}})

    # ---------- storage ----------

    def _path(self, name):
        return os.path.join(self.root, name)

    def _load(self, name, default):
        try:
            with open(self._path(name), encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def save(self):
        for name, data in (("working.json", self.working),
                           ("semantic.json", self.semantic)):
            with open(self._path(name), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    def _ledger(self, action, data):
        entry = {"ts": _iso(datetime.now(timezone.utc)), "run_id": RUN_ID,
                 "action": action, **data}
        with open(self._path("ledger.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        log("hmem", action, data)

    def _append_episode(self, episode):
        with open(self._path("episodic.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, default=str) + "\n")
        self._ledger("episode_written", {"key": episode["key"],
                                         "kind": episode["kind"]})

    def episodes(self, key=None):
        try:
            with open(self._path("episodic.jsonl"), encoding="utf-8") as f:
                eps = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
        return [e for e in eps if key is None or e.get("key") == key]

    # ---------- working memory ----------

    def observe(self, wm, issues, now):
        """Update working memory from the current World Model and issues."""
        now_s = _iso(now)
        last = self.working.get("last_observe")
        if last and _mins(now, _parse(last)) > GAP_MIN:
            # Durations across a gap would be invented: close open issues at their
            # last sighting and start fresh.
            for key, rec in list(self.working["issues"].items()):
                self._append_episode({
                    "kind": "resolved", "key": key, "type": rec["type"],
                    "first_seen": rec["first_seen"], "cleared_at": rec["last_seen"],
                    "minutes_open": _mins(_parse(rec["last_seen"]), _parse(rec["first_seen"])),
                    "last_summary": rec["summary"], "run_id": RUN_ID, "gap": True})
            self.working["issues"] = {}
            self.working["status"] = {}
            self._ledger("observation_gap", {
                "since": last, "minutes": _mins(now, _parse(last)),
                "reason": "no checks during the gap; open clocks restarted"})
        self.working["last_observe"] = now_s

        status = self.working["status"]
        for eid, ent in wm.entities.items():
            if ent["kind"] != "patient":
                continue
            slot = (ent["attrs"].get("status") or {}).get(LIVE)
            value = slot["value"] if slot else None
            prev = status.get(eid)
            if value in TRACKED_STATUSES:
                if not prev or prev["status"] != value:
                    status[eid] = {"status": value, "first_seen": now_s}
                    self._ledger("status_observed", {
                        "entity": eid, "status": value,
                        "replaced": prev["status"] if prev else None,
                        "reason": "newer live observation"})
            elif prev:
                del status[eid]
                self._ledger("status_cleared", {
                    "entity": eid, "was": prev["status"], "now": value,
                    "reason": "newer live observation"})

        open_issues = self.working["issues"]
        seen, new, ongoing, resolved = set(), [], [], []
        for issue in issues:
            key = issue_key(issue)
            seen.add(key)
            rec = open_issues.get(key)
            if rec is None:
                open_issues[key] = {"type": issue["type"], "summary": issue["summary"],
                                    "first_seen": now_s, "last_seen": now_s,
                                    "seen_count": 1}
                new.append(key)
            else:
                rec.update(last_seen=now_s, summary=issue["summary"])
                rec["seen_count"] += 1
                ongoing.append(key)

        for key in [k for k in open_issues if k not in seen]:
            if _mins(now, _parse(open_issues[key]["last_seen"])) < RESOLVE_AFTER_MIN:
                continue  
            rec = open_issues.pop(key)
            self._append_episode({
                "kind": "resolved", "key": key, "type": rec["type"],
                "first_seen": rec["first_seen"], "cleared_at": now_s,
                "minutes_open": _mins(now, _parse(rec["first_seen"])),
                "last_summary": rec["summary"], "run_id": RUN_ID})
            resolved.append(key)

        self._ledger("observe", {"new": len(new), "ongoing": len(ongoing),
                                 "resolved": len(resolved)})
        return {"new": new, "ongoing": ongoing, "resolved": resolved}

    def status_age(self, eid, now):
        rec = self.working["status"].get(eid)
        if not rec:
            return None
        return rec["status"], _mins(now, _parse(rec["first_seen"]))

    # ---------- recall ----------

    def recall(self, issue, now):
        key = issue_key(issue)
        lines = []
        rec = self.working["issues"].get(key)
        if rec:
            lines.append(f"This issue was first observed "
                         f"{_mins(now, _parse(rec['first_seen']))} min ago and has "
                         f"been seen in {rec['seen_count']} check(s).")
        for eid in issue.get("entities") or []:
            age = self.status_age(eid, now)
            if age:
                lines.append(f"{eid} has been {age[0]} for at least {age[1]} min "
                             f"(since first observed).")
        for ep in [e for e in self.episodes(key) if e["kind"] == "resolved"][-3:]:
            lines.append(f"Recently this issue cleared after {ep['minutes_open']} "
                         f"min (cleared at {ep['cleared_at']}).")
        lesson = self.semantic["lessons"].get(key)
        if lesson:
            lines.append(f"Lesson: {lesson['text']}")
        self._ledger("recall", {"key": key, "lines": len(lines)})
        return "\n".join(lines) or "No prior memory of this issue."

    # ---------- episodic + semantic ----------

    def record_investigation(self, issue, result):
        rounds = result.get("rounds") or []
        final = rounds[-1]["draft"] if rounds else {}
        self._append_episode({
            "kind": "investigation", "key": issue_key(issue), "type": issue["type"],
            "at": _iso(datetime.now(timezone.utc)), "verified": result["verified"],
            "rounds": len(rounds), "cause": final.get("cause"),
            "actions": final.get("actions"), "run_id": RUN_ID})

    def consolidate(self):
        """Turn repeated episodes into lessons (semantic tier)."""
        by_key = defaultdict(list)
        for ep in self.episodes():
            by_key[ep["key"]].append(ep)
        lessons = {}
        for key, eps in by_key.items():
            verified = [e for e in eps if e["kind"] == "investigation" and e.get("verified")]
            cleared = [e for e in eps if e["kind"] == "resolved"]
            if len(verified) + len(cleared) < 2:
                continue
            parts = []
            if cleared:
                avg = round(sum(e["minutes_open"] for e in cleared) / len(cleared))
                parts.append(f"cleared {len(cleared)} time(s) before, "
                             f"after {avg} min on average")
            if verified:
                parts.append(f"verified {len(verified)} time(s); last verified "
                             f"cause: {verified[-1].get('cause')}")
            text = "; ".join(parts)
            prev = self.semantic["lessons"].get(key)
            if not prev or prev["text"] != text:
                self._ledger("lesson_updated", {
                    "key": key, "text": text,
                    "replaced": prev["text"] if prev else None,
                    "reason": "consolidated from episodes"})
            lessons[key] = {"text": text, "support": len(verified) + len(cleared),
                            "updated": _iso(datetime.now(timezone.utc))}
        self.semantic["lessons"] = lessons
        return lessons