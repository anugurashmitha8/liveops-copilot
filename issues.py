"""Operational issue detection over the World Model.

Joins beds, patients, cleaning, transport and staff to surface bottlenecks
that no single feed shows on its own. Every issue carries evidence: the exact
entity, attribute, value, source and timestamp it was based on.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from logger import log
from world_model import LIVE, REF, _eid

OPEN_CLEANING = {"pending", "queued", "requested", "assigned", "in_progress"}
OPEN_TRANSPORT = {"requested", "pending", "assigned", "in_progress"}
DISCHARGE_STATES = {"discharge_ordered", "ready_for_discharge"}


def _parse_ts(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _get(ent, attr, prefer=(LIVE, REF)):
    """(value, source, as_of) for an attribute, preferring live data."""
    if not ent:
        return None, None, None
    slots = ent["attrs"].get(attr) or {}
    for src in prefer:
        if src in slots:
            return slots[src]["value"], src, slots[src]["as_of"]
    return None, None, None


def _val(ent, attr):
    return _get(ent, attr)[0]


def _ev(eid, ent, attr):
    value, src, as_of = _get(ent, attr)
    return {"entity": eid, "attribute": attr, "value": value,
            "source": src, "as_of": as_of}


def _minutes(later, earlier):
    if not later or not earlier:
        return None
    return round((later - earlier).total_seconds() / 60)


def _model_now(wm):
    """Latest live timestamp in the model = the hospital's 'now'."""
    latest = None
    for ent in wm.entities.values():
        for slots in ent["attrs"].values():
            slot = slots.get(LIVE)
            ts = _parse_ts(slot.get("as_of")) if slot else None
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest or datetime.now(timezone.utc)


def _staff_by_label(wm):
    idx = {}
    for eid, ent in wm.entities.items():
        if ent["kind"] == "staff" and _val(ent, "label"):
            idx[str(_val(ent, "label"))] = eid
    return idx


def find_issues(wm, now=None):
    now = now or _model_now(wm)
    staff_idx = _staff_by_label(wm)
    issues, seen = [], set()

    def add(kind, severity, summary, evidence, entities):
        key = (kind, tuple(sorted(set(entities))))
        if key in seen:
            return
        seen.add(key)
        issue = {"type": kind, "severity": severity, "summary": summary,
                 "entities": sorted(set(entities)), "evidence": evidence}
        issues.append(issue)
        log("issue_detector", "issue", {"type": kind, "summary": summary,
                                         "entities": issue["entities"]})

    def cleaner_note(bed_eid, bed, ev, ents):
        cleaner = _val(bed, "cleaning_assigned_to")
        ev += [_ev(bed_eid, bed, "cleaning_status"),
               _ev(bed_eid, bed, "cleaning_assigned_to")]
        if not cleaner:
            return "no one assigned to clean it", False
        sid = staff_idx.get(str(cleaner))
        if not sid:
            return f"cleaning assigned to {cleaner}", False
        staff = wm.entities[sid]
        ents.append(sid)
        ev.append(_ev(sid, staff, "status"))
        status = _val(staff, "status")
        return f"cleaning assigned to {cleaner} ({status})", status == "on_break"

    patients = [(e, p) for e, p in wm.entities.items() if p["kind"] == "patient"]
    beds = [(e, b) for e, b in wm.entities.items() if b["kind"] == "bed"]

    # Discharge backlog per unit: cleared to leave, still in a bed, no transport.
    backlog = defaultdict(list)
    for pid, p in patients:
        if (_val(p, "status") in DISCHARGE_STATES
                and _val(p, "transport_status") not in OPEN_TRANSPORT
                and _val(p, "unit_id")):
            backlog[str(_val(p, "unit_id"))].append(pid)

    # 1. ED boarders: why hasn't the patient moved upstairs?
    for pid, p in patients:
        if _val(p, "status") != "boarding":
            continue
        name = p["native_id"]
        waited = _minutes(now, _parse_ts(_val(p, "admit_time")))
        ev = [_ev(pid, p, a) for a in
              ("status", "assigned_bed", "admit_time", "transport_status")]
        ents = [pid]
        target = _val(p, "assigned_bed")
        if not target:
            w = waited or 0
            sev = "high" if w >= 60 else "medium" if w >= 30 else "low"
            add("boarder_no_bed", sev,
                f"{name} admitted, boarding in ED bed {_val(p, 'bed_id')} for "
                f"{w} min; no inpatient bed assigned yet", ev, ents)
            continue
        bid = _eid("bed", target)
        bed = wm.entities.get(bid)
        ents.append(bid)
        ev.append(_ev(bid, bed, "status"))
        bstatus = _val(bed, "status")
        if bstatus in ("dirty", "cleaning"):
            note, _ = cleaner_note(bid, bed, ev, ents)
            cause = f"assigned bed {target} is {bstatus}; {note}"
        elif bstatus == "occupied":
            cause = f"assigned bed {target} is still occupied"
            occ = _val(bed, "patient_id")
            if occ:
                oid = _eid("patient", occ)
                other = wm.entities.get(oid)
                ents.append(oid)
                ev.append(_ev(oid, other, "status"))
                cause += f" by {occ} ({_val(other, 'status')})"
        elif bstatus in ("available", "reserved"):
            cause = (f"assigned bed {target} is {bstatus}; transport is "
                     f"{_val(p, 'transport_status') or 'not requested'}")
        else:
            cause = f"assigned bed {target} status is {bstatus}"
        tunit = _val(bed, "unit_id")
        if tunit and backlog.get(str(tunit)):
            cause += (f"; meanwhile {len(backlog[str(tunit)])} patient(s) on "
                      f"{tunit} are cleared for discharge with no active transport request")
        kind, sev = "boarder_blocked", "high"
        if bstatus == "reserved" and _val(p, "transport_status") == "completed":
            kind, sev = "stale_boarder", "medium"
            cause = (f"transport to {target} is completed, but the patient is still "
                     f"listed as boarding and the bed as reserved: records not updated")
        elif (bstatus in ("available", "reserved")
                and _val(p, "transport_status") in OPEN_TRANSPORT):
            kind = "boarder_moving"
            sev = "low" if (waited or 0) < 30 else "medium"
        add(kind, sev, f"{name} boarding in ED for {waited} min: {cause}", ev, ents)

    # 2. Cleaning past its target time
    for bid, bed in beds:
        if _val(bed, "cleaning_status") not in OPEN_CLEANING:
            continue
        age = _minutes(now, _parse_ts(_val(bed, "cleaning_created_at")))
        try:
            target = int(float(_val(bed, "cleaning_target_minutes")))
        except (TypeError, ValueError):
            continue
        if age is None or age <= target:
            continue
        ev = [_ev(bid, bed, "status"), _ev(bid, bed, "cleaning_created_at"),
              _ev(bid, bed, "cleaning_target_minutes")]
        ents = [bid]
        note, on_break = cleaner_note(bid, bed, ev, ents)
        add("cleaning_overdue", "high" if on_break else "medium",
            f"Bed {bed['native_id']} cleaning open {age} min "
            f"(target {target}); {note}", ev, ents)

    # 3. Discharge backlog: beds held by patients cleared to leave
    boarders_by_unit = defaultdict(list)
    for pid, p in patients:
        if _val(p, "status") == "boarding" and _val(p, "assigned_bed"):
            tbed = wm.entities.get(_eid("bed", _val(p, "assigned_bed")))
            if tbed and _val(tbed, "unit_id") and _val(tbed, "status") not in ("reserved", "available"):
                boarders_by_unit[str(_val(tbed, "unit_id"))].append(pid)  # only boarders whose bed isn't ready yet
    for unit, pids in backlog.items():
        waiting = boarders_by_unit.get(unit, [])
        ev = [_ev(pid, wm.entities[pid], a) for pid in pids[:5]
              for a in ("status", "bed_id", "transport_status")]
        ev += [_ev(bp, wm.entities[bp], a) for bp in waiting
               for a in ("status", "bed_id", "assigned_bed")]
        add("discharge_backlog", "high" if waiting else "medium",
            f"Unit {unit}: {len(pids)} patient(s) cleared for discharge with "
            f"no active transport request, still holding beds"
            + (f"; {len(waiting)} ED boarder(s) waiting for a bed here" if waiting else ""),
            ev, [_eid("unit", unit), *pids, *waiting])

    # 4. Feeds disagree about who is in which bed
    for bid, bed in beds:
        occ = _val(bed, "patient_id")
        if not occ:
            continue
        pid = _eid("patient", occ)
        p = wm.entities.get(pid)
        if p is None or _val(p, "status") is None:
            add("feed_mismatch", "medium",
                f"Beds feed lists {occ} in {bed['native_id']}, "
                f"but the patients feed has no current record of {occ}",
                [_ev(bid, bed, "patient_id")], [bid, pid])
            continue
        if _val(p, "status") == "discharged":
            add("stale_bed", "high",
                f"Bed {bed['native_id']} still shows {occ} as occupant, "
                f"but {occ} is already discharged",
                [_ev(bid, bed, "patient_id"), _ev(bid, bed, "status"),
                 _ev(pid, p, "status")], [bid, pid])
            continue    
        pbed = _val(p, "bed_id")
        if (_val(bed, "status") == "reserved"
                and _val(p, "assigned_bed") == bed["native_id"]):
            continue  # bed is held for this incoming patient: expected
        if pbed and pbed != bed["native_id"]:
            add("feed_mismatch", "medium",
                f"Beds feed lists {occ} in {bed['native_id']}, "
                f"but the patients feed puts {occ} in {pbed}",
                [_ev(bid, bed, "patient_id"), _ev(pid, p, "bed_id")],
                [bid, pid])

    # 5. Units with no free beds: who is waiting on cleaning, who could leave
    per_unit = defaultdict(Counter)
    for _, bed in beds:
        unit = _val(bed, "unit_id")
        if unit:
            per_unit[str(unit)][_val(bed, "status")] += 1
    for unit, counts in per_unit.items():
        waiting = counts["dirty"] + counts["cleaning"]
        if counts["available"] == 0 and waiting > 0:
            uid = _eid("unit", unit)
            uent = wm.entities.get(uid)
            held = backlog.get(unit, [])
            turning = [bid for bid, b in beds if _val(b, "unit_id") == unit
                       and _val(b, "status") in ("dirty", "cleaning")]
            ev = [_ev(uid, uent, "BED_COUNT"),
                  {"entity": uid, "attribute": "live_status_counts",
                   "value": dict(counts), "source": LIVE, "as_of": None}]
            ev += [_ev(bid, wm.entities[bid], "cleaning_status") for bid in turning[:5]]
            ev += [_ev(pid, wm.entities[pid], "status") for pid in held[:5]]
            summary = (f"Unit {unit} has 0 available beds (official count "
                       f"{_val(uent, 'BED_COUNT')}); {waiting} bed(s) waiting on cleaning")
            if held:
                summary += (f"; {len(held)} patient(s) cleared for discharge with "
                            f"no active transport request are still holding beds")
            add("unit_full", "high", summary, ev, [uid, *turning[:5], *held[:5]])

    return issues


if __name__ == "__main__":
    from world_model import build_and_save

    wm = build_and_save()
    found = find_issues(wm)
    print("\nissues by type:", dict(Counter(i["type"] for i in found)))
    for issue in found[:10]:
        print(f"  [{issue['severity']}] {issue['summary']}")
    with open("logs/issues_latest.json", "w", encoding="utf-8") as f:
        json.dump(found, f, indent=2, default=str)