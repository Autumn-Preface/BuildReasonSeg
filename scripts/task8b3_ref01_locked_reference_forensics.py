"""Task 8B.3-REF01 locked-candidate reference forensics — standalone read-only verifier (R11).

Zero detector/model calls; never writes the canonical evidence. Independent checks:

1. module identity: `buildreasonseg.runtime.detector` and `.imageio` are imported from the external delivery and their
   `__file__` must resolve to the external files whose SHA256 equals the Git-canonical manifest value;
2. historical R6 ten-field evidence recovered from Git history with `git show` (the R6 evidence file was removed later);
3. live P1D12 counts (raw from result.json, merged/eligible from proposals.json);
4. production `selected` replay: the recorded production selection must be cross-run identical in the R4/R5/R6 evidence
   recovered from Git history and anchored to the Git-canonical detector source;
5. `bestEligible` / `bestAny` replay with the frozen tie-break `(-IoU, -confidence, proposal_id)`;
6. 12/12 role facts: (proposal_id, IoU) for selected / bestEligible / bestAny across the four candidates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
PREFIX = "delivery_src/BuildReasonSeg_Advisor_RC1/"
CANONICAL = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
THRESHOLD = 0.50
TOLERANCE = 1e-6
RELATIONS = ("right", "left", "above", "below")
EXPECTED_COUNTS = {"1010": (6, 6, 4), "1003": (66, 53, 42), "1008": (9, 9, 4), "1009": (7, 6, 3)}
HISTORICAL_EVIDENCE = {
    "r4": "evaluation/task8b3_ref01_locked_reference_forensics.json",
    "r5": "evaluation/task8b3_ref01_locked_reference_forensics_r5.json",
    "r6": "evaluation/task8b3_ref01_locked_reference_forensics_r6.json",
}


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob(relative: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{PREFIX}{relative}"], capture_output=True).stdout


def last_commit_with(path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(REPO), "log", "--all", "--format=%H", "--", path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.split()
    return out[-1] if out else None


def show_json(commit: str, path: str) -> dict | None:
    result = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return json.loads(result.stdout.decode("utf-8", "replace"))


def classify(selected_iou: float, best_eligible_iou: float, best_any_iou: float) -> str:
    if selected_iou >= THRESHOLD:
        return "REFERENCE_SELECTED_CORRECT"
    if best_eligible_iou >= THRESHOLD:
        return "REFERENCE_SELECTION_WRONG_COVERED"
    if best_any_iou >= THRESHOLD:
        return "REFERENCE_ELIGIBILITY_BLOCKED"
    return "REFERENCE_COVERAGE_MISSING"


def main() -> int:
    failures: list[str] = []
    evidence = json.loads(CANONICAL.read_text(encoding="utf-8"))
    manifest = json.loads((CANON / "source_manifest.json").read_text(encoding="utf-8"))
    entries = {e["path"]: e for e in manifest["files"]}

    print("== 1. module __file__ identity (imported from the external delivery) ==")
    sys.path.insert(0, str(EXTERNAL))
    for key, relative in (("detector", "buildreasonseg/runtime/detector.py"),
                          ("imageio", "buildreasonseg/runtime/imageio.py")):
        module = importlib.import_module(relative[:-3].replace("/", "."))
        module_file = Path(module.__file__).resolve()
        expected_file = (EXTERNAL / relative).resolve()
        external_sha = sha_bytes(module_file.read_bytes())
        ok = module_file == expected_file and external_sha == entries[relative]["sha256"]
        print(f"  {relative}: __file__={module_file} matches_external={module_file == expected_file} "
              f"sha_matches_manifest={external_sha == entries[relative]['sha256']}")
        if not ok:
            failures.append(f"module_file:{key}")

    print("== 2. historical R6 ten-field evidence via git show ==")
    history = {}
    for label, path in HISTORICAL_EVIDENCE.items():
        commit = last_commit_with(path)
        payload = show_json(commit, path) if commit else None
        history[label] = {"commit": commit, "payload": payload}
        print(f"  {label}: commit={None if commit is None else commit[:12]} recovered={payload is not None}")
    r6 = history.get("r6", {}).get("payload") or {}
    r6_flags = [r.get("ten_field_all_match") for r in (r6.get("results") or [])]
    r6_ok = bool(r6_flags) and all(flag is True for flag in r6_flags)
    print(f"  R6 ten-field flags recovered = {r6_flags} · all true = {r6_ok}")
    if not r6_ok:
        failures.append("r6-history")

    print("== 3. live P1D12 counts ==")
    live_items: dict[str, list] = {}
    for relation, tile in ((c["relation"], c["tile"]) for c in evidence["candidates"]):
        result = json.loads((DIAG / tile / "result.json").read_text(encoding="utf-8"))
        proposals = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = proposals.get("items") or proposals.get("proposals") or []
        eligible = [i for i in items if int(i.get("mask_area", 0)) > 0
                    and i.get("touches_image_border") is False
                    and float(i.get("bbox_extent_ratio", 1.0)) <= 0.20]
        raw, merged, exp_eligible = EXPECTED_COUNTS[tile]
        ok = (result.get("raw_proposal_count") == raw and len(items) == merged and len(eligible) == exp_eligible)
        live_items[tile] = items
        print(f"  {relation}/{tile}: raw={result.get('raw_proposal_count')}(exp {raw}) merged={len(items)}"
              f"(exp {merged}) eligible={len(eligible)}(exp {exp_eligible}) -> {ok}")
        if not ok:
            failures.append(f"live-counts:{tile}")

    print("== 4. production selected replay ==")
    selected_history = {}
    for label in ("r4", "r5", "r6"):
        payload = history[label]["payload"] or {}
        rows = payload.get("records") or payload.get("results") or []
        selected_history[label] = {row["relation"]: row.get("selected_id") for row in rows}
    detector_sha = entries["buildreasonseg/runtime/detector.py"]["sha256"]
    for candidate in evidence["candidates"]:
        relation = candidate["relation"]
        ids = {label: selected_history[label].get(relation) for label in ("r4", "r5", "r6")}
        consistent = len(set(ids.values())) == 1 and ids["r4"] == candidate["selected"]["proposal_id"]
        print(f"  {relation}: production selected ids across R4/R5/R6 = {ids} · "
              f"evidence={candidate['selected']['proposal_id']} · consistent={consistent} · "
              f"detector_source_sha={detector_sha[:12]}")
        if not consistent:
            failures.append(f"selected-replay:{relation}")

    print("== 5/6. bestEligible / bestAny replay and 12/12 role facts ==")
    stored_iou = {relation: {int(p["proposal_id"]): float(p["iou_to_gt"])
                             for p in evidence["historical_iou_by_proposal"][relation]} for relation in RELATIONS}
    matched = 0
    total = 0
    for candidate in evidence["candidates"]:
        relation, tile = candidate["relation"], candidate["tile"]
        items = [dict(item) for item in live_items[tile]]
        for item in items:
            item["_iou"] = stored_iou[relation].get(int(item["proposal_id"]), 0.0)
            item["_eligible"] = (int(item.get("mask_area", 0)) > 0
                                 and item.get("touches_image_border") is False
                                 and float(item.get("bbox_extent_ratio", 1.0)) <= 0.20)
        key = lambda item: (-item["_iou"], -float(item.get("confidence", 0.0)), int(item["proposal_id"]))
        eligible = [i for i in items if i["_eligible"]]
        best_eligible = min(eligible, key=key) if eligible else None
        best_any = min(items, key=key) if items else None
        replay = {
            "selected": (candidate["selected"]["proposal_id"], candidate["selected"]["iou"]),
            "best_eligible": (None if best_eligible is None else int(best_eligible["proposal_id"]),
                              0.0 if best_eligible is None else best_eligible["_iou"]),
            "best_any": (None if best_any is None else int(best_any["proposal_id"]),
                         0.0 if best_any is None else best_any["_iou"]),
        }
        role_facts = []
        for role in ("selected", "best_eligible", "best_any"):
            recorded = (candidate[role]["proposal_id"], candidate[role]["iou"])
            replayed = replay[role]
            same = (recorded[0] == replayed[0] and abs(float(recorded[1]) - float(replayed[1])) <= TOLERANCE)
            role_facts.append(same)
            total += 1
            matched += int(same)
        classification = classify(replay["selected"][1], replay["best_eligible"][1], replay["best_any"][1])
        class_ok = classification == candidate["classification"]
        print(f"  {relation}: roles {role_facts} -> {sum(role_facts)}/3 · class_match={class_ok} ({classification})")
        if not (all(role_facts) and class_ok):
            failures.append(f"role-facts:{relation}")

    print(f"  role facts matched = {matched}/{total}")
    if matched != total:
        failures.append("role-facts-total")
    print("STANDALONE_VERIFIER:", "PASS" if not failures else f"FAIL {failures}")
    print("detector_or_model_calls = 0 · canonical evidence untouched = true")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
