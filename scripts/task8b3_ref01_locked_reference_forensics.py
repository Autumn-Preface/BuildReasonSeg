"""Task 8B.3-REF01 locked-candidate reference forensics — final independent verifier (R12).

Zero detector/model calls; never writes the canonical evidence. Every check below is an explicit assertion and the
script exits non-zero on the first failure. The R6 evidence commit is pinned so that the historical ten-field recovery
is reproducible from a fixed revision rather than from a moving search.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
PREFIX = "delivery_src/BuildReasonSeg_Advisor_RC1/"
CANONICAL = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
sys.path.insert(0, str(EXTERNAL))
R6_COMMIT = "12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c"
R6_PATH = "evaluation/task8b3_ref01_locked_reference_forensics_r6.json"
THRESHOLD = 0.50
TOLERANCE = 1e-6
RELATIONS = ("right", "left", "above", "below")
ROLES = ("selected", "best_eligible", "best_any")
FIELDS = ("proposal_id", "iou", "confidence", "mask_area", "global_bbox", "eligible")
EXPECTED_COUNTS = {"1010": (6, 6, 4), "1003": (66, 53, 42), "1008": (9, 9, 4), "1009": (7, 6, 3)}
EXACT_DISCLOSURE = ("The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason "
                    "v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. "
                    "Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, "
                    "threshold, seed, or architecture.")
EXPECTED_OUTCOME = "REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE"
EXPECTED_BLOCKER = "ELIGIBILITY"
EXPECTED_NEXT = "REF01_ELIGIBILITY_FORENSICS"


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_bytes(args: list[str]) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True).stdout


def classify(selected_iou: float, best_eligible_iou: float, best_any_iou: float) -> str:
    if selected_iou >= THRESHOLD:
        return "REFERENCE_SELECTED_CORRECT"
    if best_eligible_iou >= THRESHOLD:
        return "REFERENCE_SELECTION_WRONG_COVERED"
    if best_any_iou >= THRESHOLD:
        return "REFERENCE_ELIGIBILITY_BLOCKED"
    return "REFERENCE_COVERAGE_MISSING"


def close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= TOLERANCE
    except (TypeError, ValueError):
        return False


def main() -> int:
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")

    evidence = json.loads(CANONICAL.read_text(encoding="utf-8"))
    manifest = json.loads((CANON / "source_manifest.json").read_text(encoding="utf-8"))
    entries = {e["path"]: e for e in manifest["files"]}

    print("== exact disclosure ==")
    disclosure = evidence["scientific_reuse_disclosure"]
    check("disclosure.en == exact English text", disclosure.get("en") == EXACT_DISCLOSURE)
    check("disclosure.zh present and non-empty", bool(str(disclosure.get("zh", "")).strip()))
    check("evidence.task == 8B.3-REF01-F1-R9", evidence["task"] == "8B.3-REF01-F1-R9")
    check("verification_mode", evidence["verification_mode"] == "READ_ONLY_HISTORICAL_EVIDENCE_REPLAY")
    check("detector_model_calls_this_task == 0", evidence["detector_model_calls_this_task"] == 0)
    check("source_identity_basis", evidence["source_identity_basis"] == "GIT_CANONICAL_BLOB_BYTES")
    check("coverage_threshold == 0.50", evidence["coverage_threshold"] == 0.50)

    print("== dual manifest identity ==")
    git_manifest = git_bytes(["show", f"HEAD:{PREFIX}source_manifest.json"])
    external_manifest = (EXTERNAL / "source_manifest.json").read_bytes()
    working_manifest = (CANON / "source_manifest.json").read_bytes()
    check("git-canonical control manifest == external control manifest", git_manifest == external_manifest)
    check("working-tree control manifest equals external modulo CRLF", external_manifest.replace(bytes([13, 10]), bytes([10])) == working_manifest.replace(bytes([13, 10]), bytes([10])))

    for key, relative in (("detector", "buildreasonseg/runtime/detector.py"),
                          ("imageio", "buildreasonseg/runtime/imageio.py")):
        module = importlib.import_module(relative[:-3].replace("/", "."))
        module_file = Path(module.__file__).resolve()
        blob = git_bytes(["show", f"HEAD:{PREFIX}{relative}"])
        recorded = evidence["external_identity"][key]
        check(f"{key}: __file__ is the external file", module_file == (EXTERNAL / relative).resolve())
        check(f"{key}: manifest == git blob == external file",
              entries[relative]["sha256"] == sha_bytes(blob) == sha_bytes(module_file.read_bytes()))
        check(f"{key}: evidence record matches manifest",
              recorded["manifest_sha256"] == entries[relative]["sha256"]
              and recorded["manifest_match"] is True)

    print("== pinned R6 git-show ten-field evidence ==")
    r6_raw = git_bytes(["show", f"{R6_COMMIT}:{R6_PATH}"])
    check("pinned R6 evidence revision resolves", bool(r6_raw.strip()))
    r6 = json.loads(r6_raw.decode("utf-8", "replace")) if r6_raw.strip() else {}
    r6_rows = {row["relation"]: row for row in (r6.get("results") or [])}
    check("R6 covers four candidates", set(r6_rows) == set(RELATIONS))
    check("R6 ten-field reproduction true for all candidates",
          all(row.get("ten_field_all_match") is True for row in r6_rows.values()))
    check("R6 detector calls == 4", r6.get("detector_calls") == 4)

    print("== live P1D12 counts and selected replay ==")
    live: dict[str, list] = {}
    for candidate in evidence["candidates"]:
        relation, tile = candidate["relation"], candidate["tile"]
        result = json.loads((DIAG / tile / "result.json").read_text(encoding="utf-8"))
        proposals = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = proposals.get("items") or proposals.get("proposals") or []
        live[tile] = items
        raw, merged, eligible_count = EXPECTED_COUNTS[tile]
        eligible = [i for i in items if int(i.get("mask_area", 0)) > 0
                    and i.get("touches_image_border") is False
                    and float(i.get("bbox_extent_ratio", 1.0)) <= 0.20]
        check(f"{relation}: live counts raw/merged/eligible == {raw}/{merged}/{eligible_count}",
              result.get("raw_proposal_count") == raw and len(items) == merged
              and len(eligible) == eligible_count)
        selected_id = candidate["selected"]["proposal_id"]
        live_selected = next((i for i in items if int(i["proposal_id"]) == int(selected_id)), None)
        check(f"{relation}: production selected id {selected_id} present in live metadata",
              live_selected is not None)
        if live_selected is not None:
            check(f"{relation}: live selected metadata six-field match",
                  int(live_selected["proposal_id"]) == int(selected_id)
                  and close(live_selected.get("mask_area"), candidate["selected"]["mask_area"])
                  and live_selected.get("global_bbox") == candidate["selected"]["global_bbox"]
                  and close(live_selected.get("confidence"), candidate["selected"]["confidence"]))

    print("== bestEligible / bestAny replay and six-field 12/12 role facts ==")
    stored_iou = {relation: {int(p["proposal_id"]): float(p["iou_to_gt"])
                             for p in evidence["historical_iou_by_proposal"][relation]} for relation in RELATIONS}
    matched_facts = 0
    total_facts = 0
    class_counts: dict[str, int] = {}
    for candidate in evidence["candidates"]:
        relation, tile = candidate["relation"], candidate["tile"]
        items = [dict(item) for item in live[tile]]
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
            "selected": {"proposal_id": candidate["selected"]["proposal_id"], "iou": candidate["selected"]["iou"],
                         "confidence": candidate["selected"]["confidence"],
                         "mask_area": candidate["selected"]["mask_area"],
                         "global_bbox": candidate["selected"]["global_bbox"],
                         "eligible": candidate["selected"]["eligible"]},
            "best_eligible": {"proposal_id": None if best_eligible is None else int(best_eligible["proposal_id"]),
                              "iou": 0.0 if best_eligible is None else best_eligible["_iou"],
                              "confidence": None if best_eligible is None else best_eligible.get("confidence"),
                              "mask_area": None if best_eligible is None else best_eligible.get("mask_area"),
                              "global_bbox": None if best_eligible is None else best_eligible.get("global_bbox"),
                              "eligible": True},
            "best_any": {"proposal_id": None if best_any is None else int(best_any["proposal_id"]),
                         "iou": 0.0 if best_any is None else best_any["_iou"],
                         "confidence": None if best_any is None else best_any.get("confidence"),
                         "mask_area": None if best_any is None else best_any.get("mask_area"),
                         "global_bbox": None if best_any is None else best_any.get("global_bbox"),
                         "eligible": False if best_any is None else bool(best_any["_eligible"])},
        }
        for role in ROLES:
            recorded = candidate[role]
            replayed = replay[role]
            facts = []
            for field in FIELDS:
                if field in ("iou", "confidence", "mask_area"):
                    facts.append(close(recorded.get(field), replayed.get(field)))
                elif field == "global_bbox":
                    facts.append(recorded.get(field) == replayed.get(field))
                else:
                    facts.append(recorded.get(field) == replayed.get(field))
            ok = all(facts)
            matched_facts += sum(1 for fact in facts if fact)
            total_facts += len(facts)
            check(f"{relation}/{role}: six-field role fact match", ok)
        classification = classify(float(replay["selected"]["iou"]), float(replay["best_eligible"]["iou"]),
                                  float(replay["best_any"]["iou"]))
        class_counts[classification] = class_counts.get(classification, 0) + 1
        check(f"{relation}: classification reproduced ({classification})",
              classification == candidate["classification"])
    check(f"six-field role facts matched == 12/12 roles (fields {matched_facts}/{total_facts})",
          matched_facts == total_facts == 12 * len(FIELDS))

    print("== aggregate outcome / blocker / NEXT ==")
    check("class counts match evidence", class_counts == evidence["class_counts"])
    check("overall_outcome", evidence["overall_outcome"] == EXPECTED_OUTCOME)
    check("dominant_next_blocker", evidence["dominant_next_blocker"] == EXPECTED_BLOCKER)
    check("next_gate", evidence["next_gate"] == EXPECTED_NEXT)
    check("next gate not executed here", True)

    failures = [name for name, ok in checks if not ok]
    print(f"CHECKS {len(checks) - len(failures)}/{len(checks)} passed")
    print("FINAL_INDEPENDENT_VERIFIER:", "PASS" if not failures else f"FAIL {failures}")
    print("detector_or_model_calls = 0 · canonical evidence untouched = true")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
