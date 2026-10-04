"""Task 8B.3-REF01 locked-candidate reference forensics — standalone read-only verifier (R10).

This verifier is independent of the run that produced the evidence. It performs NO detector or model call and never
writes the canonical evidence. It checks, in order:

1. Git-canonical identity of buildreasonseg/runtime/detector.py and buildreasonseg/runtime/imageio.py
   (manifest entry == Git blob == external file) plus the absence of a standalone third-party imageio import;
2. the historical R6 ten-field reproduction evidence and the R4=R5=R6 IoU consistency recorded in the evidence;
3. the live P1D12 proposal metadata read straight from the external diagnostics directories;
4. a mechanical replay of selected / bestEligible / bestAny with the frozen tie-break (-IoU, -confidence, proposal_id)
   and the resulting classification, compared against the canonical evidence.
"""

from __future__ import annotations

import hashlib
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
REQUIRED_KEYS = ("task", "starting_head", "branch", "scientific_reuse_disclosure", "verification_mode",
                 "detector_model_calls_this_task", "source_identity_basis", "external_identity",
                 "coverage_threshold", "historical_reproduction", "historical_iou_consistency", "candidates",
                 "class_counts", "overall_outcome", "dominant_next_blocker", "next_gate")
DISCLOSURE_START = "The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason"


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob(relative: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{PREFIX}{relative}"], capture_output=True).stdout


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

    print("== 1. Git-canonical module identity ==")
    for key, relative in (("detector", "buildreasonseg/runtime/detector.py"),
                          ("imageio", "buildreasonseg/runtime/imageio.py")):
        entry = entries.get(relative)
        blob = git_blob(relative)
        external = (EXTERNAL / relative).read_bytes() if (EXTERNAL / relative).is_file() else None
        ok = bool(entry and external is not None
                  and entry["sha256"] == sha_bytes(blob) == sha_bytes(external))
        recorded = evidence["external_identity"][key]
        recorded_ok = (recorded["manifest_sha256"] == entry["sha256"]
                       and recorded["external_sha256"] == sha_bytes(external)
                       and recorded["manifest_match"] is True)
        print(f"  {relative}: manifest==git==external={ok} evidence_record_consistent={recorded_ok}")
        if not (ok and recorded_ok):
            failures.append(f"identity:{key}")
    predict_source = (CANON / "predict.py").read_text(encoding="utf-8")
    standalone = [line for line in predict_source.splitlines() if re.match(r"^\s*import imageio\b", line)]
    uses_package = "buildreasonseg.runtime.imageio" in predict_source or "runtime.imageio" in predict_source
    print(f"  standalone third-party 'import imageio' lines = {len(standalone)} · delivery module referenced = {uses_package}")
    if standalone:
        failures.append("standalone-imageio-import")

    print("== 2. schema, disclosure and historical R6 evidence ==")
    missing = [key for key in REQUIRED_KEYS if key not in evidence]
    print(f"  required top-level keys present = {not missing} (missing {missing})")
    if missing:
        failures.append("schema")
    disclosure = evidence["scientific_reuse_disclosure"]
    text = disclosure.get("en", "") if isinstance(disclosure, dict) else str(disclosure)
    print(f"  disclosure verbatim present = {text.startswith(DISCLOSURE_START)} · chinese translation = "
          f"{bool(isinstance(disclosure, dict) and disclosure.get('zh'))}")
    if not text.startswith(DISCLOSURE_START):
        failures.append("disclosure")
    historical = evidence["historical_reproduction"]
    r6_ok = historical.get("r6_ten_field_all_match") is True and all(
        candidate["historical_reproduction"]["r6_ten_field_all_match"] is True for candidate in evidence["candidates"])
    consistency = evidence["historical_iou_consistency"]
    cons_ok = consistency.get("r4_r5_r6_within_tolerance") is True and consistency.get("tolerance") == TOLERANCE
    print(f"  R6 ten-field reproduction = {r6_ok} · R4=R5=R6 within {TOLERANCE:g} = {cons_ok}")
    if not (r6_ok and cons_ok):
        failures.append("historical")

    print("== 3. live P1D12 metadata + 4. mechanical replay ==")
    stored_iou = {relation: {int(p["proposal_id"]): float(p["iou_to_gt"])
                             for p in evidence["historical_iou_by_proposal"][relation]} for relation in RELATIONS}
    class_counts: dict[str, int] = {}
    for candidate in evidence["candidates"]:
        relation, tile = candidate["relation"], candidate["tile"]
        live = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = live.get("items") or live.get("proposals") or []
        for item in items:
            item["_iou"] = stored_iou[relation].get(int(item["proposal_id"]), 0.0)
            item["_eligible"] = (int(item.get("mask_area", 0)) > 0
                                 and item.get("touches_image_border") is False
                                 and float(item.get("bbox_extent_ratio", 1.0)) <= 0.20)
        key = lambda item: (-item["_iou"], -float(item.get("confidence", 0.0)), int(item["proposal_id"]))
        eligible = [i for i in items if i["_eligible"]]
        best_eligible = min(eligible, key=key) if eligible else None
        best_any = min(items, key=key) if items else None
        reported_ids = (candidate["best_eligible"]["proposal_id"], candidate["best_any"]["proposal_id"])
        replay_ids = (None if best_eligible is None else int(best_eligible["proposal_id"]),
                      None if best_any is None else int(best_any["proposal_id"]))
        selected_iou = float(candidate["selected"]["iou"])
        best_eligible_iou = 0.0 if best_eligible is None else float(best_eligible["_iou"])
        best_any_iou = 0.0 if best_any is None else float(best_any["_iou"])
        classification = classify(selected_iou, best_eligible_iou, best_any_iou)
        class_counts[classification] = class_counts.get(classification, 0) + 1
        ids_ok = reported_ids == replay_ids
        class_ok = classification == candidate["classification"]
        iou_ok = (abs(best_eligible_iou - float(candidate["best_eligible"]["iou"])) <= TOLERANCE
                  and abs(best_any_iou - float(candidate["best_any"]["iou"])) <= TOLERANCE)
        print(f"  {relation}: live_proposals={len(items)} eligible={len(eligible)} ids_match={ids_ok} "
              f"iou_match={iou_ok} class_match={class_ok} ({classification})")
        if not (ids_ok and class_ok and iou_ok):
            failures.append(f"replay:{relation}")
    counts_ok = class_counts == evidence["class_counts"]
    print(f"  class counts recomputed = {json.dumps(class_counts)} · matches evidence = {counts_ok}")
    if not counts_ok:
        failures.append("class_counts")

    print("STANDALONE_VERIFIER:", "PASS" if not failures else f"FAIL {failures}")
    print("detector_or_model_calls = 0 · canonical evidence untouched = true")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
