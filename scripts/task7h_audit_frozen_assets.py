"""Task 7H Part B — audit the frozen development assets and build the evidence registry.

Read-only audit of already-authorized local assets: every required module/checkpoint is hashed and recorded
with its role, task of origin, evidence population, verified result and claim boundary. Rejected support
modules are recorded as frozen **negative** evidence only. The test split is never read (its identity is
recorded as present-but-not-read).

Writes `evaluation/task7h_evidence_registry.json`.

    python scripts/task7h_audit_frozen_assets.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7h_evidence_registry.json"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
NATIVE_VIEW = REPO_ROOT / "artifacts" / "whu_native_vector" / "reasoning_view" / "scene_disjoint_v1"
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
EXPECTED_L3_TRAIN = {"largest_to_left_of_to_nearest": 323, "largest_to_right_of_to_nearest": 347,
                     "largest_to_above_to_nearest": 338, "largest_to_below_to_nearest": 336}
EXPECTED = {
    "yolo": "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474",
    "parser": "c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a",
    "z_b3": "74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc",
    "d_b1": "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0",
    "n_b3": "7556e4a4862b75d47e61b5c6391c2a05689d5bde3e74586aed1d94a1c3c7d6ab",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Path, expected: str | None = None) -> dict:
    entry = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
    if expected is not None:
        entry["expected_sha256"] = expected
        entry["matches"] = entry.get("sha256") == expected
    return entry


def artifact_verdict(name: str) -> str | None:
    path = EVAL / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("verdict")
    except ValueError:  # pragma: no cover - defensive
        return None


def jsonl_rows(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    yolo = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
        / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
    parser = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
    z_b3 = REPO_ROOT / "artifacts" / "checkpoints" / "task6z" / "zb3_minitrain1200.pt"
    d_b1 = REPO_ROOT / "artifacts" / "checkpoints" / "task7d" / "db1_minitrain1200.pt"
    n_b3 = REPO_ROOT / "artifacts" / "task6o" / "checkpoints" / "o2_B3.pt"

    modules = {
        "direction_field": REPO_ROOT / "buildreasonseg_mvp" / "geometric_relation_field_v02.py",
        "nearest_field": REPO_ROOT / "buildreasonseg_mvp" / "nearest_boundary_field.py",
        "l3_decoder_z_b3": REPO_ROOT / "buildreasonseg_mvp" / "task6z_l3_decoder.py",
        "d_b1_decoder": REPO_ROOT / "buildreasonseg_mvp" / "task7d_global_competition_decoder.py",
        "reference_resolver": REPO_ROOT / "buildreasonseg_mvp" / "task6q_reference_resolver.py",
        "program_parser": REPO_ROOT / "buildreasonseg_mvp" / "program_parser.py",
        "u_c1_config": REPO_ROOT / "scripts" / "task6u_common.py",
    }

    train_rows = jsonl_rows(DATA / "train.jsonl") or []
    val_rows = jsonl_rows(DATA / "val.jsonl") or []
    l3_train = Counter(str(row["query_type"]) for row in train_rows
                       if str(row["query_type"]) in L3_PROGRAMS)
    l3_val = Counter(str(row["query_type"]) for row in val_rows
                     if str(row["query_type"]) in L3_PROGRAMS)
    train_counts = {program: l3_train.get(program, 0) for program in L3_PROGRAMS}
    val_counts = {program: l3_val.get(program, 0) for program in L3_PROGRAMS}

    checkpoint_entries = {
        "yolo_task6m1": {**file_entry(yolo, EXPECTED["yolo"]),
                         "role": "frozen proposal generator (U-C1)",
                         "task_of_origin": "6M.1",
                         "config": {"imgsz": 640, "conf": 0.05, "max_det": 300, "nms": "default",
                                    "tta": False, "tiling": False},
                         "frozen": True,
                         "evidence_population": "Task 6U/6Z/7F/7G proposal caches (val tiles)",
                         "verified_result": "proposal source for every reference experiment",
                         "claim_boundary": "proposal quality is a measured bottleneck carrier; "
                                           "YOLO was never retrained"},
        "parser_task7c": {**file_entry(parser, EXPECTED["parser"]),
                          "role": "controlled-language 20-class ProgramHead (development front end)",
                          "task_of_origin": "7C",
                          "frozen": True,
                          "evidence_population": "full v0.2 val 18,222 prompts + Task 7A/7B "
                                                 "paraphrase packs",
                          "verified_result": {"canonical_full_val_accuracy": 1.0,
                                              "canonical_macro_f1": 1.0,
                                              "every_class_recall": 1.0,
                                              "z_minival240": "240/240",
                                              "z_paired_members": "40/40",
                                              "fixed24": "5/24", "fixed24_compact": "0/8",
                                              "minimal96": "60/96", "stress_accuracy": 0.6667},
                          "claim_boundary": "canonical/program-template behaviour verified; free-form "
                                            "L3 paraphrase robustness NOT solved"},
        "z_b3_task6z": {**file_entry(z_b3, EXPECTED["z_b3"]),
                        "role": "frozen L3 target-decoder baseline/ablation",
                        "task_of_origin": "6Z",
                        "frozen": True,
                        "evidence_population": "Z-MiniVal240, Z-PairedVal20, Task 7E E-HoldoutL3 (669)",
                        "verified_result": {"minival_miou": 0.3242128982543474,
                                            "holdout_oracle_miou": 0.3141113773177512,
                                            "holdout_paired": "15/20"},
                        "claim_boundary": "baseline/ablation only; not the preferred L3 decoder"},
        "d_b1_task7d": {**file_entry(d_b1, EXPECTED["d_b1"]),
                        "role": "preferred L3 target-decoder architecture candidate",
                        "task_of_origin": "7D",
                        "frozen": True,
                        "evidence_population": "Task 7E untouched oracle-reference E-HoldoutL3 "
                                               "(669 records) + E-PairedHoldout20",
                        "verified_result": {"minival_miou": 0.3978996298363562,
                                            "holdout_oracle_miou": 0.38549570532647004,
                                            "holdout_delta_over_z_b3": 0.07138432800871886,
                                            "holdout_bootstrap_ci": [0.058623364793963954,
                                                                     0.08426264360286147],
                                            "holdout_paired": "18/20",
                                            "holdout_margin": 0.3193402994,
                                            "practical_predicted_reference_strict_miou":
                                                0.24540501038500215},
                        "claim_boundary": "preferred architecture candidate pending formal three-seed "
                                          "retraining and the locked test evaluation; NOT end-to-end ready"},
        "n_b3_task6o": {**file_entry(n_b3, EXPECTED["n_b3"]),
                        "role": "verified directional-field visual segmentation component "
                                "(L2 directional path)",
                        "task_of_origin": "6O",
                        "frozen": True,
                        "evidence_population": "Task 6O MiniVal240 (field causal decomposition)",
                        "verified_result": {"minival_miou": 0.4299680351479113,
                                            "source": "evaluation/task6o_o2_training.json"},
                        "claim_boundary": "field-guided visual segmentation supported; universal relation "
                                          "reasoning is NOT claimed; N-B2/N-B4 must not be substituted"},
    }

    negative_entries = {
        "task6p_reference_mask_head": {
            "role": "rejected reference-support module (frozen negative evidence)",
            "task_of_origin": "6P", "frozen": True, "adopted": False,
            "artifact": "evaluation/task6p_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task6p_verdict.json")},
            "claim_boundary": "must not appear in the development chain"},
        "task6u_proposal_set_ranker_v01": {
            "role": "rejected reference ranker (frozen negative evidence)",
            "task_of_origin": "6U", "frozen": True, "adopted": False,
            "checkpoint": file_entry(REPO_ROOT / "artifacts" / "checkpoints" / "task6u"
                                    / "reference_ranker_v01.pt"),
            "artifact": "evaluation/task6u_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task6u_verdict.json")},
            "claim_boundary": "must not appear in the development chain"},
        "task6w_proposal_quality_estimator_v01": {
            "role": "rejected proposal quality filter (frozen negative evidence)",
            "task_of_origin": "6W", "frozen": True, "adopted": False,
            "checkpoint": file_entry(REPO_ROOT / "artifacts" / "checkpoints" / "task6w"
                                    / "proposal_quality_v01.pt"),
            "artifact": "evaluation/task6w_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task6w_verdict.json")},
            "claim_boundary": "must not appear in the development chain"},
        "task6x_sam2_proposal_refinement": {
            "role": "rejected SAM2 mask refinement (frozen negative evidence)",
            "task_of_origin": "6X", "frozen": True, "adopted": False,
            "artifact": "evaluation/task6x_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task6x_verdict.json")},
            "claim_boundary": "must not appear in the development chain"},
        "task7g_set_context_largest_selector_v1": {
            "role": "rejected learned largest-reference selector (frozen negative evidence)",
            "task_of_origin": "7G", "frozen": True, "adopted": False,
            "checkpoint": file_entry(REPO_ROOT / "artifacts" / "checkpoints" / "task7g"
                                    / "largest_set_context_selector_v1.pt"),
            "artifact": "evaluation/task7g_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task7g_verdict.json"),
                                "internal_gate_passed": False,
                                "internal_mean_selected_iou": 0.6198877725708375,
                                "external_scene_disjoint_run": False},
            "claim_boundary": "internal gate failed; the external scene-disjoint stage never ran, so no "
                              "scene-disjoint failure or success may be claimed"},
        "task7d_d_b2_learned_global_competition": {
            "role": "rejected learned competition decoder (frozen negative evidence)",
            "task_of_origin": "7D", "frozen": True, "adopted": False,
            "artifact": "evaluation/task7d_verdict.json",
            "verified_result": {"verdict": artifact_verdict("task7d_verdict.json"),
                                "minival_miou": 0.3264764207334784,
                                "target_mass": 0.0067, "argmax_in_target": 0.0, "entropy": 0.8848},
            "claim_boundary": "must not appear in the development chain"},
    }

    dataset = {
        "name": "BuildSpatialReason", "version": "v0.2", "level_scope": "L1-L3",
        "native_vector": {"name": "WHU-EA-NativeVector", "version": "v1.0",
                          "source_dataset": "WHU Building Dataset",
                          "source_subset": "Satellite dataset II (East Asia)"},
        "split_view": "scene_disjoint_v1",
        "splits": {
            "train": {**file_entry(DATA / "train.jsonl"), "rows": len(train_rows),
                      "l3_programs": train_counts, "l3_total": sum(train_counts.values()),
                      "expected_l3_total": 1344,
                      "matches_expected": train_counts == EXPECTED_L3_TRAIN},
            "val": {**file_entry(DATA / "val.jsonl"), "rows": len(val_rows),
                    "l3_programs": val_counts, "l3_total": sum(val_counts.values()),
                    "expected_l3_total": 936, "matches_expected": sum(val_counts.values()) == 936},
            "test": {"path": str(DATA / "test.jsonl"), "exists": (DATA / "test.jsonl").is_file(),
                     "read": False, "hashed": False,
                     "note": "Task 7H does not read, hash or evaluate the test split"},
        },
        "native_view": {"path": str(NATIVE_VIEW), "exists": NATIVE_VIEW.is_dir(),
                        "metadata": {split: {"path": str(NATIVE_VIEW / "metadata" / f"{split}.jsonl"),
                                             "exists": (NATIVE_VIEW / "metadata"
                                                        / f"{split}.jsonl").is_file(),
                                             "read": split != "test"}
                                     for split in ("train", "val", "test")}},
        "claim_boundary": "the native split demonstrates raster/scene separation, NOT unseen-city "
                          "generalization",
    }

    required_hashes = {name: entry.get("matches") for name, entry in checkpoint_entries.items()}
    payload = {
        "_doc": ("Task 7H section 3. Read-only evidence registry for the frozen BuildReasonSeg development "
                 "architecture: data identity, proposal model, parser, relation fields, directional L2 "
                 "decoder, L3 baseline, preferred L3 decoder and the rejected support modules recorded as "
                 "frozen negative evidence. No training, no test access."),
        "task": "7H", "stage": "B-evidence-registry",
        "architecture_name": "BuildReasonSeg-DevFreeze-2026-10",
        "data": dataset,
        "checkpoints_and_modules": {
            **checkpoint_entries,
            "modules": {name: {**file_entry(path),
                               "role": role}
                        for (name, path), role in zip(
                            modules.items(),
                            ("directional relation field v0.2 (field source)",
                             "nearest boundary field v0.1 (field source)",
                             "Z-B3 L3 decoder architecture source",
                             "D-B1 decoder architecture source",
                             "Task 6Q deterministic reference resolver source",
                             "20-class ProgramHead model source",
                             "frozen U-C1 proposal configuration source"))},
        },
        "rejected_negative_evidence": negative_entries,
        "hash_checks": {"required": required_hashes,
                        "all_required_match": all(value is True
                                                  for value in required_hashes.values())},
        "development_chain": [
            "controlled instruction", "Task 7C ProgramHead (canonical 20-class ids)",
            "U-C1 YOLO26m-seg proposals", "deterministic largest reference selector",
            "GeometricRelationField v0.2 (P_dir) + NearestBoundaryField v0.1 (P_near)",
            "Task 7D D-B1 target decoder", "target mask"],
        "excluded_from_development_chain": list(negative_entries),
        "training_performed": False, "test_split_used": False, "test_records_read": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = ("FROZEN_ASSETS_VERIFIED" if payload["hash_checks"]["all_required_match"]
                          else "FROZEN_ASSET_HASH_MISMATCH")
    write_json(OUT, payload)
    print("[7h.audit] required hashes: "
          + ", ".join(f"{name}={'ok' if value else 'MISMATCH'}"
                      for name, value in required_hashes.items()))
    print(f"[7h.audit] L3 train {sum(train_counts.values())} {train_counts} (expected 1344, "
          f"match {train_counts == EXPECTED_L3_TRAIN}) | L3 val {sum(val_counts.values())} "
          f"(expected 936) | negative evidence {len(negative_entries)} modules | "
          f"{payload['verdict']}", flush=True)
    return 0 if payload["hash_checks"]["all_required_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
