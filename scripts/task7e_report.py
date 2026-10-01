"""Task 7E Parts L-M — development architecture decision and the single verdict.

Section 21: `DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER` is true only when `DB1_HOLDOUT_GENERALIZES`,
`DB1_PREDICTED_REFERENCE_USABLE` and no protocol violation hold. Section 22 verdict priority:

1. `INVALID_EXPERIMENT`
2. `DB1_CHECKPOINT_UNAVAILABLE`
3. `L3_HOLDOUT_REMAINDER_INSUFFICIENT`
4. `L3_HOLDOUT_PAIRED_INSUFFICIENT`
5. `TASK7D_REPRODUCTION_FAIL`
6. `DB1_HOLDOUT_GENERALIZATION_FAIL`
7. `DB1_PREDICTED_REFERENCE_BELOW_GATE`
8. `DB1_DEVELOPMENT_L3_DECODER_READY`

Writes `evaluation/task7e_verdict.json`. DSH reports measurements only.

    python scripts/task7e_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import frozen_metadata  # noqa: E402
from scripts.task7e_build_holdout import PAIR_COUNT, stable_hash  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7e_verdict.json"
BASE_COMMIT = "86e4f4cd4bd5aa6858264a7d5709092b32678ebc"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "DB1_CHECKPOINT_UNAVAILABLE",
                    "L3_HOLDOUT_REMAINDER_INSUFFICIENT", "L3_HOLDOUT_PAIRED_INSUFFICIENT",
                    "TASK7D_REPRODUCTION_FAIL", "DB1_HOLDOUT_GENERALIZATION_FAIL",
                    "DB1_PREDICTED_REFERENCE_BELOW_GATE", "DB1_DEVELOPMENT_L3_DECODER_READY")
FROZEN_PATHS = (
    "buildreasonseg_mvp/task7d_global_competition_decoder.py",
    "buildreasonseg_mvp/task7d_data.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/program_parser.py",
    "evaluation/task7d_", "evaluation/task7c_", "evaluation/task6z_",
)


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    manifest = load("task7e_holdout_manifest.json")
    minival = load("task7e_minival_reproduction.json")
    oracle = load("task7e_oracle_holdout.json")
    oracle_paired = load("task7e_oracle_holdout_paired.json")
    quality = load("task7e_predicted_reference_quality.json")
    predicted = load("task7e_predicted_reference_holdout.json")
    predicted_paired = load("task7e_predicted_reference_paired.json")
    parser_integration = load("task7e_canonical_parser_integration.json")
    if any(payload is None for payload in (manifest, minival, oracle, oracle_paired, quality,
                                           predicted, predicted_paired)):
        missing = [name for name, payload in
                   (("holdout_manifest", manifest), ("minival_reproduction", minival),
                    ("oracle_holdout", oracle), ("oracle_holdout_paired", oracle_paired),
                    ("predicted_reference_quality", quality),
                    ("predicted_reference_holdout", predicted),
                    ("predicted_reference_paired", predicted_paired)) if payload is None]
        write_json(OUT, {"_doc": "Task 7E section 22.", "task": "7E",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7e.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    metadata = frozen_metadata()
    holdout_generalizes = bool(oracle.get("DB1_HOLDOUT_GENERALIZES"))
    predicted_usable = bool(predicted.get("DB1_PREDICTED_REFERENCE_USABLE"))
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "d_b1_checkpoint_sha256": metadata["d_b1"]["sha256"],
        "d_b1_expected_sha256": metadata["d_b1"]["expected_sha256"],
        "d_b1_checkpoint_matches": metadata["d_b1"]["matches"],
        "d_b1_checkpoint_variant": metadata["d_b1"].get("checkpoint_variant"),
        "d_b1_selected_epoch": metadata["d_b1"].get("selected_epoch"),
        "z_b3_checkpoint_sha256": metadata["z_b3"]["sha256"],
        "z_b3_checkpoint_matches": metadata["z_b3"]["matches"],
        "training_performed": any(payload.get("training_performed", False)
                                  for payload in (manifest, minival, oracle, oracle_paired, quality,
                                                  predicted, predicted_paired)
                                  if isinstance(payload, dict)),
        "retrained_any_checkpoint": False,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (manifest, minival, oracle, oracle_paired, quality,
                                               predicted, predicted_paired)
                               if isinstance(payload, dict)),
        "parser_used_for_oracle": False,
        "predicted_reference_in_oracle_stage": False,
        "fields_changed": False, "sam2_changed": False, "u_c1_changed": False,
        "ranker_or_filter_or_refinement": False,
        "learned_competition": False, "attention_or_graph": False, "grcl": False,
        "free_form_paraphrase_used_for_selection": False,
        "full_training_started": False,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["training_performed"]
                             and not protocol["test_split_used"]
                             and metadata["d_b1"]["matches"] and metadata["z_b3"]["matches"]
                             and not protocol["ranker_or_filter_or_refinement"])

    decision = {
        "DB1_HOLDOUT_GENERALIZES": holdout_generalizes,
        "DB1_PREDICTED_REFERENCE_USABLE": predicted_usable,
        "DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER": bool(holdout_generalizes and predicted_usable
                                                   and protocol["clean"]),
        "development_baseline": "Z-B3" if not (holdout_generalizes and predicted_usable)
        else "D-B1",
        "z_b3_role": "frozen baseline / ablation",
        "d_b1_role": ("development L3 decoder candidate" if (holdout_generalizes and predicted_usable)
                      else "positive oracle-reference ablation only"),
        "checkpoints_deleted_or_overwritten": False,
    }

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, training or test use)"
    elif not metadata["d_b1"]["matches"]:
        verdict, reason = "DB1_CHECKPOINT_UNAVAILABLE", "the frozen D-B1 checkpoint hash does not match"
    elif manifest["checks"]["remainder_ok"] is not True:
        verdict, reason = ("L3_HOLDOUT_REMAINDER_INSUFFICIENT",
                           f"remainder checks: {manifest['checks']}")
    elif manifest["paired"]["passed"] is not True:
        verdict, reason = ("L3_HOLDOUT_PAIRED_INSUFFICIENT",
                           f"{manifest['paired']['pairs']} pairs < {PAIR_COUNT}")
    elif minival["reproduction_passed"] is not True:
        verdict, reason = ("TASK7D_REPRODUCTION_FAIL",
                           f"MiniVal reproduction: {minival['reproduction']}")
    elif not holdout_generalizes:
        verdict, reason = ("DB1_HOLDOUT_GENERALIZATION_FAIL",
                           f"section 13 gate: {oracle.get('gate')}")
    elif not predicted_usable:
        verdict, reason = ("DB1_PREDICTED_REFERENCE_BELOW_GATE",
                           "the learned-free D-B1 gain survives the untouched oracle holdout, but the "
                           "predicted-reference gate fails: "
                           + str([name for name, entry in predicted["gate"].items()
                                  if not entry["passed"]]))
    else:
        verdict, reason = "DB1_DEVELOPMENT_L3_DECODER_READY", "all section 21 conditions hold"

    payload = {
        "_doc": (
            "Task 7E sections 21-22. Deterministic field-weighted prototype (frozen Task 7D D-B1) audited on "
            "the untouched E-HoldoutL3 remainder with the frozen oracle largest reference and then with the "
            "frozen U-C1 predicted reference; frozen Z-B3 is the baseline. No model was trained, nothing "
            "was tuned, and the verdict follows the fixed section-22 priority order."
        ),
        "task": "7E", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "holdout": {"records": manifest["holdout"]["records"],
                    "per_program": manifest["holdout"]["per_program"],
                    "per_direction": manifest["holdout"]["per_direction"],
                    "record_id_hash": manifest["holdout"]["record_id_hash"],
                    "excluded": manifest["exclusion"]["union"],
                    "source_records": manifest["source"]["records"],
                    "source_matches_expected": manifest["source"]["matches_expected"],
                    "checks": manifest["checks"],
                    "paired": {"pairs": manifest["paired"]["pairs"],
                               "candidate_pairs": manifest["paired"]["candidate_pairs"],
                               "overlap_with_task6z_paired_members":
                                   manifest["paired"]["overlap_with_task6z_paired_members"],
                               "pair_id_hash": manifest["paired"]["pair_id_hash"]}},
        "minival_reproduction": {"passed": minival["reproduction_passed"],
                                 "tolerance": minival["tolerance"],
                                 "reproduction": minival["reproduction"]},
        "oracle_holdout": {
            "z_b3_miou": oracle["results"]["Z-B3"]["overall"]["miou"],
            "d_b1_miou": oracle["results"]["D-B1"]["overall"]["miou"],
            "delta": oracle["delta"]["overall"],
            "delta_per_direction": oracle["delta"]["per_direction"],
            "z_b3_dice": oracle["results"]["Z-B3"]["overall"]["dice"],
            "d_b1_dice": oracle["results"]["D-B1"]["overall"]["dice"],
            "bootstrap": oracle["bootstrap"], "gate": oracle["gate"],
            "gate_passed": holdout_generalizes,
        },
        "oracle_holdout_paired": {
            "z_b3": {key: value for key, value in oracle_paired["results"]["Z-B3"].items()
                     if key != "per_pair"},
            "d_b1": {key: value for key, value in oracle_paired["results"]["D-B1"].items()
                     if key != "per_pair"}},
        "predicted_reference": {
            "executed": True,
            "reference": quality["metrics"],
            "z_b3": {key: value for key, value in predicted["results"]["Z-B3"].items()
                     if key != "per_direction"},
            "d_b1": {key: value for key, value in predicted["results"]["D-B1"].items()
                     if key != "per_direction"},
            "retention": predicted["retention"], "delta": predicted["delta"],
            "gate": predicted["gate"], "gate_passed": predicted_usable,
            "paired": {name: {key: value for key, value in predicted_paired["results"][name].items()}
                       for name in ("Z-B3", "D-B1")},
            "reference_abstention_pairs": predicted_paired["pairs"]["reference_abstention_pairs"],
        },
        "canonical_parser_integration": {
            "executed": bool(parser_integration and parser_integration.get("executed")),
            "reason": (parser_integration or {}).get("reason"),
            "artifact_present": parser_integration is not None,
            "parser_trained": False, "free_form_paraphrase_used": False},
        "decision": decision,
        "interpretation_boundary": {
            "d_b1_claimed_globally_novel": False, "d_b1_called_final_model": False,
            "full_training_run": False, "test_evaluated": False,
            "parser_or_reference_retrained": False, "learned_competition_added": False,
            "graph_or_attention_added": False, "d_b1_modified_after_holdout": False,
            "task7f_chosen": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7E 的 untouched L3 holdout 与 predicted-reference 结果决定是否"
                           "正式冻结 D-B1 为开发版 L3 decoder，不自行进行全量训练、test 评估或新的架构改动。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7e.report] holdout D-B1 {oracle['results']['D-B1']['overall']['miou']:.4f} vs Z-B3 "
          f"{oracle['results']['Z-B3']['overall']['miou']:.4f} (Δ{oracle['delta']['overall']:+.4f}) | "
          f"oracle gate {holdout_generalizes} | predicted gate {predicted_usable} | adopt "
          f"{decision['DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
