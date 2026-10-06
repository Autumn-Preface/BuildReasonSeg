"""Task 8D offline forensics. Saved automatic maps are never recomputed.

Only the explicitly authorized GT-reference deterministic fields are recomputed.
No model construction, checkpoint loading, subprocess execution or RC1 writes.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
TASK_ID = "TASK8D_ABOVE_TARGET_CHAIN_DIAGNOSIS_V1"
ACCEPTED_HEAD = "764a805ef7d9651bfffd27dfaeb5808e8c21df1f"
SAMPLE = "buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314"
PROGRAM = "largest_to_above_to_nearest"
RUNTIME_SHA = "2fcbb705acf1ceeb28fc9609bfda869150e7b800b2bea0d9fba9a723c12ff8ca"
JSON_PATH = REPO / "evaluation/task8d_above_target_chain_diagnosis_v1.json"
REPORT_PATH = REPO / "docs/task8d_above_target_chain_diagnosis_v1.md"
FIGURE_ROOT = REPO / "evaluation/task8d_above_target_chain_diagnosis_v1"
CANONICAL = REPO / "delivery_src/BuildReasonSeg_Advisor_RC1"
RUNTIME_SOURCE = "delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/"
STAGES = ("W", "A", "C", "logits", "probability", "final_mask")
STAGE_KEYS = {"P_dir": "P_dir_mean", "P_near": "P_near_mean", "W": "W_mass",
              "A": "A_attention_mass", "C": "C_mean", "logits": "logit_mean",
              "probability": "probability_mean", "positive_logit_fraction": "positive_logit_fraction",
              "final_mask": "final_mask_iou"}
MAP_SHAPES = {k: (64, 64) for k in ("P_dir", "P_near", "W", "A", "C")}
MAP_SHAPES.update({"logits": (512, 512), "probability": (512, 512)})
FIGURES = ("01_reference_and_gt.png", "02_P_dir.png", "03_P_near.png", "04_W_and_A.png",
           "05_C.png", "06_logits_probability.png", "07_final_mask_vs_gt.png",
           "08_stage_trajectory.png", "contact_sheet.png")


def identity(path: Path) -> dict:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for b in iter(lambda: handle.read(1 << 20), b""):
            h.update(b)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def verify_identities(records: list[dict]) -> None:
    for row in records:
        if identity(Path(row["path"])) != row:
            raise ValueError("frozen identity changed: " + row["path"])


def lock_above(task8c: dict) -> dict:
    rows = [r for r in task8c["cases"] if r["relation"] == "above"]
    if len(rows) != 1:
        raise ValueError("ABOVE sample not unique")
    row = rows[0]
    if (row["sample_id"] != SAMPLE or row["expected_program"] != PROGRAM or
            row["runtime_status"] != "SUCCESS" or row["reference_mode"] != "automatic" or
            row["reference_id"] != 5 or row["mask_area"] != 1580 or
            task8c["formal_runner_invocation_count"] != 1 or task8c["real_candidates_attempted"] != 4 or
            task8c["frozen_runtime_identity"]["sha256"] != RUNTIME_SHA or not task8c["no_inference_retry"]):
        raise ValueError("ABOVE lock differs")
    audit = row["gt_audit"]
    expected = {"canonical_reference_id": 4, "canonical_target_id": 6,
                "selected_reference_best_gt_instance_id": 4,
                "selected_reference_iou_with_canonical_gt_reference": 0.9097432024169184,
                "predicted_target_best_gt_instance_id": 7,
                "predicted_target_best_gt_iou": 0.48158096699923253,
                "target_iou": 0.07762201453790239, "target_dice": 0.14406167188629246,
                "semantic_chain_status": "TARGET_IDENTITY_MISMATCH"}
    if any(audit[k] != v for k, v in expected.items()):
        raise ValueError("GT lock differs")
    if identity(Path(row["image"]))["sha256"] != row["image_sha256"]:
        raise ValueError("frozen TIFF identity differs")
    verify_identities([task8c["frozen_runtime_identity"]] + row["artifacts"] +
                      [row["transcript_identity"]] + audit["truth_inputs"])
    actual = {str(p.resolve()) for p in Path(row["run_root"]).rglob("*") if p.is_file()}
    if actual != {a["path"] for a in row["artifacts"]}:
        raise ValueError("ABOVE run-root file set changed")
    return row


def load_maps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(MAP_SHAPES):
            raise ValueError("unexpected frozen map keys")
        maps = {k: z[k].copy() for k in MAP_SHAPES}
    for k, v in maps.items():
        if v.shape != MAP_SHAPES[k] or not np.isfinite(v).all():
            raise ValueError("frozen map shape/finiteness conflict: " + k)
        v.setflags(write=False)
    return maps


def to_context(mask: np.ndarray, context: dict) -> np.ndarray:
    """Inverse of frozen crop-to-global mapping; GT is zero in reflection padding."""
    left, top = map(int, context["origin"])
    size = int(context["size"])
    h, w = mask.shape
    y0, x0, y1, x1 = max(0, top), max(0, left), min(h, top + size), min(w, left + size)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("context has no valid tile extent")
    output = np.zeros((size, size), dtype=bool)
    output[y0-top:y1-top, x0-left:x1-left] = mask[y0:y1, x0:x1]
    return output


def fractional_occupancy(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (512, 512) or mask.dtype != np.bool_:
        raise ValueError("exact 512 binary context mask required")
    return cv2.resize(mask.astype(np.float64), (64, 64), interpolation=cv2.INTER_AREA)


def mass_mean(values: np.ndarray, occupancy: np.ndarray) -> tuple[float, float | None]:
    if values.shape != occupancy.shape:
        raise ValueError("map/occupancy alignment conflict")
    mass = float(np.sum(values.astype(np.float64) * occupancy, dtype=np.float64))
    area = float(occupancy.sum(dtype=np.float64))
    return mass, mass / area if area else None


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    return intersection / union if union else 0.0


def stage_rows(maps: dict, masks: dict, context: dict, final_mask: np.ndarray,
               geometry_rows: dict) -> tuple[list[dict], dict]:
    rows, context_masks = [], {}
    for instance_id in sorted(masks):
        truth = masks[instance_id]
        ctx = to_context(truth, context)
        context_masks[instance_id] = ctx
        occ = fractional_occupancy(ctx)
        row = {"instance_id": instance_id, "GT_area": int(truth.sum()),
               "context_GT_area": int(ctx.sum()), "context_coverage_fraction": float(ctx.sum()/truth.sum()),
               "fractional_occupancy_sum": float(occ.sum()), "inside_context": bool(ctx.any()),
               **geometry_rows[instance_id]}
        for k in ("P_dir", "P_near", "W"):
            row[k + "_mass"], row[k + "_mean"] = mass_mean(maps[k], occ)
        row["A_attention_mass"], _ = mass_mean(maps["A"], occ)
        _, row["C_mean"] = mass_mean(maps["C"], occ)
        row["logit_mean"] = float(maps["logits"][ctx].astype(np.float64).mean()) if ctx.any() else None
        row["probability_mean"] = float(maps["probability"][ctx].astype(np.float64).mean()) if ctx.any() else None
        row["positive_logit_fraction"] = float((maps["logits"][ctx] > 0).mean()) if ctx.any() else None
        row["final_mask_iou"] = iou(final_mask, truth)
        rows.append(row)
    return rows, context_masks


def pair_comparison(rows: list[dict], stages=tuple(STAGE_KEYS)) -> list[dict]:
    by_id = {r["instance_id"]: r for r in rows}
    output = []
    for stage in stages:
        key = STAGE_KEYS[stage]
        a, b = by_id[6][key], by_id[7][key]
        delta = None if a is None or b is None else float(a-b)
        favoured = "NOT_ESTABLISHED" if delta is None else (6 if delta > 0 else 7 if delta < 0 else "TIE")
        output.append({"stage": stage, "statistic": key, "score_target6": a, "score_wrong7": b,
                       "delta_6_minus_7": delta, "which_instance_is_favoured": favoured})
    return output


def rank_trajectory(rows: list[dict], stages=STAGES) -> dict:
    output = {}
    for stage in stages:
        key = STAGE_KEYS[stage]
        eligible = [r for r in rows if r["inside_context"] and r[key] is not None]
        ranked = sorted(eligible, key=lambda r: (-r[key], r["instance_id"]))
        ids = [r["instance_id"] for r in ranked]
        output[stage] = {"statistic": key, "ordered_instance_ids": ids,
                         "top5_instance_ids": ids[:5], "top_instance_id": ids[0] if ids else None,
                         "target6_rank": ids.index(6)+1 if 6 in ids else None,
                         "wrong7_rank": ids.index(7)+1 if 7 in ids else None,
                         "scores": [{"instance_id": r["instance_id"], "score": r[key], "rank": i+1}
                                    for i, r in enumerate(ranked)]}
    return output


def first_divergence(pair: list[dict], ranks: dict) -> dict:
    by_stage = {r["stage"]: r for r in pair}
    for stage in STAGES:
        if by_stage[stage]["which_instance_is_favoured"] == 7:
            top = ranks[stage]["top_instance_id"]
            return {"stage": stage, "statistic": STAGE_KEYS[stage], "kind": "FIRST_OBSERVED_DIVERGENCE",
                    "actual_top_instance_id": top, "neither6_nor7_is_top": top not in (6, 7),
                    "delta_6_minus_7": by_stage[stage]["delta_6_minus_7"],
                    "interpretation": "Earliest fixed-order numerical preference, not a causal proof."}
    return {"stage": "NOT_OBSERVED", "kind": "UNRESOLVED"}


class OfflineGuard:
    """Block inference/processes and writes except exact Task8D output paths."""
    def __init__(self):
        self.stack = ExitStack()
        self.read_modes = []

    def __enter__(self):
        import subprocess
        import torch
        original_import = builtins.__import__
        denied = ("ultralytics", "transformers", "sam2", "buildreasonseg.runtime.detector",
                  "buildreasonseg.runtime.core", "buildreasonseg.runtime.pipeline",
                  "buildreasonseg_mvp.qwen", "buildreasonseg_mvp.task6n_relation_decoder")
        def guarded_import(name, *args, **kwargs):
            if any(name == v or name.startswith(v + ".") for v in denied):
                raise RuntimeError("offline-only: forbidden import " + name)
            return original_import(name, *args, **kwargs)
        def denied_call(*args, **kwargs):
            raise RuntimeError("offline-only: inference/checkpoint/process execution forbidden")
        self.stack.enter_context(patch.object(builtins, "__import__", guarded_import))
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            self.stack.enter_context(patch.object(subprocess, name, denied_call))
        for obj, name in ((torch, "load"), (torch.jit, "load"), (torch.hub, "load"), (torch.nn.Module, "__init__")):
            self.stack.enter_context(patch.object(obj, name, denied_call))
        for obj in (builtins, io):
            original = obj.open
            def guarded_open(file, mode="r", *args, _original=original, **kwargs):
                if not isinstance(file, int):
                    p = Path(file).resolve()
                    if any(v in mode for v in ("w", "a", "x", "+")):
                        require_output(p)
                    else:
                        self.read_modes.append({"path": str(p), "mode": mode})
                return _original(file, mode, *args, **kwargs)
            self.stack.enter_context(patch.object(obj, "open", guarded_open))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def require_output(path: Path) -> Path:
    path = Path(path).resolve()
    allowed = {JSON_PATH.resolve(), REPORT_PATH.resolve()} | {(FIGURE_ROOT / n).resolve() for n in FIGURES}
    if path not in allowed:
        raise PermissionError("diagnostic write outside authorized output set: " + str(path))
    return path


def canonical_reference_fields(reference: np.ndarray) -> dict:
    """CPU tensors only, exact frozen field functions; no neural module constructed."""
    if str(CANONICAL) not in sys.path:
        sys.path.insert(0, str(CANONICAL))
    from buildreasonseg.runtime._frozen.mvp.task6z_field_composition import record_fields
    from buildreasonseg.runtime._frozen.mvp.task7d_global_competition_decoder import field_competition
    bundle = record_fields(reference, PROGRAM)
    attention, _ = field_competition(bundle["P_dir_64"][None, None], bundle["P_near_64"][None, None])
    return {"P_dir": bundle["P_dir_64"].detach().numpy(),
            "P_near": bundle["P_near_64"].detach().numpy(),
            "W": bundle["P_prod_64"].detach().numpy(), "A": attention[0, 0].detach().numpy()}


def counterfactual_interpretation(automatic_pair: list[dict], gt_pair: list[dict]) -> dict:
    auto = {r["stage"]: r["which_instance_is_favoured"] for r in automatic_pair}
    gt = {r["stage"]: r["which_instance_is_favoured"] for r in gt_pair}
    if all(gt[k] == 7 for k in ("W", "A")):
        return {"interpretation": "NOT_SOLE_EXPLANATION", "preference_restored": False,
                "interpretation_detail": "GT-reference fields still favour wrong7; reference-shape residual cannot be the sole geometric explanation."}
    if all(auto[k] == 7 and gt[k] == 6 for k in ("W", "A")):
        return {"interpretation": "PLAUSIBLE_CONTRIBUTOR", "preference_restored": True,
                "interpretation_detail": "GT-reference fields restore target6 preference from wrong7; plausible contributor, not causal proof."}
    return {"interpretation": "MIXED / NOT RESOLVED", "preference_restored": False,
            "interpretation_detail": "No demonstrated restoration from wrong7. Automatic and GT-reference preferences must both be disclosed; no q/C/decoder counterfactual is available."}


def counterfactual(maps: dict, context_masks: dict) -> dict:
    gt = canonical_reference_fields(context_masks[4])
    rows = []
    for instance_id in sorted(context_masks):
        occupancy = fractional_occupancy(context_masks[instance_id])
        rows.append({"instance_id": instance_id, "inside_context": bool(context_masks[instance_id].any()),
                     "W_mass": mass_mean(gt["W"], occupancy)[0],
                     "A_attention_mass": mass_mean(gt["A"], occupancy)[0]})
    pair = pair_comparison(rows, ("W", "A"))
    differences = {}
    for k in gt:
        delta = gt[k].astype(np.float64)-maps[k].astype(np.float64)
        differences[k] = {"mean_signed_difference": float(delta.mean()),
                          "mean_absolute_difference": float(np.abs(delta).mean()),
                          "max_absolute_difference": float(np.abs(delta).max()),
                          "rms_difference": float(np.sqrt((delta**2).mean())),
                          "automatic_sum": float(maps[k].astype(np.float64).sum()),
                          "gt_reference_sum": float(gt[k].astype(np.float64).sum()),
                          "gt_reference_array_sha256": hashlib.sha256(gt[k].tobytes()).hexdigest()}
    auto_rows = [{"instance_id": i, "W_mass": mass_mean(maps["W"], fractional_occupancy(context_masks[i]))[0],
                  "A_attention_mass": mass_mean(maps["A"], fractional_occupancy(context_masks[i]))[0]}
                 for i in (6, 7)]
    interpretation = counterfactual_interpretation(pair_comparison(auto_rows, ("W", "A")), pair)
    return {"fields_recomputed": list(gt), "only_deterministic_fields": True,
            "C_recomputed": False, "q_recomputed": False, "decoder_called": False,
            "checkpoint_loads": 0, "automatic_maps_recomputed": False,
            "normalization": "Exact frozen field_competition: W/(sum(W)+EPS), EPS=1e-6; frozen invalid mass guard retained.",
            "task_shorthand_note": "W/sum(W) shorthand is recorded with the exact source EPS; no contract/normalization change.",
            "all_instance_W_A_table": rows, "target6_vs_wrong7": pair,
            "rank_trajectory": rank_trajectory(rows, ("W", "A")),
            "field_difference_statistics": differences, **interpretation,
            "causal_proof": False}


def canonical_geometry() -> tuple[dict, dict]:
    sys.path.insert(0, str(REPO / "spatial_reasoning"))
    from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset, image_record_for_reasoning
    import geometry as G
    import relations as R
    import thresholds as T
    dataset = NativeVectorDataset()
    record = image_record_for_reasoning(dataset, "1008", "test", split_view="scene_disjoint_v1")
    geometry = G.image_geometry_from_record(record)
    config = T.load_config(str(REPO / "configs/spatial_relations_v1.yaml"))
    labels = dataset.label_map("1008")
    result = {}
    for component in geometry.components:
        relation = R.evaluate_direction("above", geometry, component, geometry.get(4), config)
        result[component.component_id] = {"relation_validity": vars(relation),
                                         "canonical_boundary_distance_px": G.boundary_distance(labels, component.component_id, 4),
                                         "native_geometry_centroid_xy": list(component.centroid_px)}
    return result, {"relation_function": "spatial_reasoning/relations.py:evaluate_direction",
                    "geometry_bridge": "buildreasonseg_mvp/native_vector_adapter.py:image_record_for_reasoning",
                    "distance_function": "spatial_reasoning/geometry.py:boundary_distance",
                    "config": "configs/spatial_relations_v1.yaml", "active": config.direction.active_candidate,
                    "alpha": config.direction.alpha, "tau": config.direction.tau,
                    "distance_convention": "canonical minimum boundary-pixel Euclidean gap with existing -1 convention"}


def source_trace(path: str, function: str) -> dict:
    text = (REPO / path).read_text(encoding="utf-8")
    nodes = ast.parse(text).body
    for name in function.split("."):
        node = next(n for n in nodes if getattr(n, "name", None) == name)
        nodes = getattr(node, "body", [])
    return {"path": path, "function": function, "line": node.lineno, "end_line": node.end_lineno,
            "source": "\n".join(text.splitlines()[node.lineno-1:node.end_lineno])}


def audit_sources(payload: dict, result: dict) -> None:
    specs = [("core.py", "Db1Runtime.load"), ("core.py", "Db1Runtime.forward"), ("core.py", "run_core_chain"),
             ("context.py", "context_to_global"), ("context.py", "guard_directional_candidates"),
             ("context.py", "directional_candidates"), ("context.py", "direction_satisfied"),
             ("pipeline.py", "predict_one"), ("pipeline.py", "_non_padding_mask"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "field_competition"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "_group_norm"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "VisualProjection.__init__"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "VisualProjection.forward"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "DirectionEmbedding.__init__"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "MaskDecoderTrunk.__init__"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "MaskDecoderTrunk.forward"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "target_prototype"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "prototype_similarity"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "GlobalCompetitionDecoder.competition"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "GlobalCompetitionDecoder.forward"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "GlobalCompetitionDecoder.upsampled"),
             ("_frozen/mvp/task7d_global_competition_decoder.py", "GlobalCompetitionDecoder.input_spec"),
             ("_frozen/mvp/task6z_field_composition.py", "record_fields"),
             ("_frozen/mvp/geometric_relation_field_v02.py", "geometric_relation_field_v02"),
             ("_frozen/mvp/nearest_boundary_field.py", "nearest_boundary_field_512")]
    payload["source_function_trace"] = [source_trace(RUNTIME_SOURCE+p, f) for p, f in specs]
    payload["source_function_trace"] += [source_trace(p, f) for p, f in
        [("spatial_reasoning/relations.py", "evaluate_direction"),
         ("spatial_reasoning/geometry.py", "boundary_distance"),
         ("buildreasonseg_mvp/native_vector_adapter.py", "image_record_for_reasoning")]]
    payload["db1_mechanism_audit"] = {
        "W": "clamp(P_dir_64 * P_near_64, 0, 1)", "A": "deterministic W/(spatial sum W+EPS); not learned; no score head for D-B1",
        "q": "sum_i A_i F_i using projected SAM2 features F (256->128 Conv1x1/GroupNorm/GELU)",
        "C": "plain normalized feature/prototype cosine with EPS; no learned scale, sigmoid or threshold",
        "decoder_inputs": "148 channels: projected F 128, P_dir 1, P_near 1, direction embedding 16, A_vis=A*4096 1, C 1",
        "decoder": "dense convolution trunk 148->128->64->1, bilinear 64->512 align_corners=False",
        "output_threshold": "upsampled_logits > 0.0, sigmoid probability recorded separately",
        "target_proposal_selection": False, "graph_construction": False, "dense_mask_emitted": True,
        "target_identity_or_nearest_identity_gate": False,
        "saved_C_and_logits_paths": "Saved C is from the separately computed projected-feature competition state; logits use a second model forward under CUDA bfloat16 autocast when device is not cpu. Equality of those internal states is not established by saved maps.",
        "scope": "D-B1 selected in core.Db1Runtime.load; other variants in source are not the active runtime."}
    payload["runtime_guard_audit"] = {
        "frozen_directional_guard": result["directional_guard"],
        "direction_availability": "Pre-inference proposals in requested centroid half-plane; fails only if candidates exist and none has centroid inside context. Zero candidates alone does not fail this guard.",
        "final_direction_check": "above: target_centroid_row < selected_reference_centroid_row",
        "observed_target_centroid_yx": result["target_centroid"],
        "observed_reference_centroid_yx": result["reasoning_context"]["reference_centroid_global"],
        "observed_final_direction_pass": result["target_centroid"][0] < result["reasoning_context"]["reference_centroid_global"][0],
        "nonempty_mask": result["mask_area"] > 0,
        "padding_check": "context_to_global already crops padding; _non_padding_mask returns all ones. Not an independent semantic gate.",
        "nearest_target_identity_checked": False, "GT_identity_checked": False,
        "runtime_success_scope": "RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED",
        "why_wrong_target_can_succeed": "After other pipeline stages complete, any nonempty mapped mask with centroid above the selected reference can pass; neither nearest building nor native GT identity is verified."}
    traces = {r["function"]: {k: r[k] for k in ("path", "function", "line", "end_line")}
              for r in payload["source_function_trace"]}
    mechanism_functions = {
        "W": ["field_competition"], "A": ["field_competition", "GlobalCompetitionDecoder.input_spec"],
        "q": ["target_prototype", "VisualProjection.__init__", "VisualProjection.forward", "_group_norm", "GlobalCompetitionDecoder.input_spec"],
        "C": ["prototype_similarity", "GlobalCompetitionDecoder.competition"],
        "decoder_inputs": ["GlobalCompetitionDecoder.forward", "DirectionEmbedding.__init__", "GlobalCompetitionDecoder.input_spec"],
        "decoder": ["MaskDecoderTrunk.__init__", "MaskDecoderTrunk.forward", "GlobalCompetitionDecoder.upsampled"],
        "output_threshold": ["Db1Runtime.forward"],
        "target_proposal_selection": ["run_core_chain", "Db1Runtime.forward", "GlobalCompetitionDecoder.forward"],
        "graph_construction": ["run_core_chain", "GlobalCompetitionDecoder.competition", "GlobalCompetitionDecoder.forward"],
        "dense_mask_emitted": ["Db1Runtime.forward", "GlobalCompetitionDecoder.forward"],
        "target_identity_or_nearest_identity_gate": ["predict_one", "direction_satisfied"],
        "saved_C_and_logits_paths": ["Db1Runtime.forward"], "scope": ["Db1Runtime.load"]}
    payload["db1_mechanism_source_citations"] = {k: [traces[f] for f in functions]
                                                for k, functions in mechanism_functions.items()}
    payload["runtime_guard_source_citations"] = {
        k: [traces[f] for f in functions] for k, functions in {
            "direction_availability": ["directional_candidates", "guard_directional_candidates"],
            "final_direction_check": ["direction_satisfied", "predict_one"],
            "padding_check": ["context_to_global", "_non_padding_mask", "predict_one"],
            "nearest_target_identity_checked": ["predict_one", "direction_satisfied"],
            "GT_identity_checked": ["predict_one"],
            "runtime_success_scope": ["predict_one", "direction_satisfied"],
            "why_wrong_target_can_succeed": ["predict_one", "direction_satisfied"]}.items()}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NOT_ESTABLISHED"
    return value


def table(headers, rows) -> str:
    fmt = lambda v: "N/A" if v is None else f"{v:.9g}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


def diagnosis_conclusions(d: dict) -> list[str]:
    first, ranks, cf = d["first_observed_target_divergence_stage"], d["rank_trajectory"], d["canonical_reference_field_counterfactual"]
    pair = {r["stage"]: r for r in d["target6_vs_wrong7"]}
    rows = {r["instance_id"]: r for r in d["all_gt_instance_stage_table"]}
    guard = d["runtime_guard_audit"]
    return [
        f"FIRST_OBSERVED_DIVERGENCE: {first['stage']} / {first.get('statistic')}. Actual top instance {first.get('actual_top_instance_id')}; target6 rank {ranks[first['stage']]['target6_rank']}, wrong7 rank {ranks[first['stage']]['wrong7_rank']}. Delta6-minus7={first.get('delta_6_minus_7')}. This is the earliest fixed-order observable preference, not a proven root cause.",
        f"PRIMARY_EVIDENCE_SUPPORTS: P_dir mean favours{pair['P_dir']['which_instance_is_favoured']}, P_near mean favours{pair['P_near']['which_instance_is_favoured']}. Frozen W mass favours{pair['W']['which_instance_is_favoured']} and A mass favours{pair['A']['which_instance_is_favoured']} over the other member of the 6-vs7 pair; W/A top instance is {ranks['W']['top_instance_id']}/{ranks['A']['top_instance_id']}. Thus in this locked case a wrong7 preference is not present in the combined W/A comparison before C, but no discrete correct target selection is claimed.",
        f"PRIMARY_EVIDENCE_SUPPORTS: C is where the observed 6-vs7 preference reverses. Logit mean ({rows[6]['logit_mean']} vs {rows[7]['logit_mean']}), probability mean ({rows[6]['probability_mean']} vs {rows[7]['probability_mean']}), positive fraction and final IoU continue to favour7. This is an observed downstream preference; saved C and decoder-forward states are computed separately. It does not prove C caused the decoder mask or isolate F, embedding, attention, C or learned-trunk contributions.",
        f"{cf['interpretation']}: GT-reference deterministic fields still favour6, as automatic-reference W/A already did. W and A deltas increase numerically, but target6/wrong7 ranks remain2/3; there is no preference restoration. Reference-shape residual does not account for an observed W/A reversal in this comparison; its possible effect on q/C/decoder remains untested, not ruled out.",
        f"Canonical geometry: both6 and7 satisfy the frozen above predicate. Existing canonical boundary gap to GT ref4 is {rows[6]['canonical_boundary_distance_px']}px for6 versus {rows[7]['canonical_boundary_distance_px']}px for7. The frozen nearest/proximity field gives greater mean to6; the eventual dense mask overlaps7 more.",
        f"Runtime structural SUCCESS permits the semantic error: saved guard reports {guard['frozen_directional_guard']['candidate_count']} directional proposals and {guard['frozen_directional_guard']['inside_context']} inside context, final nonempty mask centroid row {guard['observed_target_centroid_yx'][0]} < selected-reference row {guard['observed_reference_centroid_yx'][0]}. Runtime checks this direction and mapped nonemptiness; it does not verify nearest building or native GT identity."]


def cited_audit(audit: dict, citations: dict) -> str:
    chunks = []
    for k, v in audit.items():
        refs = "; ".join(f"`{r['path']}:{r['line']}` ({r['function']}, end line {r['end_line']})"
                         for r in citations.get(k, []))
        chunks.append(f"**{k}**: {v}" + ("\n\nSource: " + refs + "." if refs else ""))
    return "\n\n".join(chunks)


def render_report(d: dict) -> str:
    pair, rank, cf = d["target6_vs_wrong7"], d["rank_trajectory"], d["canonical_reference_field_counterfactual"]
    first = d["first_observed_target_divergence_stage"]
    p = ["# Task 8D ABOVE Target Chain Diagnosis V1", "Status: " + d["status"] + ". Diagnosis only; no repair. Next gate: CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT."]
    def section(n, title, body):
        p.extend([f"## {n}. {title}", body])
    section(1, "Question being diagnosed", "At what earliest observable stage does native GT target 6 lose numerically to wrong instance 7? Fixed order: W -> A -> C -> logits -> probability -> final mask. No pass threshold or root-cause claim is introduced.")
    section(2, "Why ABOVE is the clean case", "Sample " + SAMPLE + "; program " + PROGRAM + ". Reference best-overlap identity is GT 4, IoU 0.9097432024169184. Frozen target best overlap is GT 7, IoU 0.48158096699923253; canonical target 6 IoU 0.07762201453790239 / Dice 0.14406167188629246. This reduces the reference-identity confounder; reference shape remains imperfect.")
    closure = d.get("closure_summary", "Final dedicated tests and integrity gate are pending.")
    section(3, "Frozen evidence / no-rerun guarantee", "Task8C accepted HEAD: " + ACCEPTED_HEAD + ". Frozen runtime SHA256: " + RUNTIME_SHA + ". All ABOVE 15 artifacts/transcript and Task8C 10 repository evidence/harness files are SHA-locked. Saved maps are read-only primary evidence and none of the seven automatic maps is recomputed. Model/detector/predict/checkpoint calls and external writes are zero. Original Task8C formal runner remains one invocation / four attempts / zero retries. Pre/post identities and read-only audit are in JSON.\n\n" + closure + "\n\n" + table(["Map", "Shape", "Finite", "Min", "Max", "Sum"], [(k,v["shape"],v["finite"],v["min"],v["max"],v["sum"]) for k,v in d["frozen_map_shapes"].items()]))
    mechanism = cited_audit(d["db1_mechanism_audit"], d.get("db1_mechanism_source_citations", {}))
    citations = table(["Exact source path", "Function", "Line / end line"], [(r["path"],r["function"],str(r["line"])+" / "+str(r["end_line"])) for r in d["source_function_trace"]])
    section(4, "Actual target mechanism in RC1", mechanism + "\n\nSource traces (full exact snippets and SHA identities also in JSON):\n\n" + citations + "\n\nTarget-stage candidate rankings below are post-hoc GT aggregations of dense maps; they are not a target-proposal selector. No graph is constructed. A_fixed is deterministic, while projected visual features and the dense decoder are learned components. Exact normalization retains source EPS=1e-6; W/sum(W) is shorthand, not a changed rule.")
    geo = table(["GT id", "Full GT area", "Context area", "Coverage", "Canonical above valid", "Reason", "Boundary gap to GT ref4 (px)"], [(r["instance_id"],r["GT_area"],r["context_GT_area"],r["context_coverage_fraction"],r["relation_validity"]["valid"],r["relation_validity"]["reason"],r["canonical_boundary_distance_px"]) for r in d["all_gt_instance_stage_table"]])
    section(5, "Reference geometry audit", "Frozen origin [left,top]=[-197,-41], size 512. Context GT is exact integer crop/translation with zero in reflected-image padding; no GT reflection or resizing at 512. Canonical ref4, target6 and wrong7 are nonempty and their exact represented areas are shown below. Every GT instance in tile1008 is reported; rankings include every nonempty context intersection, including ref4 and any canonically invalid direction, so no competitor is hidden. Geometry uses native_vector_adapter.image_record_for_reasoning -> geometry.image_geometry_from_record -> relations.evaluate_direction with unchanged config, and geometry.boundary_distance (existing gap convention). Runtime half-plane checks are audited separately.\n\n" + geo)
    rows = d["all_gt_instance_stage_table"]
    section(6, "P_dir / P_near analysis", "64x64 aggregation uses cv2.INTER_AREA fractional occupancy of exact context masks, never a threshold. mass=sum(X*M), mean=mass/sum(M); target6/wrong7 comparison uses their means. Saved arrays are not altered for quantitative analysis.\n\n" + table(["ID", "P_dir mass", "P_dir mean", "P_near mass", "P_near mean"], [(r["instance_id"],r["P_dir_mass"],r["P_dir_mean"],r["P_near_mass"],r["P_near_mean"]) for r in rows]))
    section(7, "W / A analysis", "Candidate W mass=sum(W*M); attention mass=sum(A*M). All candidates use the same global A normalization. Mass measures total support and can reflect candidate size; means are separately reported. There is no discrete target-instance decision at either stage.\n\n" + table(["ID", "W mass", "W mean", "A attention mass"],[(r["instance_id"],r["W_mass"],r["W_mean"],r["A_attention_mass"]) for r in rows]))
    section(8, "C prototype-similarity analysis", "C is the saved cosine field from the A-weighted projected visual prototype. The table reports occupancy-weighted C mean, not an instance-selected prototype. F and q are not saved in this bundle, so contributions to q cannot be disentangled without forbidden model execution.\n\n" + table(["ID", "C mean"],[(r["instance_id"],r["C_mean"]) for r in rows]))
    section(9, "Decoder logits / probability analysis", "Exact context GT pixels aggregate frozen 512x512 upsampled logits and saved sigmoid probabilities. Positive-logit fraction uses the existing runtime logits>0 rule only. Different means can rank differently after nonlinear sigmoid; this does not isolate causal decoder inputs.\n\n" + table(["ID", "Logit mean", "Probability mean", "Positive-logit fraction"],[(r["instance_id"],r["logit_mean"],r["probability_mean"],r["positive_logit_fraction"]) for r in rows]))
    section(10, "Final mask analysis", "All IoUs use the untouched frozen 1008_mask.png and exact full native tile masks. The binary context logits>0 mask maps back to exactly the frozen output. No native GT occupancy rounding, morphology, component picking or repainting is applied.\n\n" + table(["ID", "Final predicted-mask IoU"],[(r["instance_id"],r["final_mask_iou"]) for r in rows]))
    trajectory = table(["Stage / statistic", "Target6", "Wrong7", "Delta 6-7", "Favoured"],[(r["stage"]+" / "+r["statistic"],r["score_target6"],r["score_wrong7"],r["delta_6_minus_7"],r["which_instance_is_favoured"]) for r in pair])
    ranks = table(["Stage", "6 rank", "7 rank", "Top5 IDs", "All ordered IDs"],[(k,v["target6_rank"],v["wrong7_rank"],v["top5_instance_ids"],v["ordered_instance_ids"]) for k,v in rank.items()])
    section(11, "Target-6 vs wrong-7 trajectory", trajectory + "\n\n" + ranks + "\n\nDescending scores; exact ties break by ascending instance ID. Undefined means are excluded and no numeric pass threshold is used. Predeclared statistics were pushed in the evidence-lock checkpoint before aggregation.")
    cf_rows = cf["target6_vs_wrong7"]
    section(12, "Canonical-reference deterministic counterfactual", "Only P_dir, P_near, W and A are recomputed from exact GT ref4 in the same frozen context using the frozen field code. No SAM2/F/q/C/decoder/checkpoint is evaluated. Normalization calls the frozen pure field_competition function and retains EPS; no neural module is constructed. Interpretation: **" + cf["interpretation"] + "**; not causal proof. " + cf["interpretation_detail"] + "\n\n" + table(["GT-ref stage", "6 mass", "7 mass", "Delta 6-7", "Favoured"],[(r["stage"],r["score_target6"],r["score_wrong7"],r["delta_6_minus_7"],r["which_instance_is_favoured"]) for r in cf_rows]) + "\n\n" + table(["Stage", "Auto 6/7 ranks", "GT-ref 6/7 ranks", "GT-ref top5"],[(k,str(rank[k]["target6_rank"])+"/"+str(rank[k]["wrong7_rank"]),str(cf["rank_trajectory"][k]["target6_rank"])+"/"+str(cf["rank_trajectory"][k]["wrong7_rank"]),cf["rank_trajectory"][k]["top5_instance_ids"]) for k in ("W","A")]) + "\n\n" + table(["Field", "Mean signed diff", "Mean abs diff", "Max abs diff", "RMS diff"],[(k,v["mean_signed_difference"],v["mean_absolute_difference"],v["max_absolute_difference"],v["rms_difference"]) for k,v in cf["field_difference_statistics"].items()]))
    section(13, "First observed divergence", json.dumps(first,ensure_ascii=False,indent=2) + "\n\nThis is the earliest observed preference under the fixed aggregation rule. It is not a proven root cause, and the actual top competitor is reported even when neither instance6 nor instance7 leads.")
    section(14, "Runtime SUCCESS guard explanation", cited_audit(d["runtime_guard_audit"], d.get("runtime_guard_source_citations", {})) + "\n\nProposal availability and final mask centroid direction do not check nearest identity or GT identity.")
    section(15, "Evidence-supported diagnosis", "\n\n".join(d["conclusions"]))
    section(16, "What remains unresolved", "\n\n".join(d["limitations"]))
    section(17, "Possible future research directions - hypotheses only, NO IMPLEMENTATION", "Questions for future Supervisor-authorized research: how dense soft direction/proximity priors relate to strict canonical direction filtering and minimum boundary distance; whether global/background token contributions shape the prototype; how visual feature similarity and dense decoder inputs interact. These are hypotheses, not proposed repairs or an architecture prescription. No parameter/threshold/ranking change, ablation, training or inference is performed here.")
    section(18, "Questions worth discussing with the advisor", "How should strict L3 relation/nearest semantics be represented in a dense attention distribution? Which observed stage statistic best reflects the intended identity semantics, and how should area dependence be interpreted? What evidence would distinguish prototype-feature effects from decoder behavior without conflating them? What independent population would test such hypotheses without reusing these frozen cases? Final scientific interpretation remains with ChatGPT Supervisor.\n\nDiagnostic figure paths and SHA256:\n\n" + table(["Figure", "Path", "SHA256"], [(Path(r["path"]).name,r["path"],r["sha256"]) for r in d["diagnostic_figures"]]))
    return "\n\n".join(p)+"\n"


def figures(d: dict, maps: dict, masks: dict, contexts: dict, row: dict, auto_ref: np.ndarray) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {4: "#58e36c", 6: "#00d9ff", 7: "#ff9b37"}
    labels = {4: "GT reference4", 6: "GT target6", 7: "Wrong best-overlap7"}
    handles = [Line2D([0],[0],color=colors[i],lw=2,label=labels[i]) for i in (4,6,7)]
    def outlines(ax, pool):
        for i in (4,6,7):
            ax.contour(pool[i], levels=[0.5], colors=[colors[i]], linewidths=1.2)
        ax.set_xlim(-0.5,511.5);ax.set_ylim(511.5,-0.5)
        ax.set_xlabel("column / x");ax.set_ylabel("row / y")
    def save(fig, name):
        fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=11)
        fig.subplots_adjust(bottom=0.16,wspace=0.24,top=0.87)
        path=require_output(FIGURE_ROOT/name)
        if path.exists():
            raise ValueError("preserve existing diagnostic figure: "+str(path))
        fig.savefig(path,dpi=130);plt.close(fig)
    rgb=np.array(Image.open(row["image"]).convert("RGB"))
    context_rgb=np.array(Image.open(Path(row["diagnostics_path"])/"reasoning_context.png").convert("RGB"))
    from scripts.task8c_final_demo_evaluate import context_to_global
    auto_global=context_to_global(auto_ref,row["reasoning_context"],(512,512))
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.suptitle("01 | Frozen reference geometry: tile and exact reasoning context")
    for ax,image,pool,reference,title in zip(axes,[rgb,context_rgb],[masks,contexts],[auto_global,auto_ref],["Original tile / selected-ref IoU=0.909743","Saved context / origin=(-197,-41), size=512"]):
        ax.imshow(image);outlines(ax,pool);ax.contour(reference,levels=[0.5],colors=["#f05bdf"],linewidths=0.9);ax.set_title(title)
    handles.append(Line2D([0],[0],color="#f05bdf",lw=2,label="Saved automatic reference"))
    save(fig,FIGURES[0]);handles.pop()
    for name,keys in [(FIGURES[1],["P_dir"]),(FIGURES[2],["P_near"]),(FIGURES[3],["W","A"]),(FIGURES[4],["C"]),(FIGURES[5],["logits","probability"])]:
        fig,axes=plt.subplots(1,len(keys),figsize=(14 if len(keys)>1 else 8,6),squeeze=False)
        fig.suptitle(name[:-4]+" | Frozen values, native GT boundaries in context coordinates")
        for ax,k in zip(axes[0],keys):
            v=maps[k]
            im=ax.imshow(v,extent=(-0.5,511.5,511.5,-0.5),interpolation="nearest",cmap="coolwarm" if k=="logits" else "viridis",vmin=float(v.min()),vmax=float(v.max()))
            outlines(ax,contexts);fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04);ax.set_title(f"{k} {v.shape} / display-only color scaling")
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.suptitle("07 | Frozen output mask; no new mask or overlay is written")
    axes[0].imshow(np.array(Image.open(row["overlay_path"]).convert("RGB")));outlines(axes[0],masks);axes[0].set_title("Saved overlay / target6 IoU=0.077622 / wrong7 IoU=0.481581")
    axes[1].imshow(maps["logits"]>0,cmap="gray",vmin=0,vmax=1);outlines(axes[1],contexts);axes[1].set_title("Frozen context logits>0 (existing runtime rule)")
    save(fig,FIGURES[6])
    fig,axes=plt.subplots(1,2,figsize=(15,6));fig.suptitle("08 | Fixed-statistic rank trajectory (lower rank is greater score)")
    ranks=d["rank_trajectory"]
    relevant=ranks["W"]["ordered_instance_ids"]
    for i in sorted(relevant):
        ys=[next(x["rank"] for x in ranks[k]["scores"] if x["instance_id"]==i) for k in STAGES]
        axes[0].plot(range(6),ys,marker="o",lw=3 if i in (6,7) else 1,alpha=1 if i in (6,7) else 0.45,color=colors.get(i),label=f"GT {i}")
    axes[0].set_xticks(range(6),STAGES);axes[0].set_yticks(range(1,len(relevant)+1));axes[0].invert_yaxis();axes[0].set_ylabel("Descending score rank, exact ties -> ascending ID");axes[0].legend(ncol=3);axes[0].grid(alpha=0.2)
    axes[1].imshow(context_rgb);outlines(axes[1],contexts);axes[1].set_title("Context reference4 / target6 / wrong7 boundaries")
    save(fig,FIGURES[7])
    panels=[]
    for name in FIGURES[:8]:
        with Image.open(FIGURE_ROOT/name) as im:
            panel=Image.new("RGB",(1500,850),"white");small=im.convert("RGB");small.thumbnail((1500,850),Image.Resampling.LANCZOS);panel.paste(small,((1500-small.width)//2,(850-small.height)//2));panels.append(panel)
    sheet=Image.new("RGB",(4500,2550),"white")
    for i,panel in enumerate(panels):sheet.paste(panel,((i%3)*1500,(i//3)*850))
    from PIL import ImageDraw
    ImageDraw.Draw(sheet).text((3050,1820),"Task8D | DIAGNOSIS ONLY\nGT ref4: green | target6: cyan | wrong7: orange\nFrozen maps; display scaling only\nModel/detector/predict/checkpoint calls: 0\nExternal writes: 0\nNo repair implemented",fill="black")
    sheet.save(require_output(FIGURE_ROOT/FIGURES[8]))
    return [identity(FIGURE_ROOT/name) for name in FIGURES]


def verify_baseline(d: dict, *, external_inventory: bool = True) -> None:
    verify_identities(d["preflight"]["task8c_evidence_identities"] + d["preflight"]["protected_canonical_and_governance_identities"] + d["source_identities"])
    if external_inventory:
        records=d["preflight"]["external_before"]
        current={p.relative_to(EXTERNAL).as_posix():p for p in EXTERNAL.rglob("*") if p.is_file()}
        if set(current)!={r["path"] for r in records}:
            raise ValueError("external file set changed")
        for row in records:
            p=current[row["path"]];ident=identity(p)
            if (ident["bytes"],ident["sha256"],p.stat().st_mtime_ns)!=(row["bytes"],row["sha256"],row["mtime_ns"]):
                raise ValueError("external bytes/hash/mtime changed: "+row["path"])


def diagnose(d: dict) -> tuple[dict, dict]:
    sys.path.insert(0,str(REPO))
    from scripts import task8c_final_demo_evaluate as truth_reader
    task8c=json.loads((REPO/"evaluation/task8c_final_demo_v1.json").read_text(encoding="utf-8"))
    row=lock_above(task8c)
    result=json.loads((Path(row["diagnostics_path"])/"result.json").read_text(encoding="utf-8"))
    if result["reasoning_context"]!=row["reasoning_context"]:
        raise ValueError("frozen result/context differs")
    maps=load_maps(Path(d["frozen_maps_identity"]["path"]))
    masks,provenance=truth_reader.load_truth(row)
    final=truth_reader.read_binary(Path(row["mask_path"]),(512,512))
    geometry, geometry_provenance=canonical_geometry()
    rows,contexts=stage_rows(maps,masks,row["reasoning_context"],final,geometry)
    for i in (4,6,7):
        if not contexts[i].any():raise ValueError("required GT instance absent from frozen context")
        mapped=truth_reader.context_to_global(contexts[i],row["reasoning_context"],(512,512))
        if not np.array_equal(mapped,masks[i]):raise ValueError("required GT truncated/misaligned")
    auto=truth_reader.read_binary(Path(row["diagnostics_path"])/"reference_context_mask.png",(512,512))
    if iou(truth_reader.context_to_global(auto,row["reasoning_context"],(512,512)),masks[4])!=row["gt_audit"]["selected_reference_iou_with_canonical_gt_reference"]:
        raise ValueError("reference identity/alignment differs")
    if not np.array_equal(truth_reader.context_to_global(maps["logits"]>0,row["reasoning_context"],(512,512)),final):
        raise ValueError("frozen positive logits/global final mask differs")
    by_id={r["instance_id"]:r for r in rows}
    if by_id[6]["final_mask_iou"]!=row["gt_audit"]["target_iou"] or by_id[7]["final_mask_iou"]!=row["gt_audit"]["predicted_target_best_gt_iou"]:
        raise ValueError("native truth/frozen target metrics differ")
    pair=pair_comparison(rows);rank=rank_trajectory(rows)
    d.update({"frozen_map_shapes":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"finite":bool(np.isfinite(v).all()),"min":float(v.min()),"max":float(v.max()),"sum":float(v.astype(np.float64).sum())} for k,v in maps.items()},
              "all_gt_instance_stage_table":rows,"target6_vs_wrong7":pair,"rank_trajectory":rank,
              "first_observed_target_divergence_stage":first_divergence(pair,rank),
              "canonical_reference_field_counterfactual":counterfactual(maps,contexts),
              "native_truth_provenance":provenance,"canonical_geometry_provenance":geometry_provenance,
              "mapping_verification":{"reference4_target6_wrong7_fully_represented":True,"native_exact_integer_mapping":True,"fractional_no_threshold":True,"saved_positive_logits_map_to_exact_final_mask":True},
              "model_calls":0,"detector_calls":0,"predict_calls":0,"checkpoint_loads":0,"external_writes":0})
    audit_sources(d,result)
    d["conclusions"]=diagnosis_conclusions(d)
    d["limitations"]=["One locked qualitative test example; not a new population metric or generalization result. Task7J metrics, model, seed, architecture and thresholds remain unchanged.",
                       "Instance mass depends on covered area; C/logit/probability means measure different quantities. Changes in their numerical preferences do not isolate causal contributions.",
                       "F and q and raw 64x64 decoder logits are not saved. C uses the saved competition state while logits follow the runtime forward path; no equality of mixed-precision internal feature states can be independently established from these artifacts.",
                       "Reflected context imagery has padding, but GT is defined only on the original tile. Partially represented instances are disclosed; all nonempty GT context intersections are ranked, with reference included.",
                       "The deterministic GT-reference counterfactual changes fields only, cannot establish what a model would output, and is not causal proof. No repair or new acceptance threshold is authorized."]
    return d,{"maps":maps,"masks":masks,"contexts":contexts,"row":row,"auto_ref":auto}


def main() -> int:
    if len(sys.argv)!=1:raise ValueError("no diagnostic overrides accepted")
    d=json.loads(JSON_PATH.read_text(encoding="utf-8"))
    verify_baseline(d)
    # Initialize plotting/font cache in the already-existing outside-repo MPLCONFIGDIR.
    # No RC1 path is used; all diagnosis work below is guarded.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    with OfflineGuard() as guard:
        d,visual=diagnose(d)
        d["diagnostic_figures"]=figures(d,**visual)
        verify_baseline(d)
        lock_above(json.loads((REPO/"evaluation/task8c_final_demo_v1.json").read_text(encoding="utf-8")))
        d["offline_read_audit"]=guard.read_modes
        d["task8c_artifacts_unchanged"]=True
        d["integrity_postcheck"]={"pass":True,"external_file_count":len(d["preflight"]["external_before"]),"external_bytes_sha_mtime_and_file_set_unchanged":True,"Task8C_repo_artifacts_unchanged":True,"canonical_source_governance_unchanged":True,"all_source_identities_unchanged":True}
        d["status"]="DIAGNOSIS_COMPLETE_PENDING_TESTS_AND_VISUAL_REVIEW"
        d=clean(d)
        REPORT_PATH.write_text(render_report(d),encoding="utf-8",newline="\n")
        JSON_PATH.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"first_divergence":d["first_observed_target_divergence_stage"],"target6_vs_wrong7":d["target6_vs_wrong7"],"counterfactual":d["canonical_reference_field_counterfactual"]["target6_vs_wrong7"],"model_calls":0,"external_writes":0},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
