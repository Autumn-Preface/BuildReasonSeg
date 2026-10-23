"""Exact anchored V2-only observation edits, plus scientific AST equivalence audit.

Input is always accepted Git blob bytes. This never edits a canonical or existing delivery file.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
from pathlib import Path

PREFIX = "buildreasonseg/runtime/"
FILES = (PREFIX + "pipeline.py", PREFIX + "core.py", PREFIX + "_frozen/mvp/task7d_global_competition_decoder.py")
HELPER = Path(__file__).resolve().parent / "engine_observation/observation.py"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Accepted source anchor conflict; STOP: " + old[:100])
    return source.replace(old, new)


def pipeline_patch(source):
    source = replace_once(source, "from buildreasonseg import paths\n", "from buildreasonseg import paths\nfrom buildreasonseg.runtime.observation import emit as _trace_emit\n")
    source = replace_once(source, "def predict_one(runtime: PredictRuntime, request: PipelineRequest) -> PipelineResult:",
                          "def predict_one(runtime: PredictRuntime, request: PipelineRequest, *, observer=None) -> PipelineResult:")
    source = replace_once(source, "    timings = StageTimings()\n", "    timings = StageTimings()\n    _trace_stage = 1\n")
    source = replace_once(source, "        if parsed is None:\n", """        if parsed is None:
            if observer is not None:
                _trace_emit(observer, 1, "RUNNING", {"original_prompt": request.prompt})
""")
    source = replace_once(source, "        return PipelineResult(status=\"FAILED\", image=request.image, result_payload=payload,",
        """        if observer is not None:
            _trace_emit(observer, 9, "FAILED", {"runtime_status": "FAILED", "error_code": error.code,
                        "reason": payload.get("reason"), "context": error.context,
                        "guard_executed": _trace_stage == 9, "final_output_valid": False,
                        "elapsed_seconds": time.time() - total_started,
                        "validity_scope": "RUNTIME_STRUCTURAL_ONLY", "semantic_status": "NOT_EVALUATED"})
        return PipelineResult(status="FAILED", image=request.image, result_payload=payload,""")
    source = replace_once(source, "        # ---- global detection + merge\n", """        if observer is not None:
            if request.parsed is None:
                _trace_emit(observer, 1, "COMPLETED", {"original_prompt": request.prompt,
                            "initial_program": parsed.program, "initial_supported": validation.supported,
                            "final_program": parsed.program, "language_mode": payload.get("language_mode", "qwen")})
        # ---- global detection + merge
        _trace_stage = 2
        if observer is not None:
            _trace_emit(observer, 2, "RUNNING")
""")
    source = replace_once(source, "        if not proposals:\n", """        if observer is not None:
            _trace_emit(observer, 2, "COMPLETED", {"raw_count": detection["raw_count"],
                        "merged_count": len(proposals), "detector_seconds": detection["detector_seconds"],
                        "proposal_ids": [p.proposal_id for p in proposals]},
                        {"rgb": loaded.rgb, "proposals": [{"proposal_id": p.proposal_id,
                         "bbox": p.global_bbox, "mask_crop": p.mask_crop} for p in proposals]})
        if not proposals:
""")
    source = replace_once(source, "        # ---- global reference\n", """        # ---- global reference
        _trace_stage = 3
        if observer is not None:
            _trace_emit(observer, 3, "RUNNING")
""")
    source = replace_once(source, "        # ---- deterministic 512 reasoning context\n", """        if observer is not None:
            _trace_emit(observer, 3, "COMPLETED", {"selected_id": reference.proposal_id,
                        "mask_area": int(reference.mask_area), "confidence": float(reference.confidence),
                        "bbox": list(reference.global_bbox), "mode": reference_mode},
                        {"reference_mask_crop": reference.mask_crop})
        # ---- deterministic 512 reasoning context
        _trace_stage = 4
        if observer is not None:
            _trace_emit(observer, 4, "RUNNING")
""")
    source = replace_once(source, "        guard = guard_directional_candidates(reference, proposals, direction, context)",
        """        if observer is not None:
            _trace_emit(observer, 4, "RUNNING", {"direction": direction, "context": context.to_dict(),
                        "context_rgb_available": False})
        guard = guard_directional_candidates(reference, proposals, direction, context)""")
    source = replace_once(source, "        chain = run_core_chain(rgb_context, reference_mask, parsed.program, sam2=runtime.sam2,\n                               db1=runtime.db1)",
        """        if observer is not None:
            _trace_emit(observer, 4, "COMPLETED", {"direction": direction, "context": context.to_dict(),
                        "reference_pixels": int(reference_mask.sum())},
                        {"context_rgb": rgb_context, "reference_mask": reference_mask})
        _trace_stage = 5
        if observer is None:
            chain = run_core_chain(rgb_context, reference_mask, parsed.program, sam2=runtime.sam2,
                                   db1=runtime.db1)
        else:
            chain = run_core_chain(rgb_context, reference_mask, parsed.program, sam2=runtime.sam2,
                                   db1=runtime.db1, observer=observer)""")
    source = replace_once(source, "        # ---- hard post-inference validity (no \"pick the best\")\n", """        # ---- hard post-inference validity (no "pick the best")
        _trace_stage = 9
        if observer is not None:
            _trace_emit(observer, 9, "RUNNING", {"guard_executed": True})
""")
    source = replace_once(source, "        return PipelineResult(status=\"SUCCESS\", image=request.image, result_payload=payload,\n                              mask=mask_full, outputs=outputs)",
        """        if observer is not None:
            _trace_emit(observer, 9, "COMPLETED", {"runtime_status": "SUCCESS", "guard_executed": True,
                        "final_output_valid": True, "validity_scope": payload["validity_scope"],
                        "semantic_status": payload["semantic_status"], "elapsed_seconds": timings.total,
                        "mask_path": info["mask"], "overlay_path": info["overlay"]})
        return PipelineResult(status="SUCCESS", image=request.image, result_payload=payload,
                              mask=mask_full, outputs=outputs)""")
    return source


def core_patch(source):
    source = replace_once(source, "from buildreasonseg import paths\n", "from buildreasonseg import paths\nfrom buildreasonseg.runtime.observation import emit as _trace_emit, observation_scope as _trace_scope\n")
    source = replace_once(source, "    def forward(self, visual, direction: str, directional, nearest) -> dict:",
                          "    def forward(self, visual, direction: str, directional, nearest, *, observer=None) -> dict:")
    source = replace_once(source, "                logits = model(visual_batch, [relation], directional_batch, nearest_batch)",
        """                if observer is None:
                    logits = model(visual_batch, [relation], directional_batch, nearest_batch)
                else:
                    with _trace_scope(observer):
                        logits = model(visual_batch, [relation], directional_batch, nearest_batch)""")
    source = replace_once(source, "                   sam2: Sam2Runtime, db1: Db1Runtime, visual=None) -> CoreChainResult:",
                          "                   sam2: Sam2Runtime, db1: Db1Runtime, visual=None, observer=None) -> CoreChainResult:")
    source = replace_once(source, "    features = sam2.encode(rgb_context) if visual is None else visual", """    if observer is not None:
        _trace_emit(observer, 5, "RUNNING")
    features = sam2.encode(rgb_context) if visual is None else visual""")
    source = replace_once(source, "    timings[\"sam2\"] = round(time.time() - started, 3)", """    timings["sam2"] = round(time.time() - started, 3)
    if observer is not None:
        _trace_emit(observer, 5, "COMPLETED", {"encoding_complete": True, "feature_shape": list(features.shape),
                    "elapsed_seconds": timings["sam2"], "raw_feature_tensor_saved": False})
        _trace_emit(observer, 6, "RUNNING")""")
    source = replace_once(source, "    outputs = db1.forward(features, direction, directional, nearest)", """    if observer is not None:
        _trace_emit(observer, 6, "RUNNING", {"source": "actual frozen field calculation; W/A pending actual forward"},
                    {"P_dir": fields["P_dir_64"], "P_near": fields["P_near_64"]})
    if observer is None:
        outputs = db1.forward(features, direction, directional, nearest)
    else:
        outputs = db1.forward(features, direction, directional, nearest, observer=observer)""")
    source = replace_once(source, "    timings[\"db1\"] = round(time.time() - started, 3)", """    timings["db1"] = round(time.time() - started, 3)
    if observer is not None:
        _trace_emit(observer, 8, "COMPLETED", {"candidate_only": True, "validation_state": "NOT_YET_GUARDED",
                    "mask_pixels": int(outputs["mask"].sum()), "elapsed_seconds": timings["db1"]},
                    {"logits": outputs["logits"], "probability": outputs["probability"], "mask": outputs["mask"]})""")
    return source


def competition_patch(source):
    source = replace_once(source, "import torch.nn.functional as F\n", "import torch.nn.functional as F\nfrom buildreasonseg.runtime.observation import active_observer as _trace_active, emit_current as _trace_current\n")
    source = replace_once(source, "    return attention, mass", """    if _trace_active() is not None:
        _trace_current(6, "COMPLETED", {"source": "actual D-B1 forward field_competition",
                       "field_shapes": {"P_dir": list(directional.shape), "P_near": list(nearest.shape),
                                         "W": list(weight.shape), "A": list(attention.shape)}},
                       {"P_dir": directional, "P_near": nearest, "W": weight, "A": attention})
        _trace_current(7, "RUNNING")
    return attention, mass""")
    source = replace_once(source, "        return CompetitionState(attention=attention, attention_vis=attention_vis, prototype=prototype,", """        if _trace_active() is not None:
            _trace_current(7, "COMPLETED", {"q_formed": prototype is not None,
                           "q_shape": list(prototype.shape) if prototype is not None else None,
                           "q_dimension": int(prototype.shape[-1]) if prototype is not None else None,
                           "C_shape": list(similarity.shape) if similarity is not None else None,
                           "source": "actual D-B1 forward competition state",
                           "source_dtype": str(similarity.dtype) if similarity is not None else None,
                           "similarity_scope": "visual cosine similarity, not target correctness probability"},
                           {"C": similarity})
            _trace_current(8, "RUNNING")
        return CompetitionState(attention=attention, attention_vis=attention_vis, prototype=prototype,""")
    return source


class RemoveObservation(ast.NodeTransformer):
    """Erase only explicit observation additions, retaining the observer=None scientific path."""
    def __init__(self, enabled=False):
        self.enabled = enabled
    def visit_ImportFrom(self, node):
        if node.module == "buildreasonseg.runtime.observation":
            return None
        return node

    def visit_Assign(self, node):
        if all(isinstance(t, ast.Name) and t.id.startswith("_trace_") for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        text = ast.unparse(node.test)
        if text == "observer is None":
            return self._statements(node.orelse if self.enabled else node.body)
        if text == "observer is not None" or text == "_trace_active() is not None":
            return None
        return self.generic_visit(node)

    def visit_With(self, node):
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call) and ast.unparse(node.items[0].context_expr.func) == "_trace_scope":
            return self._statements(node.body)
        return self.generic_visit(node)

    def _statements(self, statements):
        result = []
        for item in statements:
            visited = self.visit(item)
            if isinstance(visited, list):
                result.extend(visited)
            elif visited is not None:
                result.append(visited)
        return result

    def visit_Call(self, node):
        node.keywords = [keyword for keyword in node.keywords if keyword.arg != "observer"]
        return self.generic_visit(node)

    def visit_arguments(self, node):
        kept = [(arg, default) for arg, default in zip(node.kwonlyargs, node.kw_defaults) if arg.arg != "observer"]
        node.kwonlyargs = [arg for arg, default in kept]
        node.kw_defaults = [default for arg, default in kept]
        return self.generic_visit(node)


def audit_scientific_ast(original: bytes, patched: bytes):
    old = ast.parse(original.decode("utf-8-sig"))
    modes = {}
    for enabled in (False, True):
        new = RemoveObservation(enabled).visit(ast.parse(patched.decode("utf-8-sig")))
        same = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
        if not same:
            raise ValueError("Scientific AST differs in observer enabled/disabled path; STOP")
        modes["enabled" if enabled else "disabled"] = True
    return {"pass": True, "paths": modes, "method": "Full module AST equality for both observer paths after erasing only observation transport/metadata and observer keyword."}


def patched_sources(blobs: dict[str, bytes]):
    result, records = {}, []
    for relative, function in zip(FILES, (pipeline_patch, core_patch, competition_patch)):
        original = blobs[relative]
        patched = function(original.decode("utf-8-sig")).encode("utf-8")
        audit = audit_scientific_ast(original, patched)
        result[relative] = patched
        records.append({"path": relative, "provenance": "ACCEPTED_GIT_CANONICAL_BLOB_PLUS_EXPLICIT_V2_OBSERVATION_ONLY_EDITS",
                        "base_bytes": len(original), "base_sha256": hashlib.sha256(original).hexdigest(),
                        "v2_bytes": len(patched), "v2_sha256": hashlib.sha256(patched).hexdigest(), "scientific_ast_audit": audit,
                        "unified_diff": "".join(difflib.unified_diff(original.decode().splitlines(True), patched.decode().splitlines(True),
                                                                   fromfile="accepted/"+relative, tofile="v2/"+relative))})
    result[PREFIX + "observation.py"] = HELPER.read_bytes()
    records.append({"path": PREFIX + "observation.py", "provenance": "NEW_V2_READ_ONLY_OBSERVATION_TRANSPORT_NO_SCIENTIFIC_ALGORITHM",
                    "v2_bytes": len(result[PREFIX+"observation.py"]), "v2_sha256": hashlib.sha256(result[PREFIX+"observation.py"]).hexdigest()})
    return result, records
