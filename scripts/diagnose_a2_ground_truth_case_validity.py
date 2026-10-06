"""Read-only A2 provenance audit. No model library or inference entry point is used.

Use the existing .conda/buildreasonseg-mvp/python.exe with -B.
--phase provenance saves history, identity and the plan BEFORE any corpus search.
--phase provenance --search executes only that saved exact-pixel plan.
--phase gt refuses non-exact evidence. --phase finalize validates saved evidence.
Writes are restricted to this milestone's JSON, report and executor handoff.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
WHU = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
EVIDENCE = ROOT / "evaluation/task8b3_a2_ground_truth_case_validity_audit_v1.json"
REPORT = ROOT / "docs/task8b3_a2_ground_truth_case_validity_audit_v1.md"
TASK = "A2_GROUND_TRUTH_AND_LOCKED_CASE_VALIDITY_AUDIT_V1"
START = "104bcde03ff8bedbd563a1c0eba6fe230f221dad"
BRANCH = "audit/task8b3-a2-ground-truth-case-validity-v1"
A2 = EXTERNAL / "inference/input/A2.png"
LOCK = "10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f"
ALLOWED = {
    "scripts/diagnose_a2_ground_truth_case_validity.py",
    "docs/task8b3_a2_ground_truth_case_validity_audit_v1.md",
    "evaluation/task8b3_a2_ground_truth_case_validity_audit_v1.json",
    "evaluation/task8b3_a2_ground_truth_case_validity_audit_v1_overlay.png",
    "handoff/CURRENT_TASK.md", "handoff/EXECUTOR_STATE.yaml",
}
ENUMS = {
    "provenance_status": {"EXACT_DOCUMENTARY_AND_PIXEL_CONFIRMED", "EXACT_PIXEL_CONFIRMED",
                          "PARTIAL_EXACT_EVIDENCE", "NOT_ESTABLISHED", "CONFLICTING_PROVENANCE_EVIDENCE"},
    "gt_status": {"GT_AVAILABLE_ALIGNED", "GT_ALIGNMENT_CONFLICT", "GT_UNAVAILABLE",
                  "NOT_EVALUATED_NO_EXACT_PROVENANCE"},
    "case_validity": {"VALID_REFERENCE_RELATION_TARGET_CHAIN", "INVALID_NO_BUILDING_INSTANCES",
                      "INVALID_NO_VALID_REFERENCE", "INVALID_NO_LEFT_OF_TARGET", "INVALID_NO_NEAREST_TARGET",
                      "AMBIGUOUS_GT_RELATION", "SEMANTICS_NOT_PORTABLE_TO_A2", "NOT_EVALUABLE_WITHOUT_PROVENANCE"},
    "prop01_interpretation_candidate": {"VALID_GT_CASE_DETECTOR_ZERO_PERSISTS",
                                      "LOCKED_CASE_CONSTRUCTION_DEFECT_CANDIDATE",
                                      "IMAGE_GT_ALIGNMENT_DEFECT_CANDIDATE", "PROP01_ROOT_CAUSE_REMAINS_UNRESOLVED"},
}
GUARDS = ("no_model_inference", "no_threshold_sweep", "no_parameter_tuning", "no_product_change",
          "no_external_write", "no_candidate_replacement")
PLAN = {
    "coordinate_convention": "[x,y,width,height], zero-based decoded RGB; no transformations",
    "C1_quadrants": [[0, 0, 512, 512], [512, 0, 512, 512], [0, 512, 512, 512], [512, 512, 512, 512]],
    "C2_interior_windows": [[256, 0, 512, 512], [0, 256, 512, 512], [256, 256, 512, 512],
                            [512, 256, 512, 512], [256, 512, 512, 512]],
    "C1_C2_corpus": "ALL source_image_ref paths in tracked native-vector tiles/index.jsonl; verify complete cropped image inventory",
    "C3_patches": [[256, 256, 16, 16], [496, 496, 16, 16], [752, 752, 16, 16]],
    "C3_corpus": ["test.tif", "train1.tif", "train2.tif"],
    "C3_method": "scan uncompressed RGB TIFF tiles for BOTH exact 8-pixel halves of the patch first row; verify entire 16x16 patch",
    "C3_boundary_coverage": "a 16-pixel row can cross at most one 128-pixel tile boundary; at least one 8-pixel half is contiguous",
    "C3_chunk_bytes": 33554432,
    "C3_max_seconds_total": 600,
    "C3_max_anchor_occurrences": 10000,
    "C4": "reconstruct complete in-bounds 1024x1024 source window and compare every decoded RGB byte",
    "exact_only": True, "fuzzy_or_perceptual_method": False,
    "transformations_searched": [],
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def save(data):
    EVIDENCE.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8", errors="replace").strip()


def identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": file_hash(path)}


def inventory(root):
    # Only stat operations; content hashes separately identify consulted files.
    rows = [[p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob("*")) if p.is_file()]
    return {"root": str(root), "files": len(rows), "total_bytes": sum(r[1] for r in rows),
            "relative_path_size_mtime_ns_sha256": digest(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()),
            "definition": "SHA256 of sorted JSON [relative_path,bytes,mtime_ns] rows; read access time excluded"}


def rgb_locked():
    import numpy as np
    from PIL import Image
    with Image.open(A2) as handle:
        assert handle.size == (1024, 1024) and handle.mode == "RGB"
        rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    assert file_hash(A2) == LOCK
    return rgb


def source_record(path, classification, finding, line_range=None, commit=None):
    payload = (git("show", f"{commit}:{path}").encode() if commit else Path(path).read_bytes())
    return {"source": str(path), "commit": commit, "sha256": digest(payload),
            "hash_basis": "git-show UTF8 text without trailing whitespace" if commit else "file bytes",
            "classification": classification, "finding": finding, "line_range": line_range}


def historical_search():
    # Retain a bounded excerpt per matching line; the file hash + line number is the full source pointer.
    searches = []
    for roots, globs in [(["docs", "scripts", "handoff", "evaluation"], ["*.md", "*.py", "*.json", "*.yaml"]),
                         ([str(EXTERNAL / "logs")], ["*.md", "*.py", "*.json", "*.txt", "*.log"])]:
        cmd = ["rg", "-n", "-i", "A2\\.png|10286b1e|A2.*(origin|provenance|source|crop|来源|裁剪)", *roots]
        for pattern in globs:
            cmd.extend(["-g", pattern])
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True)
        assert result.returncode in (0, 1), result.stderr.decode(errors="replace")
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()
        # Current milestone files cannot corroborate historical provenance.
        lines = [line for line in lines if not any(x in line.split(":", 1)[0] for x in
                 ("diagnose_a2_ground_truth_case_validity", "task8b3_a2_ground_truth_case_validity"))
                 and not line.startswith("handoff\\CURRENT_TASK")]
        searches.append({"command": cmd, "exit": result.returncode, "matching_lines": len(lines),
                         "output_sha256": digest(result.stdout), "excerpts": [line[:420] for line in lines]})
    history = git("log", "--all", "--reverse", "--format=%H %ad %an %s", "--date=iso-strict",
                  "-G", "A2.png|10286b1e", "--", "handoff/TO_DSH.md")
    versions = []
    for line in history.splitlines():
        commit = line.split()[0]
        text = git("show", f"{commit}:handoff/TO_DSH.md")
        versions.append({"commit_record": line, "path": "handoff/TO_DSH.md",
                         "text_sha256": digest(text.encode()),
                         "A2_lines": [{"line": i, "text": t[:420]} for i, t in enumerate(text.splitlines(), 1)
                                      if "A2" in t or LOCK in t]})
    return {"searches": searches, "historical_task_versions": versions,
            "suite_introduction_log": git("log", "--all", "--reverse", "--format=%H %ad %an %s",
                                          "--date=iso-strict", "--", "scripts/task8b3_interactive_suite.py"),
            "tracked_A2_path_history": git("log", "--all", "--format=%H %s", "--", "*A2*"),
            "P0_established": False,
            "conclusion": "Inspected records fix file identity and intended program; no authoritative reproducible source raster/window record found."}


def write_state(data, step, status="IN_PROGRESS"):
    import yaml
    previous = yaml.safe_load((ROOT / "handoff/EXECUTOR_STATE.yaml").read_text(encoding="utf-8-sig"))
    previous.update(task_id=TASK, status=status, branch=BRANCH, head=git("rev-parse", "HEAD"),
                    completed=data["completed"], current_step=step,
                    next_action="CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT" if status == "READY_FOR_SUPERVISOR_AUDIT" else step,
                    tests_run=data.get("validation", {}).get("checks", []),
                    files_modified=sorted(p for p in ALLOWED if (ROOT / p).exists()
                                          and "overlay" not in p),
                    uncommitted_changes=False, blocked_on_supervisor=False,
                    block_reason=None, decision_level_required=None,
                    last_checkpoint={"commit": git("rev-parse", "HEAD"),
                                     "pushed": git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}")
                                     if git("branch", "-r", "--list", f"origin/{BRANCH}") else False},
                    provenance_status=data["provenance_status"], gt_status=data["gt_status"],
                    case_validity=data["case_validity"], PROP01_status="OPEN", **{g: True for g in GUARDS})
    (ROOT / "handoff/EXECUTOR_STATE.yaml").write_text(yaml.safe_dump(previous, sort_keys=False, allow_unicode=True), encoding="utf-8")


def report(data):
    c = data.get("exact_search_coverage", {})
    text = f"""# A2 ground-truth and locked-case validity audit V1

Task: {TASK}. Status: {data['status']}.
Starting branch: fix/task8b3-detector-rgb-bgr-contract-v1, remote HEAD `{START}`.
Task branch: `{BRANCH}`. All findings are evidence for Supervisor review; PROP-01 remains OPEN.

## Result dimensions

- provenance_status: `{data['provenance_status']}`
- gt_status: `{data['gt_status']}`
- case_validity: `{data['case_validity']}`
- prop01_interpretation_candidate: `{data['prop01_interpretation_candidate']}`

## History and locked identity

Current external A2 file SHA256: `{data['locked_a2_file_sha256']}`.
Decoded RGB SHA256: `{data['locked_a2_decoded_rgb_sha256']}`; 1024 x 1024 RGB uint8, 1,677,040 bytes.
Actual external runtime imageio.load_image output equals direct Pillow RGB byte-for-byte.
File timestamp metadata is descriptive only and does not establish source provenance.

The first tracked suite definition is commit `1287d0a2bab50a278452d4cd0ec8d492347afdfe`
(2026-10-03, author recorded by Git). Its Supervisor task book specifies the A2 path, prompt and
largest_to_left_of_to_nearest program. This establishes introduction into the tracked suite,
not who created the original image. The pre-existing RC1 Task 8B.2 inventory already lists A2.
The R4B freeze/results fix the input hash and program, not a GT-backed construction.
P1D5, P1D8 and P1D10 records report unresolved origin; prior supported-domain conclusions are
not adopted. Search commands, inspected task-book versions, source hashes and exact line pointers
are saved in the JSON. No authoritative source raster/window or transformation chain was recovered.
The reason for selecting these exact pixels is not established by the inspected documentary records.

## Predeclared exact search

C1: four disjoint 512 x 512 quadrants. C2: five additional fixed 512 x 512 windows.
C3: three fixed 16 x 16 patches at (256,256), (496,496), (752,752), with both first-row
8-pixel halves searched exactly. The saved plan predates corpus execution and is checkpointed in Git.
The two halves cover tile-boundary crossings; every candidate must pass full patch equality.
C4 requires complete in-bounds 1024 x 1024 RGB equality. No transformations, fuzzy matching,
perceptual comparisons, threshold fitting or model calls are used.

Search coverage: `{json.dumps(c, ensure_ascii=False)}`.
Exact matches: {len(data['exact_matches'])}. Full-window records: {len(data['full_window_verification'])}.
Limitations and unavailable-field reasons are saved explicitly in the JSON.

## Conditional GT and frozen semantics

GT mapping, image/GT alignment and relation classification require exact full-window provenance.
Native-vector tile index and Task 6K.1 mapping/validation evidence are inspected read-only.
Frozen lineage: spatial_reasoning/annotator.py generate_level3; semantic_policy.py
resolve_size_extreme, direction_candidates_over_visible and resolve_nearest; geometry.py
boundary distance; configs/spatial_relations_v1.yaml; BuildSpatialReason v0.2 manifest.
Global visible largest is checked for frozen ambiguity/eligibility; left_of is the frozen
subject-relative direction predicate; nearest is boundary distance over the full direction set,
with frozen margin/eligibility, without substituting an eligible runner-up.
No A2 GT-derived answer or 1024-window portability decision is made without exact provenance.
No overlay is created without exact GT-backed provenance.

## Validation and claim boundary

Validation: {json.dumps(data.get('validation', {}), ensure_ascii=False)}.
The audit makes no domain reclassification, detector improvement, semantic segmentation,
PROP-01 closure or replacement-case claim. The accepted detector zero result is historical evidence;
no inference is repeated. Only the authorized diagnostic, report, JSON and two handoff paths change.
External RC1 and WHU archive inventory identities are checked before and after read-only work.
No package installation, data/model download, external write or product edit occurs.

Next gate: CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT. STOP at READY_FOR_SUPERVISOR_AUDIT.
"""
    REPORT.write_text(text, encoding="utf-8")


def prepare():
    import numpy as np
    from PIL import Image
    assert not EVIDENCE.exists(), "Existing evidence must not be overwritten by prepare"
    assert git("branch", "--show-current") == BRANCH and git("merge-base", START, "HEAD") == START
    sys.path.insert(0, str(EXTERNAL))
    from buildreasonseg.runtime.imageio import load_image
    rgb = rgb_locked()
    loaded = load_image(A2)
    assert np.array_equal(rgb, loaded.rgb)
    with Image.open(A2) as handle:
        info = {"mode": handle.mode, "size": list(handle.size), "info": dict(handle.info)}
    stat = A2.stat()
    sources = [
        source_record("handoff/TO_DSH.md", "AUTHORITATIVE", "Frozen A2 suite path/prompt/program, no source crop.", [112, 118],
                      "1287d0a2bab50a278452d4cd0ec8d492347afdfe"),
        source_record("scripts/task8b3_interactive_suite.py", "AUTHORITATIVE", "First tracked case tuple, no source construction.", [50, 52],
                      "1287d0a2bab50a278452d4cd0ec8d492347afdfe"),
    ]
    for rel, finding in [
        ("docs/task8b3_six_image_demo_suite.md", "R4B freeze identifies exact A2 bytes/program; no original raster/window."),
        ("docs/task8b3_p1d5_detector_provenance_domain_gap.md", "Prior documentary audit: origin NOT ESTABLISHED."),
        ("docs/task8b3_p1d8_a2_input_domain_audit.md", "Historical whole-file search and photometric audit; no source mapping."),
        ("docs/task8b3_p1d10_prop01_resolution_decision.md", "Prior input provenance and GT availability NOT ESTABLISHED."),
        ("docs/task8b3_prop01_a2_zero_proposals_forensics_v1.md", "Locked identity and accepted earlier engineering evidence."),
        ("docs/task8b3_detector_rgb_bgr_contract_repair_v1.md", "Accepted corrected zero detector result; not rerun."),
    ]:
        sources.append(source_record(ROOT / rel, "DESCRIPTIVE_ONLY", finding))
    for rel, finding in [
        ("logs/task8b2_r1_delivery_inventory.txt", "Pre-existing inventory lists A2, 1677040 bytes; no original source."),
        ("logs/task8b3_suite_results.json", "Existing suite input hash/program and failure records."),
        ("logs/task8b3_transcripts/A2.txt", "Preserved transcript; no creation provenance."),
        ("logs/task8b3_p1d8_domain_audit.json", "Historical domain audit retained, not repeated."),
        ("logs/build_scripts/task8b3_suite.py", "Earlier untracked driver consumes A2; does not construct it."),
    ]:
        sources.append(source_record(EXTERNAL / rel, "DESCRIPTIVE_ONLY", finding))
    semantics_paths = ["spatial_reasoning/annotator.py", "spatial_reasoning/semantic_policy.py",
                       "spatial_reasoning/relations.py", "spatial_reasoning/geometry.py",
                       "configs/spatial_relations_v1.yaml", "configs/build_spatial_reason_v0.2.yaml",
                       "datasets/build_spatial_reason/v0.2/manifest.json"]
    data = dict(task_id=TASK, status="IN_PROGRESS", starting_branch="fix/task8b3-detector-rgb-bgr-contract-v1",
                starting_head=START, verified_starting_remote_head=START, task_branch=BRANCH,
                locked_a2_path=str(A2), locked_a2_file_sha256=LOCK,
                locked_a2_decoded_rgb_sha256=digest(rgb.tobytes()),
                locked_a2_identity={**identity(A2), **info, "dtype": str(rgb.dtype),
                    "runtime_decode_equal_pillow_rgb": True, "runtime_source": identity(EXTERNAL / "buildreasonseg/runtime/imageio.py"),
                    "creation_time_ns": stat.st_ctime_ns, "last_write_time_ns": stat.st_mtime_ns,
                    "timestamp_evidence_class": "DESCRIPTIVE_ONLY"},
                historical_sources_inspected=sources, historical_documentary_search=historical_search(),
                source_roots_inspected=[str(ROOT), str(EXTERNAL), str(WHU),
                                       "datasets/whu_native_vector/v1.0", "datasets/build_spatial_reason/v0.2"],
                exact_search_plan=PLAN, exact_search_anchors=[], exact_matches=[], full_window_verification=[],
                provenance_status="NOT_ESTABLISHED", resolved_source_raster=None, resolved_source_window=None,
                gt_source_identities=[], gt_building_instance_count=None, gt_instances=[],
                relation_semantics_source=[identity(ROOT / p) for p in semantics_paths],
                gt_reference=None, gt_left_of_candidates=[], gt_target=None,
                gt_status="NOT_EVALUATED_NO_EXACT_PROVENANCE", case_validity="NOT_EVALUABLE_WITHOUT_PROVENANCE",
                prop01_interpretation_candidate="PROP01_ROOT_CAUSE_REMAINS_UNRESOLVED",
                unavailable_fields_reason="No exact full-window provenance; GT recovery/alignment/relation application are gated.",
                completed=["Required startup and remote anchor verified; task book installed byte-for-byte.",
                           "Phase A historical documentary search complete; no authoritative source/window recovered.",
                           "Phase B locked SHA256 and runtime/Pillow RGB equality verified.",
                           "Exact-pixel plan predeclared before corpus search."],
                read_only_integrity_before=[inventory(EXTERNAL), inventory(WHU)],
                inspected_mapping_sources=[identity(ROOT / p) for p in [
                    "datasets/whu_native_vector/v1.0/tiles/index.jsonl", "datasets/whu_native_vector/v1.0/manifest.json",
                    "evaluation/task6k1_tile_mapping.json", "evaluation/task6k1_vector_raster_alignment.json"]],
                validation={}, **{guard: True for guard in GUARDS})
    for phase, windows in [("C1", PLAN["C1_quadrants"]), ("C2", PLAN["C2_interior_windows"]), ("C3", PLAN["C3_patches"])]:
        for i, (x, y, w, h) in enumerate(windows):
            data["exact_search_anchors"].append({"id": f"{phase}_{i}", "window": [x, y, w, h],
                                                  "decoded_rgb_sha256": digest(rgb[y:y+h, x:x+w].tobytes())})
    save(data)
    report(data)
    write_state(data, "CHECKPOINT_1_HISTORY_IDENTITY_AND_PREDECLARED_PLAN")
    print("Phase A/B complete; search plan saved before execution", flush=True)


class ExactTiledRGB:
    """Bounded read-only decoder for this archive's uncompressed RGB TIFF tiles.

    Parses BigTIFF LONG8 arrays (not supported by the older metadata-only helper).
    No acceptance rule or existing mapping definition is changed.
    """
    def __init__(self, path):
        import numpy as np
        self.path = Path(path)
        self.handle = self.path.open("rb")
        self.mm = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        header = self.mm[:16]
        assert header[:4] == b"II+\x00" and struct.unpack_from("<H", header, 4)[0] == 8
        offset = struct.unpack_from("<Q", header, 8)[0]
        count = struct.unpack_from("<Q", self.mm, offset)[0]
        tags = {}
        for i in range(count):
            pos = offset + 8 + i * 20
            tag, kind, n = struct.unpack_from("<HHQ", self.mm, pos)
            if tag not in (256, 257, 258, 259, 262, 274, 277, 284, 317, 322, 323, 324, 325):
                continue
            fmt, unit = {3: ("H", 2), 4: ("I", 4), 16: ("Q", 8)}[kind]
            where = pos + 12 if n * unit <= 8 else struct.unpack_from("<Q", self.mm, pos + 12)[0]
            tags[tag] = list(struct.unpack_from("<" + str(n) + fmt, self.mm, where))
        self.tags = tags
        self.width, self.height = tags[256][0], tags[257][0]
        self.tw, self.th = tags[322][0], tags[323][0]
        self.cols = (self.width + self.tw - 1) // self.tw
        self.rows = (self.height + self.th - 1) // self.th
        self.offsets, self.sizes = tags[324], tags[325]
        assert tags[258] == [8, 8, 8] and tags[259] == [1] and tags[262] == [2]
        assert tags[277] == [3] and tags.get(284, [1]) == [1] and tags.get(317, [1]) == [1]
        assert tags.get(274, [1]) == [1]
        assert len(self.offsets) == self.cols * self.rows == len(self.sizes)
        assert all(n == self.tw * self.th * 3 for n in self.sizes)
        assert all(a + n <= b for a, n, b in zip(self.offsets, self.sizes, self.offsets[1:]))
        assert self.offsets[-1] + self.sizes[-1] <= len(self.mm)
        self.np = np

    def window(self, x, y, w, h):
        assert 0 <= x <= self.width - w and 0 <= y <= self.height - h
        out = self.np.empty((h, w, 3), dtype=self.np.uint8)
        for ty in range(y // self.th, (y + h - 1) // self.th + 1):
            for tx in range(x // self.tw, (x + w - 1) // self.tw + 1):
                index = ty * self.cols + tx
                tile = self.np.frombuffer(self.mm, dtype=self.np.uint8, count=self.sizes[index],
                                          offset=self.offsets[index]).reshape(self.th, self.tw, 3)
                lx, ly = max(x, tx * self.tw), max(y, ty * self.th)
                rx, ry = min(x + w, (tx + 1) * self.tw), min(y + h, (ty + 1) * self.th)
                out[ly-y:ry-y, lx-x:rx-x] = tile[ly-ty*self.th:ry-ty*self.th, lx-tx*self.tw:rx-tx*self.tw]
        return out

    def pixel_at_byte(self, pos, length_pixels):
        index = bisect.bisect_right(self.offsets, pos) - 1
        if index < 0:
            return None
        delta = pos - self.offsets[index]
        if delta % 3 or delta + length_pixels * 3 > self.sizes[index]:
            return None
        py, px = divmod(delta // 3, self.tw)
        if px + length_pixels > self.tw:
            return None
        x, y = (index % self.cols) * self.tw + px, (index // self.cols) * self.th + py
        return (x, y) if x + length_pixels <= self.width and y < self.height else None

    def close(self):
        self.mm.close()
        self.handle.close()


def verify_full(data, raster, origin, rgb, evidence_id):
    import numpy as np
    x, y = origin
    record = {"source_raster": str(raster.path), "window": [x, y, 1024, 1024], "candidate_from": evidence_id}
    if not (0 <= x <= raster.width - 1024 and 0 <= y <= raster.height - 1024):
        record.update(equal=False, reason="candidate full window outside source raster")
    else:
        candidate = raster.window(x, y, 1024, 1024)
        record.update(equal=bool(np.array_equal(candidate, rgb)), decoded_rgb_sha256=digest(candidate.tobytes()),
                      unequal_channel_bytes=int(np.count_nonzero(candidate != rgb)))
    if not any(r["source_raster"] == record["source_raster"] and r["window"] == record["window"]
               for r in data["full_window_verification"]):
        data["full_window_verification"].append(record)
    if record["equal"]:
        if data["resolved_source_raster"] is not None and (data["resolved_source_raster"], data["resolved_source_window"]) != (str(raster.path), record["window"]):
            raise RuntimeError("CONFLICTING_PROVENANCE_EVIDENCE: multiple distinct full-window sources")
        data.update(provenance_status="EXACT_PIXEL_CONFIRMED", resolved_source_raster=str(raster.path),
                    resolved_source_window=record["window"])


def search():
    import numpy as np
    from PIL import Image
    data = read()
    assert data["exact_search_plan"] == PLAN
    assert not data.get("exact_search_coverage"), "Search already recorded; do not silently rerun"
    plan_commit = git("rev-parse", "HEAD")
    committed = json.loads(git("show", f"{plan_commit}:{EVIDENCE.relative_to(ROOT).as_posix()}"))
    assert committed["exact_search_plan"] == PLAN and not committed.get("exact_search_coverage")
    data["exact_search_plan_checkpoint"] = plan_commit
    rgb = rgb_locked()
    rows = [json.loads(line) for line in (ROOT / "datasets/whu_native_vector/v1.0/tiles/index.jsonl").read_text().splitlines()]
    refs = {r["source_image_ref"] for r in rows}
    actual = {p.relative_to(WHU).as_posix() for p in (WHU / "1. The cropped image data and raster labels").glob("*/image/*.tif")}
    assert refs == actual and len(refs) == len(rows) == 17388
    index_by_ref = {r["source_image_ref"]: r for r in rows}
    hashes = []
    tile_digest = hashlib.sha256()
    tile_matches = []
    anchors = [a for a in data["exact_search_anchors"] if a["id"].startswith(("C1", "C2"))]
    anchor_by_hash = {}
    for a in anchors:
        anchor_by_hash.setdefault(a["decoded_rgb_sha256"], []).append(a)
    counts = {}
    for i, ref in enumerate(sorted(refs), 1):
        with Image.open(WHU / ref) as handle:
            assert handle.size == (512, 512) and handle.mode == "RGB"
            tile = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        sha = digest(tile.tobytes())
        hashes.append((ref, sha))
        tile_digest.update(json.dumps([ref, sha], ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
        r = index_by_ref[ref]
        counts[r["source_raster"]] = counts.get(r["source_raster"], 0) + 1
        for a in anchor_by_hash.get(sha, []):
            x, y, w, h = a["window"]
            assert np.array_equal(tile, rgb[y:y+h, x:x+w])
            tile_matches.append({"anchor_id": a["id"], "source_image_ref": ref, "decoded_rgb_sha256": sha,
                                 "tile_id": r["tile_id"], "source_raster": r["source_raster"],
                                 "source_origin": [r["grid_column"] * 512 - x, r["grid_row"] * 512 - y],
                                 "equality_verified": True})
        if i % 2000 == 0:
            print(f"C1/C2 decoded {i}/17388 canonical tiles", flush=True)
    data["exact_matches"].extend(tile_matches)
    data["exact_search_coverage"] = {"C1": {"tiles": len(hashes), "counts_by_raster": counts,
        "all_available_cropped_images_indexed": True, "errors": [], "matches": sum(m["anchor_id"].startswith("C1") for m in tile_matches),
        "ordered_ref_decoded_rgb_hash_index_sha256": tile_digest.hexdigest()},
        "C2": {"tiles": len(hashes), "windows": 5, "matches": sum(m["anchor_id"].startswith("C2") for m in tile_matches),
               "execution": "same complete decoded index, predeclared windows; no adaptive locations"}, "C3": []}
    readers = {}
    try:
        for match in tile_matches:
            name = match["source_raster"] + ".tif"
            raster = readers.setdefault(name, None)
            if raster is None:
                raster = readers[name] = ExactTiledRGB(WHU / "3. The whole area image dataset/image" / name)
            verify_full(data, raster, match["source_origin"], rgb, match["anchor_id"])
        if data["resolved_source_raster"] is None:
            patch_search(data, rgb, readers)
    finally:
        for raster in readers.values():
            raster.close()
    if data["resolved_source_raster"] is None:
        data["provenance_status"] = "PARTIAL_EXACT_EVIDENCE" if data["exact_matches"] else "NOT_ESTABLISHED"
    data["completed"].append("Phase C exact canonical tile and bounded whole-source patch search completed; coverage saved.")
    save(data)
    report(data)
    write_state(data, "CHECKPOINT_2_EXACT_PROVENANCE_SEARCH_COMPLETE")
    print(data["provenance_status"], flush=True)


def patch_search(data, rgb, readers):
    import numpy as np
    started = time.monotonic()
    occurrences = 0
    needles = []
    for a in data["exact_search_anchors"]:
        if a["id"].startswith("C3"):
            x, y, w, h = a["window"]
            for half in (0, 8):
                needles.append((rgb[y, x+half:x+half+8].tobytes(), a, half))
    seen = set()
    for name in PLAN["C3_corpus"]:
        if time.monotonic() - started >= PLAN["C3_max_seconds_total"]:
            data["exact_search_coverage"]["C3"].append({"raster": name, "status": "NOT_STARTED_TIME_BOUND"})
            continue
        raster = readers.get(name)
        if raster is None:
            raster = readers[name] = ExactTiledRGB(WHU / "3. The whole area image dataset/image" / name)
        # Decoder is checked against pre-existing pixel-validated canonical tiles.
        tile_rows = [json.loads(line) for line in (ROOT / "datasets/whu_native_vector/v1.0/tiles/index.jsonl").read_text().splitlines()
                     if json.loads(line)["source_raster"] == name[:-4]]
        from PIL import Image
        checks = []
        for row in (tile_rows[0], tile_rows[len(tile_rows)//2], tile_rows[-1]):
            source = np.asarray(Image.open(WHU / row["source_image_ref"]).convert("RGB"), dtype=np.uint8)
            x, y = row["grid_column"]*512, row["grid_row"]*512
            same = bool(np.array_equal(source, raster.window(x, y, 512, 512)))
            checks.append({"tile_id": row["tile_id"], "window": [x, y, 512, 512], "equal": same})
            assert same, "EXACT_DECODER_CONTRADICTS_VALIDATED_TILE_MAPPING"
        coverage = {"raster": name, "bytes": len(raster.mm), "bytes_scanned": 0,
                    "dimensions": [raster.width, raster.height], "tile_dimensions": [raster.tw, raster.th],
                    "tile_count": len(raster.offsets), "decoder_exact_checks": checks,
                    "anchor_occurrences": 0, "patch_candidates_checked": 0, "exact_patch_hits": 0,
                    "status": "IN_PROGRESS"}
        data["exact_search_coverage"]["C3"].append(coverage)
        sha = hashlib.sha256()
        step = PLAN["C3_chunk_bytes"]
        for begin in range(0, len(raster.mm), step):
            if time.monotonic() - started >= PLAN["C3_max_seconds_total"] or occurrences >= PLAN["C3_max_anchor_occurrences"]:
                coverage["status"] = "BOUNDED_PARTIAL_COVERAGE"
                break
            end = min(begin + step, len(raster.mm))
            chunk = raster.mm[begin:min(end + 23, len(raster.mm))]
            sha.update(chunk[:end-begin])
            for needle, a, half in needles:
                at = chunk.find(needle)
                while at >= 0 and begin + at < end:
                    occurrences += 1
                    coverage["anchor_occurrences"] += 1
                    if occurrences >= PLAN["C3_max_anchor_occurrences"]:
                        coverage["status"] = "BOUNDED_PARTIAL_COVERAGE"
                        break
                    pixel = raster.pixel_at_byte(begin + at, 8)
                    if pixel is not None:
                        ax, ay, w, h = a["window"]
                        px, py = pixel[0] - half, pixel[1]
                        key = (name, a["id"], px, py)
                        if key not in seen and 0 <= px <= raster.width-w and 0 <= py <= raster.height-h:
                            seen.add(key)
                            coverage["patch_candidates_checked"] += 1
                            patch = raster.window(px, py, w, h)
                            if np.array_equal(patch, rgb[ay:ay+h, ax:ax+w]):
                                coverage["exact_patch_hits"] += 1
                                hit = {"anchor_id": a["id"], "source_raster": str(raster.path),
                                       "patch_window": [px, py, w, h], "source_origin": [px-ax, py-ay],
                                       "decoded_rgb_sha256": digest(patch.tobytes()), "equality_verified": True}
                                data["exact_matches"].append(hit)
                                verify_full(data, raster, hit["source_origin"], rgb, a["id"])
                    at = chunk.find(needle, at + 1)
            coverage["bytes_scanned"] = end
            if end % (step * 8) == 0:
                print(f"C3 {name}: {end}/{len(raster.mm)} bytes, exact patch hits={coverage['exact_patch_hits']}", flush=True)
        else:
            coverage["status"] = "COMPLETE_ALL_BYTES"
            coverage["source_file_sha256"] = sha.hexdigest()
        coverage["seconds_total_elapsed"] = round(time.monotonic() - started, 3)
        save(data)


def gt():
    data = read()
    if data["provenance_status"] not in ("EXACT_DOCUMENTARY_AND_PIXEL_CONFIRMED", "EXACT_PIXEL_CONFIRMED"):
        raise SystemExit("GT_REFUSED_NO_EXACT_FULL_WINDOW_PROVENANCE")
    assert data["resolved_source_raster"] and data["resolved_source_window"]
    assert any(r.get("equal") for r in data["full_window_verification"])
    # GT work must use this milestone's separately inspected frozen helpers.
    raise SystemExit("Exact provenance gate passes; continue authorized GT/alignment inspection before implementing its application.")


def finalize():
    data = read()
    assert data["exact_search_plan"] == PLAN and data.get("exact_search_plan_checkpoint")
    assert git("branch", "--show-current") == BRANCH and git("merge-base", START, "HEAD") == START
    for key, values in ENUMS.items():
        assert data[key] in values, (key, data[key])
    assert all(data[g] is True for g in GUARDS)
    if data["provenance_status"] in ("NOT_ESTABLISHED", "PARTIAL_EXACT_EVIDENCE"):
        assert data["resolved_source_raster"] is None and data["resolved_source_window"] is None
        assert not any(r.get("equal") for r in data["full_window_verification"])
        assert data["gt_status"] == "NOT_EVALUATED_NO_EXACT_PROVENANCE"
        assert data["case_validity"] == "NOT_EVALUABLE_WITHOUT_PROVENANCE"
        assert data["gt_building_instance_count"] is None and not data["gt_instances"]
        assert data["gt_reference"] is None and data["gt_target"] is None
        assert not (ROOT / "evaluation/task8b3_a2_ground_truth_case_validity_audit_v1_overlay.png").exists()
    else:
        assert data["gt_status"] == "GT_AVAILABLE_ALIGNED", "Exact provenance requires completed GT work before finalize"
    changed = set(git("diff", "--name-only", START).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    assert changed <= ALLOWED, changed - ALLOWED
    assert not git("diff", "--check", START)
    assert file_hash(A2) == LOCK and digest(rgb_locked().tobytes()) == data["locked_a2_decoded_rgb_sha256"]
    after = [inventory(EXTERNAL), inventory(WHU)]
    assert after == data["read_only_integrity_before"], "READ_ONLY_EXTERNAL_OR_WHU_INVENTORY_CHANGED"
    # Recheck every consulted source identity without running any historical script.
    for row in data["historical_sources_inspected"]:
        if row["commit"] is None:
            assert file_hash(row["source"]) == row["sha256"]
    for row in data["relation_semantics_source"] + data["inspected_mapping_sources"]:
        assert file_hash(row["path"]) == row["sha256"]
    data["read_only_integrity_final"] = after
    data["validation"].update(checks=[
        "Diagnostic syntax compile and module import PASS (-B; no bytecode writes).",
        "Saved JSON parse, required fields/enums/guard assertions PASS.",
        "Exact plan checkpoint predates search and all anchors remain frozen.",
        "GT phase refuses saved non-exact provenance (no evidence changes).",
        "Final allowed-path and whitespace diff audits PASS.",
        "Locked A2 file and decoded RGB SHA256 rechecks PASS.",
        "External RC1 and WHU path/size/mtime inventory unchanged; consulted identities unchanged.",
        "Repeated finalize is deterministic and performs zero inference."],
        changed_paths=sorted(changed), finalize_zero_model_inference=True,
        external_and_whu_inventory_unchanged=True)
    data["status"] = "READY_FOR_SUPERVISOR_AUDIT"
    completion = "Deterministic saved-evidence validation complete; STOP for Supervisor audit."
    if completion not in data["completed"]:
        data["completed"].append(completion)
    save(data)
    report(data)
    write_state(data, "STOP_FOR_SUPERVISOR_AUDIT", data["status"])
    print(json.dumps({k: data[k] for k in ("status", *ENUMS)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("provenance", "gt", "finalize"))
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    if args.phase == "provenance":
        search() if args.search else prepare()
    elif args.phase == "gt":
        gt()
    else:
        finalize()


if __name__ == "__main__":
    main()
