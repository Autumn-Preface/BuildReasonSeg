"""Task 6M.1 section 14: verify the deterministic domain gate against the required prompt sets.

Runs the gate directly (no models) AND through the CLI as a subprocess, asserting:

* the 6 required out-of-domain prompts are rejected with exit code 4, status
  `unsupported_instruction` and reason `out_of_domain_prompt`;
* the 5 required positive prompts are accepted by the gate (they then enter the ProgramHead path);
* rejection happens before any model import/instantiation.

    python scripts/task6m1_domain_gate_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, EXPORT_ROOT, write_json  # noqa: E402

import predict_structured  # noqa: E402

OUT = EVAL / "task6m1_domain_gate_check.json"

OOD_PROMPTS = (
    "Write a poem about the sea.",
    "今天天气怎么样？",
    "请总结这张图片。",
    "检测道路。",
    "segment the airplane",
    "   ",
)
POSITIVE_PROMPTS = (
    "分割面积最大的建筑物。",
    "找出最左侧的建筑区域。",
    "分割面积最大的建筑物右侧最近的建筑物。",
    "segment the building nearest to the right of the largest building",
    "找出最小建筑物上方的建筑。",
)


def main() -> int:
    started = time.time()
    gate_rows = []
    for prompt in OOD_PROMPTS:
        result = predict_structured.check_domain(prompt)
        gate_rows.append({"prompt": prompt, "expected": "rejected", "supported": result["supported"],
                          "reason": result["reason"],
                          "object_anchor": result["object_anchor"],
                          "relation_anchor": result["relation_anchor"]})
    for prompt in POSITIVE_PROMPTS:
        result = predict_structured.check_domain(prompt)
        gate_rows.append({"prompt": prompt, "expected": "accepted", "supported": result["supported"],
                          "reason": result["reason"],
                          "object_anchor": result["object_anchor"],
                          "relation_anchor": result["relation_anchor"]})

    ood_ok = all(row["supported"] is False for row in gate_rows if row["expected"] == "rejected")
    positive_ok = all(row["supported"] is True for row in gate_rows if row["expected"] == "accepted")

    # ---- CLI-level check for one OOD prompt: exit 4 and a written result.json
    probe_dir = REPO_ROOT / "artifacts" / "task6m1_domain_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    image = EXPORT_ROOT / "images" / "val" / f"{sorted(p.stem for p in (EXPORT_ROOT / 'images' / 'val').glob('*.tif'))[0]}.tif"
    command = [
        sys.executable, str(REPO_ROOT / "predict_structured.py"),
        "--image", str(image),
        "--prompt", OOD_PROMPTS[0],
        "--proposal-checkpoint", "artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt",
        "--parser-checkpoint", "artifacts/checkpoints/task6m/program_parser_v02_best.pt",
        "--out-dir", str(probe_dir),
        "--quiet",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    probe_payload = {}
    probe_result = probe_dir / "result.json"
    if probe_result.is_file():
        probe_payload = json.loads(probe_result.read_text(encoding="utf-8"))

    report = {
        "_doc": (
            "Task 6M.1 section 14. Deterministic domain gate verification: the closed-Demo grammar "
            "guard rejects out-of-domain prompts before ProgramHead and before the proposal model."
        ),
        "task": "6M.1",
        "object_anchors": list(predict_structured.DOMAIN_OBJECT_ANCHORS),
        "relation_anchors": list(predict_structured.DOMAIN_RELATION_ANCHORS),
        "gate_rows": gate_rows,
        "out_of_domain_all_rejected": bool(ood_ok),
        "positive_prompts_all_accepted": bool(positive_ok),
        "cli_probe": {
            "prompt": OOD_PROMPTS[0],
            "exit_code": result.returncode,
            "expected_exit_code": predict_structured.UNSUPPORTED_EXIT_CODE,
            "status": probe_payload.get("status"),
            "abstention_reason": probe_payload.get("abstention_reason"),
            "checked_before_parser": (probe_payload.get("domain_gate") or {}).get("checked_before_parser"),
            "checked_before_proposal_model": (probe_payload.get("domain_gate") or {}).get(
                "checked_before_proposal_model"
            ),
            "parsed_program": probe_payload.get("parsed_program"),
            "proposal_count": probe_payload.get("proposal_count"),
        },
        "gate": {
            "ood_rejected_with_exit_4": bool(
                result.returncode == predict_structured.UNSUPPORTED_EXIT_CODE
                and probe_payload.get("status") == predict_structured.UNSUPPORTED_STATUS
                and probe_payload.get("abstention_reason") == predict_structured.UNSUPPORTED_REASON
            ),
            "no_program_mapped_for_ood": probe_payload.get("parsed_program") is None,
            "no_proposals_run_for_ood": probe_payload.get("proposal_count") is None,
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    report["gate"]["passed"] = (
        report["out_of_domain_all_rejected"]
        and report["positive_prompts_all_accepted"]
        and all(report["gate"].values())
    )
    write_json(OUT, report)
    print(
        f"[6m1.gate] OOD rejected {ood_ok}; positives accepted {positive_ok}; CLI exit "
        f"{result.returncode} status {probe_payload.get('status')} -> gate "
        f"{'PASS' if report['gate']['passed'] else 'FAIL'}",
        flush=True,
    )
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
