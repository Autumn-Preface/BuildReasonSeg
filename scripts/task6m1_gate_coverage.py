"""Task 6M.1 section 13/15: measure domain-gate coverage over the frozen v0.2 instruction templates.

For every distinct `(query_type, instruction_zh, instruction_en)` template in the frozen
BuildSpatialReason v0.2 records, records whether the gate accepts it and why not. This is the
regression guard for the documented gate vocabulary.

    python scripts/task6m1_gate_coverage.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, write_json  # noqa: E402

import predict_structured  # noqa: E402

OUT = EVAL / "task6m1_domain_gate_coverage.json"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"


def main() -> int:
    started = time.time()
    report = {
        "_doc": (
            "Task 6M.1 section 13. Coverage of the documented closed-Demo domain gate over every "
            "frozen BuildSpatialReason v0.2 instruction template (zh and en). Any falsely rejected "
            "in-domain template would make the Demo CLI unreachable for that program."
        ),
        "task": "6M.1",
        "object_anchors": list(predict_structured.DOMAIN_OBJECT_ANCHORS),
        "relation_anchors": list(predict_structured.DOMAIN_RELATION_ANCHORS),
        "per_split": {},
    }
    all_rejected = []
    for split in ("train", "val", "test"):
        records = []
        with (V02 / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        templates = defaultdict(set)
        for record in records:
            templates[record["query_type"]].add((record["instruction_zh"], record["instruction_en"]))
        accepted_zh = sum(
            1 for record in records
            if predict_structured.check_domain(str(record["instruction_zh"]))["supported"]
        )
        accepted_en = sum(
            1 for record in records
            if predict_structured.check_domain(str(record["instruction_en"]))["supported"]
        )
        rejected_templates = []
        for query_type, pairs in sorted(templates.items()):
            for zh, en in sorted(pairs):
                gate_zh = predict_structured.check_domain(zh)
                gate_en = predict_structured.check_domain(en)
                if not gate_zh["supported"] or not gate_en["supported"]:
                    rejected_templates.append(
                        {
                            "query_type": query_type,
                            "instruction_zh": zh,
                            "instruction_en": en,
                            "zh_supported": gate_zh["supported"],
                            "zh_reason": gate_zh["reason"],
                            "en_supported": gate_en["supported"],
                            "en_reason": gate_en["reason"],
                        }
                    )
        report["per_split"][split] = {
            "records": len(records),
            "distinct_templates": sum(len(pairs) for pairs in templates.values()),
            "programs": len(templates),
            "accepted_zh": accepted_zh,
            "accepted_zh_rate": round(accepted_zh / len(records), 6),
            "accepted_en": accepted_en,
            "accepted_en_rate": round(accepted_en / len(records), 6),
            "falsely_rejected_templates": rejected_templates,
        }
        all_rejected.extend(rejected_templates)

    rejected_programs = sorted({entry["query_type"] for entry in all_rejected})
    report["falsely_rejected_template_variants"] = len(all_rejected)
    report["falsely_rejected_programs"] = rejected_programs
    report["gate"] = {
        "all_zh_templates_accepted": all(not entry for entry in all_rejected if not entry["zh_supported"]),
        "all_en_templates_accepted": all(not entry for entry in all_rejected if not entry["en_supported"]),
        "every_program_reachable_in_both_languages": not all_rejected,
    }
    report["gate"]["passed"] = all(report["gate"].values())
    report["runtime_seconds"] = round(time.time() - started, 2)
    write_json(OUT, report)
    val = report["per_split"]["val"]
    print(
        f"[6m1.coverage] val zh {val['accepted_zh']}/{val['records']} "
        f"({val['accepted_zh_rate'] * 100:.1f}%) | en {val['accepted_en']}/{val['records']} "
        f"({val['accepted_en_rate'] * 100:.1f}%) | false rejections "
        f"{len(val['falsely_rejected_templates'])} -> gate "
        f"{'PASS' if report['gate']['passed'] else 'FAIL'}",
        flush=True,
    )
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
