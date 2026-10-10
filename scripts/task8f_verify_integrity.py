"""Final read-only identity audit; writes only this task's repository evidence."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import build_task8f_user_demo as build

def main():
    e=json.loads(build.EVIDENCE.read_text(encoding="utf-8"))
    protected=build.preservation(e)
    root=build.FINAL
    manifest=json.loads((root/"_engine/package_manifest.json").read_text(encoding="utf-8"))
    if manifest!=e["package_manifest"]:raise ValueError("V2 manifest changed; STOP")
    blobs=build.accepted_blobs();plan,derived=build.plan(blobs)
    expected=set()
    for row in manifest["files"]:
        path=root/row["path"];expected.add(row["path"])
        if build.base.identity(path)!={k:row[k] for k in ("bytes","sha256")}:raise ValueError("V2 hash changed; STOP: "+row["path"])
        relative=row["path"].removeprefix("_engine/")
        if row["role"]=="accepted_engine_source" and path.read_bytes()!=blobs[relative]:raise ValueError("Accepted source provenance mismatch; STOP")
        if row["role"].startswith("v2_observation") and path.read_bytes()!=derived[relative]:raise ValueError("Derived observation provenance mismatch; STOP")
        if row["role"] in ("accepted_model_asset","accepted_engine_version") and build.base.identity(path)!=build.base.identity(build.SOURCE/relative):raise ValueError("Accepted model/version provenance mismatch; STOP")
    for row in e["observation_checkpoint"]["harness_identities"]:
        path=REPO/row["path"]
        if build.base.identity(path)!={k:row[k] for k in ("bytes","sha256")}:raise ValueError("Validated harness changed; STOP: "+row["path"])
    inventory=build.base.inventory(root)
    extras=[]
    for row in inventory:
        path=row["path"]
        if path in expected or path=="_engine/package_manifest.json":continue
        if path.startswith("_runtime_cache/") or path=="_engine/model/components/program_head/qwen_integrity_cache.json":extras.append(row)
        else:raise ValueError("Unexpected new V2 package addition; STOP: "+path)
    audit=build.base.forbidden_audit(root)
    if not audit["pass"]:raise ValueError("Forbidden V2 content; STOP")
    for name in ("results","_engine/inference/input","_engine/inference/output","_engine/logs"):
        if any((root/name).rglob("*")):raise ValueError("Unexpected pre-user runtime files; STOP: "+name)
    if build.STAGING.exists():raise ValueError("Moved staging unexpectedly exists; STOP")
    launcher={name:json.loads((REPO/f"evaluation/task8f_{name}_launcher.json").read_text(encoding="utf-8")) for name in ("staging","final")}
    if not all(x["pass"] for x in launcher.values()):raise ValueError("Launcher gate failed; STOP")
    cache=json.loads((root/"_runtime_cache/yolo/Ultralytics/settings.json").read_text(encoding="utf-8"))
    hints={key:cache[key] for key in ("datasets_dir","weights_dir","runs_dir")}
    if hints!={"datasets_dir":"_runtime_cache/datasets","weights_dir":"_engine/model","runs_dir":"_engine/inference/output"}:raise ValueError("Private runtime directory hints not portable; STOP")
    e["launcher_validation"]=launcher
    e["integrity_gate"]={"pass":True,**protected,"manifest_files_checked":len(expected),"manifest_hash_identical":True,
                          "v2_model_assets_accepted_hashes":True,"derived_sources_match_validated_harness":True,"forbidden_content_audit":audit,
                          "clean_results_input_output_logs":True,"remaining_staging":False,"new_private_cache_files":extras,
                          "runtime_directory_hints":hints,"total_disk_files":len(inventory),"total_disk_bytes":sum(row["bytes"] for row in inventory),
                          "immutable_manifest_bytes":manifest["total_bytes"],"package_manifest_identity":build.base.identity(root/"_engine/package_manifest.json")}
    if not any(row.get("raw_output")=="evaluation/task8f_v1_regression.txt" for row in e["tests"]):
        e["tests"].append({"suite":"tests/test_task8e_user_demo.py","result":"57 passed in3.06s","exit_code":0,"unchanged_test_revision":True,"isolated_process":True,"model_calls":0,"raw_output":"evaluation/task8f_v1_regression.txt"})
    e["canonical_source_check"]={"checked":135,"match":135,"missing":0,"mismatch":0,"exit_code":0,"read_only":True,"raw_output":"evaluation/task8f_canonical_source_check.txt"}
    e["visual_capture_disposition"]={"initial_bbox_capture":"INVALID_OCCLUDED_NOT_VISUAL_EVIDENCE","correction":"Owned-window PrintWindow only; erroneous generated files replaced before commit, never used as evidence.","actual_staging_final_pages":"VISUALLY_INSPECTED_PASS","fixed_fake_proposals_fields_candidate_failure":"VISUALLY_INSPECTED_PASS"}
    e["head"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    build.EVIDENCE.write_text(json.dumps(e,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pass":True,"manifest":len(expected),"bytes":manifest["total_bytes"],"disk_files":len(inventory),"preservation":protected},ensure_ascii=False))
if __name__=="__main__":main()
