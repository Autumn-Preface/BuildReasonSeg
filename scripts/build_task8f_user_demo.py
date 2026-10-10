"""V2-only repeatable staging builder. Existing RC1/V1/research evidence is read-only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import build_task8e_user_demo as base
spec = importlib.util.spec_from_file_location("task8f_observation_patch", REPO / "demo/user_demo_v2/engine_observation_patch.py")
patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
ACCEPTED_HEAD = "93f66bc9a4fcb7b7ce5699f1c1afe412d18c43d4"
SOURCE = base.SOURCE
V1 = SOURCE.with_name("BuildReasonSeg_Demo_V1")
FINAL = SOURCE.with_name("BuildReasonSeg_Demo_V2")
STAGING = SOURCE.with_name("BuildReasonSeg_Demo_V2_task8f_staging_v1")
APP_FILES = base.APP_FILES + ("_ui/trace.py", "_ui/render.py", "_ui/trace_ui.py")
EVIDENCE = REPO / "evaluation/task8f_user_demo_live_trace_v1.json"

def accepted_blobs():
    def read(relative):
        return subprocess.run(["git","-C",str(REPO),"show",ACCEPTED_HEAD+":"+base.CANONICAL+relative],capture_output=True,check=True).stdout
    manifest = json.loads(read("source_manifest.json"))
    blobs = {}
    for item in manifest["files"]:
        data = read(item["path"])
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Accepted source manifest anomaly; STOP")
        blobs[item["path"]] = data
    return blobs

def plan(blobs):
    changed, records = patch.patched_sources(blobs)
    closure, analysis = base.dependency_closure(blobs)
    sources = sorted(set(closure) | {"configs/inference.yaml","model/buildreasonseg_advisor/model.yaml","model/buildreasonseg_advisor/metadata.json"})
    if not set(patch.FILES).issubset(sources):
        raise ValueError("Observation source is outside original runtime closure; STOP")
    assets = list(base.ASSETS) + [base.QWEN+name for name in base.QWEN_FILES] + ["VERSION"]
    return {"accepted_head":ACCEPTED_HEAD,"source_paths":sources,"assets":assets,"app_files":list(APP_FILES),
            "static_dependency_analysis":analysis,"observation_source_differences":records,
            "expected_manifest_entries":len(sources)+len(assets)+len(APP_FILES)+2}, changed

def preservation(evidence):
    baseline = evidence["preflight"]
    base.verify_inventory(SOURCE,baseline["old_rc1_inventory"])
    base.verify_inventory(V1,baseline["v1_inventory"])
    for row in baseline["locked_repository_identities"]:
        target = REPO / row["path"]
        if base.identity(target) != {k:row[k] for k in ("bytes","sha256")}:
            raise ValueError("Locked canonical/governance/Task8C/D/E file changed; STOP: "+row["path"])
    return {"pass":True,"old_rc1_files":len(baseline["old_rc1_inventory"]),"old_v1_files":len(baseline["v1_inventory"]),
            "locked_repository_files":len(baseline["locked_repository_identities"]),"old_rc1_v1_all_bytes_sha256_mtime_file_set_unchanged":True}

def assemble(blobs, build_plan, changed, evidence, source=SOURCE, staging=STAGING, final=FINAL, app=REPO/"demo/user_demo_v2"):
    base.safe_destination(source,staging,final)
    before = preservation(evidence)
    expected = {r["path"]:r for r in evidence["preflight"]["old_rc1_inventory"]}
    selected = build_plan["source_paths"]+build_plan["assets"]
    for relative in selected:
        base.reject_link(source/relative)
        if relative not in expected or base.identity(source/relative) != {k:expected[relative][k] for k in ("bytes","sha256")}:
            raise ValueError("Selected accepted asset drift; STOP: "+relative)
        if relative in blobs and (source/relative).read_bytes() != blobs[relative]:
            raise ValueError("Accepted Git source differs; STOP: "+relative)
    if shutil.disk_usage(staging.parent).free < sum(expected[p]["bytes"] for p in selected)+(128<<20):
        raise OSError("Insufficient disk space; STOP")
    staging.mkdir()
    records=[]
    differences = {r["path"]:r for r in build_plan["observation_source_differences"]}
    def write_bytes(relative,data,role,extra=None):
        target = staging / relative; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as handle: handle.write(data)
        records.append({"path":relative,**base.identity(target),"role":role,**(extra or {})})
    def copy(origin,relative,role):
        target=staging/relative; target.parent.mkdir(parents=True,exist_ok=True)
        with origin.open("rb") as reader,target.open("xb") as writer: shutil.copyfileobj(reader,writer,1<<20)
        if base.identity(origin) != base.identity(target): raise ValueError("Copy hash mismatch; STOP: "+relative)
        records.append({"path":relative,**base.identity(target),"role":role})
    for relative in build_plan["source_paths"]:
        if relative in changed:
            row=differences[relative]
            write_bytes("_engine/"+relative,changed[relative],"v2_observation_only_derived_source",{"base_sha256":row["base_sha256"],"base_head":ACCEPTED_HEAD,"scientific_ast_audit":row["scientific_ast_audit"]})
        else: copy(source/relative,"_engine/"+relative,"accepted_engine_source")
    helper="buildreasonseg/runtime/observation.py"
    write_bytes("_engine/"+helper,changed[helper],"v2_observation_only_transport_no_scientific_algorithm")
    for relative in build_plan["assets"]:
        copy(source/relative,"_engine/"+relative,"accepted_engine_version" if relative=="VERSION" else "accepted_model_asset")
        print("Verified asset "+relative,flush=True)
    for relative in APP_FILES: copy(app/relative,relative,"user_demo_v2_source")
    write_bytes("_ui/runtime.json",(json.dumps({"validated_python":sys.executable,"runtime_environment_bundled":False},ensure_ascii=False,indent=2)+"\n").encode(),"machine_python_locator_only")
    for name in ("results","_engine/inference/input","_engine/inference/output","_engine/logs","_runtime_cache/yolo","_runtime_cache/matplotlib"):
        (staging/name).mkdir(parents=True,exist_ok=True)
    manifest={"demo_version":"V2","accepted_engine_head":ACCEPTED_HEAD,"files":sorted(records,key=lambda r:r["path"]),"file_count":len(records),
              "total_bytes":sum(r["bytes"] for r in records),"runtime_environment_bundled":False,"source_model_portability":"PATH_PORTABLE_ON_CONFIGURED_MACHINE",
              "source_provenance":"Three explicit V2 observation-only derivatives plus one new transport; all other engine source and model assets accepted byte-identical.",
              "observation_source_differences":build_plan["observation_source_differences"],"scientific_contract_unchanged":True}
    if len(records)!=build_plan["expected_manifest_entries"]: raise ValueError("Unexpected package entry count; STOP")
    (staging/"_engine/package_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    audit=base.forbidden_audit(staging)
    if not audit["pass"]: raise ValueError("Forbidden content; STOP: "+str(audit))
    after=preservation(evidence)
    return {"manifest":manifest,"staging":str(staging),"final":str(final),"move_pending":True,"preservation_before":before,"preservation_after":after,"forbidden_content_audit":audit,
            "clean_results_output_logs":all(not any((staging/name).iterdir()) for name in ("results","_engine/inference/output","_engine/logs"))}

def main():
    parser=argparse.ArgumentParser()
    operations=parser.add_mutually_exclusive_group()
    operations.add_argument("--assemble",action="store_true"); operations.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    evidence=json.loads(EVIDENCE.read_text(encoding="utf-8"))
    blobs=accepted_blobs(); build_plan,changed=plan(blobs)
    evidence["assembly_plan"]=build_plan; evidence["observation_source_differences"]=build_plan["observation_source_differences"]
    if args.assemble:
        evidence["assembly"]=assemble(blobs,build_plan,changed,evidence)
        evidence["package_manifest"]=evidence["assembly"]["manifest"]
    if args.finalize:
        preservation(evidence)
        if not evidence.get("staging_validation",{}).get("pass"):
            raise ValueError("Staging validation required before move; STOP")
        evidence["folder_move"]=base.finalize(SOURCE,STAGING,FINAL,evidence["package_manifest"])
        evidence["assembly"]["move_pending"]=False
        evidence["preservation_after_move"]=preservation(evidence)
    EVIDENCE.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"source_files":len(build_plan["source_paths"]),"expected_entries":build_plan["expected_manifest_entries"],"static_audit":"PASS","assembled":args.assemble,"finalized":args.finalize}))
if __name__=="__main__": main()
