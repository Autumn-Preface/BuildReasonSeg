"""Offline-only Task8D contracts. No model loading/execution or external writes."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import task8d_above_target_chain_diagnosis as d


@pytest.fixture
def fake_lock(tmp_path):
    run=tmp_path/"1008";run.mkdir()
    artifact=run/"maps.npz";artifact.write_bytes(b"fake artifact")
    runtime=tmp_path/"runtime.json";runtime.write_bytes(b"fake runtime")
    image=tmp_path/"1008.tif";image.write_bytes(b"fake image")
    transcript=tmp_path/"above_stdout.txt";transcript.write_bytes(b"fake stdout")
    truth=tmp_path/"truth.json";truth.write_bytes(b"fake truth")
    row={"relation":"above","sample_id":d.SAMPLE,"expected_program":d.PROGRAM,
         "runtime_status":"SUCCESS","reference_mode":"automatic","reference_id":5,
         "mask_area":1580,"image":str(image),"image_sha256":d.identity(image)["sha256"],
         "run_root":str(run),"artifacts":[d.identity(artifact)],"transcript_identity":d.identity(transcript),
         "gt_audit":{"canonical_reference_id":4,"canonical_target_id":6,
             "selected_reference_best_gt_instance_id":4,
             "selected_reference_iou_with_canonical_gt_reference":0.9097432024169184,
             "predicted_target_best_gt_instance_id":7,"predicted_target_best_gt_iou":0.48158096699923253,
             "target_iou":0.07762201453790239,"target_dice":0.14406167188629246,
             "semantic_chain_status":"TARGET_IDENTITY_MISMATCH","truth_inputs":[d.identity(truth)]}}
    # Synthetic runtime SHA is patched only in the fake unit fixture, never production.
    return {"cases":[row],"formal_runner_invocation_count":1,"real_candidates_attempted":4,
            "no_inference_retry":True,"frozen_runtime_identity":d.identity(runtime)}


def test_exact_sample_lock(fake_lock,monkeypatch):
    monkeypatch.setattr(d,"RUNTIME_SHA",fake_lock["frozen_runtime_identity"]["sha256"])
    assert d.lock_above(fake_lock)["sample_id"]==d.SAMPLE


@pytest.mark.parametrize("field,value",[("sample_id","other"),("expected_program","largest_to_below_to_nearest"),
    ("reference_mode","assisted"),("reference_id",4),("mask_area",1),("runtime_status","FAILED")])
def test_sample_mutation_rejected(fake_lock,monkeypatch,field,value):
    monkeypatch.setattr(d,"RUNTIME_SHA",fake_lock["frozen_runtime_identity"]["sha256"])
    fake_lock["cases"][0][field]=value
    with pytest.raises(ValueError):d.lock_above(fake_lock)


def test_frozen_artifact_hash_mismatch(fake_lock,monkeypatch):
    monkeypatch.setattr(d,"RUNTIME_SHA",fake_lock["frozen_runtime_identity"]["sha256"])
    Path(fake_lock["cases"][0]["artifacts"][0]["path"]).write_bytes(b"modified")
    with pytest.raises(ValueError,match="identity changed"):d.lock_above(fake_lock)


def test_extra_run_file_rejected(fake_lock,monkeypatch):
    monkeypatch.setattr(d,"RUNTIME_SHA",fake_lock["frozen_runtime_identity"]["sha256"])
    (Path(fake_lock["cases"][0]["run_root"])/"extra.txt").write_text("extra")
    with pytest.raises(ValueError,match="file set"):d.lock_above(fake_lock)


def test_tiff_hash_mismatch(fake_lock,monkeypatch):
    monkeypatch.setattr(d,"RUNTIME_SHA",fake_lock["frozen_runtime_identity"]["sha256"])
    Path(fake_lock["cases"][0]["image"]).write_bytes(b"modified")
    with pytest.raises(ValueError,match="TIFF"):d.lock_above(fake_lock)


def test_context_roundtrip_padding_is_zero():
    from scripts.task8c_final_demo_evaluate import context_to_global
    mask=np.zeros((512,512),dtype=bool);mask[103:149,142:256]=True
    ctx={"origin":[-197,-41],"size":512}
    mapped=d.to_context(mask,ctx)
    assert np.array_equal(mapped[144:190,339:453],mask[103:149,142:256])
    assert not mapped[:41].any() and not mapped[:,:197].any()
    assert np.array_equal(context_to_global(mapped,ctx,(512,512)),mask)


def test_fractional_area_mapping_deterministic_not_thresholded():
    mask=np.zeros((512,512),dtype=bool);mask[0,0]=True;mask[8:16,8:16]=True
    a=d.fractional_occupancy(mask);b=d.fractional_occupancy(mask)
    assert np.array_equal(a,b)
    assert a[0,0]==1/64 and a[1,1]==1 and a.sum()==65/64
    assert a.dtype==np.float64


def test_mass_mean_and_fractional_attention():
    x=np.array([[1.,3.],[5.,7.]])
    m=np.array([[0.25,1.],[0.,0.5]])
    mass,mean=d.mass_mean(x,m)
    assert mass==6.75 and mean==6.75/1.75
    assert d.mass_mean(x,np.zeros((2,2)))==(0.,None)
    with pytest.raises(ValueError):d.mass_mean(x,np.ones((4,4)))


def fake_rows():
    return [{"instance_id":i,"inside_context":True,**{v:float(i) for v in d.STAGE_KEYS.values()}}
            for i in (7,3,6,4)]


def test_target6_wrong7_delta_no_tolerance():
    rows=fake_rows();pair=d.pair_comparison(rows)
    assert all(r["delta_6_minus_7"]==-1 and r["which_instance_is_favoured"]==7 for r in pair)
    for r in rows:r["W_mass"]=1 if r["instance_id"]!=6 else 1+1e-12
    w=d.pair_comparison(rows,("W",))[0]
    assert w["which_instance_is_favoured"]==6


def test_rank_deterministic_tie_and_all_candidates():
    rows=fake_rows()
    for r in rows:r["W_mass"]=1
    one=d.rank_trajectory(rows);two=d.rank_trajectory(rows[::-1])
    assert one==two and one["W"]["ordered_instance_ids"]==[3,4,6,7]
    assert one["W"]["target6_rank"]==3 and one["W"]["wrong7_rank"]==4


def test_empty_context_means_not_ranked():
    rows=fake_rows();rows[0]["inside_context"]=False
    assert 7 not in d.rank_trajectory(rows)["W"]["ordered_instance_ids"]


def test_first_divergence_order_is_fixed_independent_of_input_order():
    rows=fake_rows()
    for r in rows:r["W_mass"]=-r["instance_id"];r["A_attention_mass"]=-r["instance_id"]
    pair=d.pair_comparison(rows)[::-1];rank=d.rank_trajectory(rows)
    assert d.STAGES==("W","A","C","logits","probability","final_mask")
    assert d.first_divergence(pair,rank)["stage"]=="C"


def test_first_divergence_actual_top_is_not_forced_to_6_or_7():
    rows=fake_rows();next(r for r in rows if r["instance_id"]==3)["W_mass"]=100
    result=d.first_divergence(d.pair_comparison(rows),d.rank_trajectory(rows))
    assert result["stage"]=="W" and result["actual_top_instance_id"]==3
    assert result["neither6_nor7_is_top"] is True


@pytest.mark.parametrize("name",["ultralytics","transformers","sam2","buildreasonseg.runtime.detector",
    "buildreasonseg.runtime.core","buildreasonseg.runtime.pipeline"])
def test_no_inference_imports(name):
    with d.OfflineGuard():
        with pytest.raises(RuntimeError,match="forbidden import"):builtins.__import__(name)


@pytest.mark.parametrize("operation",["Popen","run","call","check_call","check_output"])
def test_no_subprocess_model_execution(operation):
    with d.OfflineGuard():
        with pytest.raises(RuntimeError):getattr(subprocess,operation)(["predict.py"])


def test_no_torch_checkpoint_or_neural_model_construction():
    import torch
    with d.OfflineGuard():
        with pytest.raises(RuntimeError):torch.load("decoder.pt")
        with pytest.raises(RuntimeError):torch.nn.Linear(1,1)


def test_no_external_writes_with_both_open_apis():
    path=d.EXTERNAL/"task8d-unauthorized-test.txt"
    with d.OfflineGuard():
        for opener in (builtins.open,io.open):
            with pytest.raises(PermissionError):opener(path,"wb")
        with pytest.raises(PermissionError):Path(path).write_text("forbidden")
    assert not path.exists()


def test_task8c_files_read_only(tmp_path):
    p=tmp_path/"task8c_final_demo_v1.json";p.write_text("{}")
    with d.OfflineGuard() as guard:
        assert p.read_text()=="{}"
        with pytest.raises(PermissionError):p.open("r+")
        assert {r["mode"] for r in guard.read_modes if r["path"]==str(p.resolve())}=={"r"}


def test_output_allowlist_is_exact():
    for n in d.FIGURES:assert d.require_output(d.FIGURE_ROOT/n)==(d.FIGURE_ROOT/n).resolve()
    for p in [d.REPO/"evaluation/task8c_final_demo_v1.json",d.CANONICAL/"buildreasonseg/runtime/core.py",d.FIGURE_ROOT/"extra.npz",d.FIGURE_ROOT/"../other.png"]:
        with pytest.raises(PermissionError):d.require_output(p)


def test_real_deterministic_counterfactual_only_fields():
    ref=np.zeros((512,512),dtype=bool);ref[320:400,220:290]=True
    with d.OfflineGuard():
        one=d.canonical_reference_fields(ref);two=d.canonical_reference_fields(ref)
    assert set(one)=={"P_dir","P_near","W","A"}
    assert all(one[k].shape==(64,64) and np.isfinite(one[k]).all() and np.array_equal(one[k],two[k]) for k in one)
    assert np.allclose(one["W"],np.clip(one["P_dir"]*one["P_near"],0,1))
    assert np.isclose(one["A"].sum(),1)


@pytest.mark.parametrize("auto,gt,expected,restored",[
    (6,6,"MIXED / NOT RESOLVED",False),
    (7,6,"PLAUSIBLE_CONTRIBUTOR",True),
    (7,7,"NOT_SOLE_EXPLANATION",False),
    (6,"TIE","MIXED / NOT RESOLVED",False)])
def test_counterfactual_requires_actual_preference_restoration(auto,gt,expected,restored):
    a=[{"stage":k,"which_instance_is_favoured":auto} for k in ("W","A")]
    g=[{"stage":k,"which_instance_is_favoured":gt} for k in ("W","A")]
    outcome=d.counterfactual_interpretation(a,g)
    assert outcome["interpretation"]==expected and outcome["preference_restored"] is restored


def test_aggregation_uses_context_mask_not_global_mask():
    masks={6:np.zeros((512,512),dtype=bool)};masks[6][100:108,100:108]=True
    ctx={"origin":[-8,-8],"size":512}
    maps={k:np.zeros(shape,dtype=np.float32) for k,shape in d.MAP_SHAPES.items()}
    maps["W"][13,13]=1;maps["A"][13,13]=0.5;maps["logits"][108:116,108:116]=2
    rows,_=d.stage_rows(maps,masks,ctx,masks[6],{6:{}})
    assert rows[0]["W_mass"]==0.25 and rows[0]["A_attention_mass"]==0.125
    assert rows[0]["logit_mean"]==2 and rows[0]["final_mask_iou"]==1


def test_static_call_graph_no_product_runtime_or_checkpoint():
    tree=ast.parse(Path(d.__file__).read_text(encoding="utf-8"))
    forbidden={"DetectorRuntime","Sam2Runtime","Db1Runtime","PredictRuntime","run_core_chain","predict_one","target_prototype","prototype_similarity","GlobalCompetitionDecoder"}
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else "" for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not (calls & forbidden)
    assert "subprocess.run" not in Path(d.__file__).read_text(encoding="utf-8")


def test_counterfactual_schema_and_report_json_consistency():
    path=d.JSON_PATH
    evidence=json.loads(path.read_text(encoding="utf-8"))
    assert evidence["task_id"]==d.TASK_ID and evidence["task8c_head"]==d.ACCEPTED_HEAD
    assert evidence["predeclared_statistics"]["first_divergence_order"]==list(d.STAGES)
    if evidence["all_gt_instance_stage_table"]:
        cf=evidence["canonical_reference_field_counterfactual"]
        assert cf["only_deterministic_fields"] and not cf["C_recomputed"] and not cf["q_recomputed"] and not cf["decoder_called"]
        assert evidence["target6_vs_wrong7"]==d.pair_comparison(evidence["all_gt_instance_stage_table"])
        assert evidence["rank_trajectory"]==d.rank_trajectory(evidence["all_gt_instance_stage_table"])
        assert evidence["first_observed_target_divergence_stage"]==d.first_divergence(evidence["target6_vs_wrong7"],evidence["rank_trajectory"])
        assert evidence["conclusions"]==d.diagnosis_conclusions(evidence)
        assert set(evidence["db1_mechanism_source_citations"])==set(evidence["db1_mechanism_audit"])
        assert all(evidence["db1_mechanism_source_citations"].values())
        assert d.REPORT_PATH.read_text(encoding="utf-8")==d.render_report(evidence)
        assert all(evidence[k]==0 for k in ("model_calls","detector_calls","predict_calls","checkpoint_loads","external_writes"))
