"""Synthetic-only DAMAGE-P1A engineering contracts. No real CUBIT training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from research.damage_p1a.data import (parse_yolo,rasterize,starts,tiles,Tile,crop_pad,Reconstructor,
    require_admission,validate_manifest)
from research.damage_p1a.metrics import Metrics
from research.damage_p1a.model import UNet,loss_function


def test_authoritative_mapping_is_required_and_not_inferred():
    with pytest.raises(ValueError,match='unmapped'):parse_yolo('0 0 0 1 0 1 1',{})
    # Arbitrary numeric IDs prove filenames/class order aren't being used by the converter.
    p=parse_yolo('7 0.125 0.125 0.375 0.125 0.375 0.375 0.125 0.375',{7:1})
    target,valid=rasterize(p,8,8,[False,True])
    expected=np.zeros((8,8),bool);expected[1:3,1:3]=True
    np.testing.assert_array_equal(target[1],expected)
    assert target[1].sum()==4 and not target[0].any()
    assert valid[1].all() and not valid[0].any()


@pytest.mark.parametrize('line',['0 0 0 1 0','0 0 0 1 0 1 nan','0 0 0 1 0 1 1.01','0 -0.1 0 1 0 1 1'])
def test_invalid_polygons_fail(line):
    with pytest.raises(ValueError):parse_yolo(line,{0:0})


def test_overlap_union_unknown_and_explicit_ignore_are_preserved():
    square=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    target,valid=rasterize([(0,square),(0,square),(1,square)],4,4,[True,True],[(0,square)])
    assert target.all() and not valid[0].any() and valid[1].all()
    with pytest.raises(ValueError,match='unknown'):rasterize([(1,square)],4,4,[True,False])


@pytest.mark.parametrize('h,w,size,overlap',[(5,7,4,2),(3,2,8,2),(15,17,8,3),(512,512,512,128),(517,725,512,128)])
def test_native_probability_reconstruction_and_padding(h,w,size,overlap):
    expected=np.random.default_rng(4).random((2,h,w),dtype=np.float32)
    reconstruction=Reconstructor(h,w)
    for tile in tiles(h,w,size,overlap):
        p=crop_pad(expected,tile,size,fill=0.99)
        reconstruction.add(tile,p)
    np.testing.assert_allclose(reconstruction.finish(),expected,rtol=1e-6,atol=1e-7)
    assert starts(h,size,overlap)[-1]==max(0,h-size)


def test_uncovered_and_invalid_probability_reconstruction_rejected():
    recon=Reconstructor(6,6);recon.add(Tile(0,0,2,2),np.ones((2,2,2),np.float32))
    with pytest.raises(ValueError,match='uncovered'):recon.finish()
    with pytest.raises(ValueError,match='probability'):recon.add(Tile(0,0,2,2),np.full((2,2,2),np.nan))


def test_confusions_known_hand_count_and_native_aggregation():
    m=Metrics();g=np.zeros((2,2,2),bool);g[0,0,:]=True;g[1,1,0]=True
    p=np.zeros((2,2,2),np.float32);p[0,:,0]=1;p[1]=g[1]
    row=m.add('synthetic',p,g,np.ones_like(g));s=m.summary()
    assert (row[0]['tp'],row[0]['fp'],row[0]['fn'])==(1,1,1)
    assert row[0]['iou']==pytest.approx(1/3) and row[0]['dice']==.5
    assert row[0]['precision']==row[0]['recall']==.5
    assert s['foreground_miou']==pytest.approx(2/3)
    # A small perfect crop must not get the same weight as a large failed native image.
    g2=np.ones((2,10,10),bool);m.add('larger',np.zeros_like(g2,float),g2,np.ones_like(g2))
    assert m.summary()['classes'][0]['iou']==pytest.approx(1/103)


def test_empty_negative_false_positive_and_unknown_channel():
    m=Metrics();g=np.zeros((2,2,2),bool);v=np.ones_like(g);v[1]=False
    m.add('correct-empty',np.zeros_like(g,float),g,v)
    p=np.zeros_like(g,float);p[0,0,0]=1;p[1]=1
    m.add('false-positive',p,g,v)
    s=m.summary();c=s['classes'][0]
    assert c['negative_images']==2 and c['negative_fp_image_rate']==.5
    assert c['negative_mean_fp_area_fraction']==.125
    assert c['macro_image_count']==1 and c['macro_image_iou']==0
    assert s['classes'][1]['known_images']==0 and s['foreground_miou'] is None


@pytest.mark.parametrize('key',['source_image_id','image_sha256','building_id','sequence_id'])
def test_split_leakage_fails_before_cropping(key):
    rows=[{'source_image_id':str(i),'image_sha256':str(i),'known_channels':[True,True],'split':s} for i,s in enumerate(['train','val'])]
    for row in rows:row[key]='shared'
    with pytest.raises(ValueError,match='leakage'):validate_manifest(rows)


def test_padding_validity_and_mask_geometry():
    valid=np.ones((2,3,5),bool)
    padded=crop_pad(valid,Tile(0,0,3,5),8)
    assert padded[:,:3,:5].all() and not padded[:,3:,:].any() and not padded[:,:,:][:,:,5:].any()


def test_loss_ignore_gradient_and_no_labelled_pixels():
    logits=torch.zeros((1,2,4,4),requires_grad=True);target=torch.zeros_like(logits);valid=torch.zeros_like(logits,dtype=torch.bool);valid[:,0]=True
    loss=loss_function(logits,target,valid);loss.backward()
    assert torch.isfinite(loss) and logits.grad[:,0].abs().sum()>0 and logits.grad[:,1].abs().sum()==0
    with pytest.raises(ValueError,match='no supervised'):loss_function(logits,target,torch.zeros_like(valid))


def test_plain_unet_dimensions_and_parameter_count():
    model=UNet()
    with torch.inference_mode():out=model(torch.zeros(1,3,32,48))
    assert out.shape==(1,2,32,48)
    assert 30_000_000<sum(p.numel() for p in model.parameters())<32_000_000
    with pytest.raises(ValueError,match='multiples'):model(torch.zeros(1,3,31,32))


def test_current_training_entry_fails_before_model_or_manifest_access(monkeypatch):
    from research.damage_p1a import model,train
    calls=[]
    monkeypatch.setattr(model,'UNet',lambda:calls.append('UNEXPECTED_MODEL_CONSTRUCTION'))
    with pytest.raises(RuntimeError,match='DATA_ADMISSION_BLOCKED'):train.run('configs/damage_p1a/unet_baseline.json')
    assert not calls
