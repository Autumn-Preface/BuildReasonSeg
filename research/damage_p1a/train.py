"""Guarded standard U-Net training entry. Current dataset admission is BLOCKED.

This runner is engineering infrastructure, not a completed CUBIT experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
from .data import require_admission,validate_manifest,load_record,Tile,crop_pad,tiles,Reconstructor
from .metrics import Metrics


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False


class CropDataset:
    def __init__(self,rows,root,class_map,config):
        self.rows,self.root,self.class_map,self.config=rows,root,class_map,config
        self.epoch=0
    def __len__(self):return len(self.rows)*self.config['crops_per_source_per_epoch']
    def __getitem__(self,index):
        import torch
        row=self.rows[index//self.config['crops_per_source_per_epoch']]
        rgb,mask,valid=load_record(row,self.root,self.class_map)
        rng=np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index]))
        size=self.config['tile_size'];height,width=rgb.shape[:2]
        top=int(rng.integers(max(1,height-size+1)));left=int(rng.integers(max(1,width-size+1)))
        tile=Tile(top,left,min(size,height-top),min(size,width-left))
        arrays=[crop_pad(rgb.transpose(2,0,1),tile,size),crop_pad(mask,tile,size),crop_pad(valid,tile,size)]
        # Preserve geometry; never rescale native cracks or interpolate GT.
        if rng.integers(2):arrays=[a[...,::-1,:] for a in arrays]
        if rng.integers(2):arrays=[a[...,:,::-1] for a in arrays]
        rotation=int(rng.integers(4));arrays=[np.rot90(a,rotation,axes=(-2,-1)).copy() for a in arrays]
        tensors=[torch.from_numpy(a) for a in arrays]
        return tensors[0].float()/255,tensors[1].float(),tensors[2].bool()


def evaluate(model,rows,root,class_map,config,device,deadline=float('inf')):
    import torch
    metrics=Metrics();timings=[];model.eval()
    for row in rows:
        if time.monotonic()>=deadline:raise TimeoutError('Budget exhausted before full validation; no partial score')
        begin=time.perf_counter();rgb,gt,valid=load_record(row,root,class_map)
        input_and_gt_seconds=time.perf_counter()-begin
        inference_begin=time.perf_counter()
        h,w=rgb.shape[:2];recon=Reconstructor(h,w);network_seconds=0.0
        windows=tiles(h,w,config['tile_size'],config['validation_overlap'])
        for tile in windows:
            if time.monotonic()>=deadline:raise TimeoutError('Budget exhausted in validation; no partial score')
            array=crop_pad(rgb.transpose(2,0,1),tile,config['tile_size'])
            x=torch.from_numpy(array.copy()).float()[None].to(device)/255
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():p=model(x).sigmoid()
            if device.type=='cuda':torch.cuda.synchronize()
            network_seconds+=time.perf_counter()-t
            recon.add(tile,p[0].cpu().numpy())
        probability=recon.finish()
        inference_system_seconds=time.perf_counter()-inference_begin
        metrics.add(row['source_image_id'],probability,gt,valid,config['threshold'])
        timings.append({'image_id':row['source_image_id'],'native_hw':[h,w],'tiles':len(windows),
            'network_seconds_synchronized':network_seconds,'input_and_gt_load_check_seconds':input_and_gt_seconds,
            'inference_system_seconds_excluding_input_and_gt_loading':inference_system_seconds,
            'evaluation_end_to_end_seconds_including_gt_and_metrics':time.perf_counter()-begin})
    return metrics,timings


def validate_config(config):
    hours=config['max_gpu_hours'];epochs=config['epochs']
    if type(hours) not in (int,float) or not math.isfinite(hours) or not 0<hours<=12:
        raise ValueError('P1A budget must be finite and in (0,12] hours')
    if type(epochs) is not int or not 1<=epochs<=60:raise ValueError('P1A epochs must be integer 1..60')
    if config.get('test_inference') is not False:raise ValueError('P1A test inference forbidden')


def run(config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    gate=require_admission(config)  # BEFORE model construction, data access or output reservation.
    validate_config(config)
    rows=json.loads(Path(config['manifest']).read_text(encoding='utf-8'));validate_manifest(rows)
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='val']
    if not training or not validation:raise ValueError('train and val required')
    import torch
    from .model import UNet,loss_function
    seed_everything(config['seed'])
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable: stop, do not silently launch CPU training')
    device=torch.device('cuda');class_map={int(k):int(v) for k,v in gate['class_map'].items()}
    directory=Path(config['output_root'])/time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (directory/'provenance.json').write_text(json.dumps({'code_head':revision,'admission':gate,
        'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'precision':'float32','tf32':False,
        'budget_scope':'conservative wall time including data/validation','test_used':False},indent=2),encoding='utf-8')
    model=UNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    dataset=CropDataset(training,config['data_root'],class_map,config)
    loader=torch.utils.data.DataLoader(dataset,batch_size=config['microbatch'],shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(config['seed']))
    begin=time.monotonic();deadline=begin+config['max_gpu_hours']*3600;best=None;steps=0
    torch.cuda.reset_peak_memory_stats();history=[]
    try:
        for epoch in range(config['epochs']):
            dataset.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True);total=0.0
            for step,(x,y,v) in enumerate(loader):
                if time.monotonic()>=deadline:raise TimeoutError('Training budget exhausted')
                x,y,v=x.to(device),y.to(device),v.to(device)
                loss=loss_function(model(x),y,v)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                group_start=(step//config['accumulate'])*config['accumulate']
                divisor=min(config['accumulate'],len(loader)-group_start)
                (loss/divisor).backward();total+=float(loss.detach());steps+=1
                if (step+1)%config['accumulate']==0 or step+1==len(loader):optimizer.step();optimizer.zero_grad(set_to_none=True)
            metrics,timings=evaluate(model,validation,config['data_root'],class_map,config,device,deadline)
            summary=metrics.summary();score=summary['foreground_miou']
            if score is None:raise RuntimeError('both validation classes must have supported GT')
            record={'epoch':epoch+1,'steps':steps,'train_loss':total/len(loader),'validation':summary,
                    'wall_seconds':time.monotonic()-begin}
            history.append(record)
            with (directory/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'steps':steps,
                'config':config,'random':random.getstate(),'numpy_random':np.random.get_state(),
                'torch_random':torch.get_rng_state(),'cuda_random':torch.cuda.get_rng_state_all()}
            torch.save(state,directory/'last.pt')
            if best is None or score>best:
                best=score;torch.save({'model':model.state_dict(),'epoch':epoch+1,'config':config},directory/'best.pt')
                with (directory/'best_val_per_image.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(metrics.rows[0]));writer.writeheader();writer.writerows(metrics.rows)
                (directory/'best_val_timing.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
        status='COMPLETED_CONFIGURED_EPOCHS'
    except TimeoutError as error:
        status='BUDGET_EXHAUSTED';(directory/'budget_stop.txt').write_text(str(error),encoding='utf-8')
    finally:
        torch.cuda.synchronize()
        (directory/'run_status.json').write_text(json.dumps({'status':locals().get('status','FAILED'),
            'fully_validated_epochs':len(history),'steps':steps,'best_validation_miou':best,
            'wall_seconds':time.monotonic()-begin,'training_context_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'training_context_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'test_inference':False},indent=2),encoding='utf-8')
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/damage_p1a/unet_baseline.json')
    args=parser.parse_args()
    try:print(run(args.config))
    except RuntimeError as error:parser.exit(2,str(error)+'\n')
