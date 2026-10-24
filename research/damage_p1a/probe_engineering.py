"""Real GPU computations on explicitly synthetic tensors; not a damage experiment."""
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def main():
    from .train import seed_everything
    seed_everything(20261025)
    import torch
    from .model import UNet,loss_function
    root=Path(__file__).resolve().parents[2];out=root/'reports/damage_p1a/20261025'
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    info={'scope':'SYNTHETIC_ONLY_NO_CUBIT_DATA_NO_CHECKPOINT','python':sys.executable,'platform':platform.platform(),
        'versions':{n:metadata.version(n) for n in ['torch','numpy','Pillow','scipy','pytest','pycocotools']},
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
        'driver_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip(),
        'deterministic_algorithms':True,'tf32':False,'precision':'float32','seed':20261025,'real_damage_training_steps':0}
    matrix=torch.ones((64,64),device='cuda');value=(matrix@matrix).sum().item();assert value==262144
    info['cuda_matrix_sum']=value
    begin=time.perf_counter();model=UNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    x=torch.rand(1,3,512,512,device='cuda');target=torch.zeros(1,2,512,512,device='cuda');target[:,0,100:110,50:450]=1;target[:,1,300:350,200:280]=1
    valid=torch.ones_like(target,dtype=torch.bool);valid[:,:,500:,:]=False
    before=model.out.weight.detach().clone();losses=[];gradient_norms=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    for i in range(2):
        optimizer.zero_grad(set_to_none=True);prediction=model(x)
        loss=loss_function(prediction,target,valid);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        gradient_norms.append(float(model.out.weight.grad.norm()));losses.append(float(loss.detach()));optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(before,model.out.weight)
    info.update({'parameters':sum(p.numel() for p in model.parameters()),'synthetic_optimizer_steps':2,'losses':losses,
        'output_weight_gradient_norms':gradient_norms,'updated_parameters':True,'shape':list(prediction.shape),
        'wall_seconds_including_model_init':time.perf_counter()-begin,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    # Check independence of key cloned binaries without changing either environment.
    source=root/'.conda/buildreasonseg-proposal';clone=root/'.conda/buildreasonseg-damage-p1a'
    checks=[]
    for rel in ['python.exe','Lib/site-packages/torch/__init__.py','Lib/site-packages/torch/lib/torch_cuda.dll','Lib/site-packages/numpy/__init__.py']:
        a,b=source/rel,clone/rel
        assert not os.path.samefile(a,b),rel
        with a.open('rb') as f:ha=hashlib.file_digest(f,'sha256').hexdigest()
        with b.open('rb') as f:hb=hashlib.file_digest(f,'sha256').hexdigest()
        assert ha==hb,rel
        checks.append({'path':rel,'source_sha256':ha,'clone_sha256':hb,'same_file':False})
    info['clone_independence_samples']=checks
    (out/'engineering_gpu.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
