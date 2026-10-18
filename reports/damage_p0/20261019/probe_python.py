"""Read-only metadata and tiny tensor test. No model import, download or training."""
import sys
sys.dont_write_bytecode = True
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path

result = {'executable': sys.executable, 'python': sys.version,
          'packages': {}, 'gpu_smoke': {'attempted': False}}
for name in ['torch','torchvision','numpy','Pillow','opencv-python','scikit-image',
             'scipy','transformers','ultralytics','SAM-2','PyYAML','huggingface-hub']:
    try: result['packages'][name] = md.version(name)
    except md.PackageNotFoundError: result['packages'][name] = None
if importlib.util.find_spec('torch'):
    try:
        import torch
        result.update(torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result['gpu_smoke']['attempted'] = True
            p = torch.cuda.get_device_properties(0)
            result['gpu'] = {'name':p.name,'total_memory':p.total_memory,'capability':torch.cuda.get_device_capability(0)}
            x = torch.ones((64,64),device='cuda')
            y = x @ x
            torch.cuda.synchronize()
            result['gpu_smoke'].update(sum=float(y.sum()), expected=262144.0,
                passed=bool((y==64).all().item()), bf16_supported=torch.cuda.is_bf16_supported())
            del x,y
    except Exception as e: result['gpu_smoke']['error'] = repr(e)
print(json.dumps(result, ensure_ascii=False, indent=2))
