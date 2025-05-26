# | You called `.half()` / `.float()` on the whole model **after** the `.to(device)`; that creates *new* parameters on the default device (CPU). | Always chain the calls: `model.to(device).half()` or call `.half()` **before** `.to(device)`. |

# ---

### 3 · Minimal micro-benchmark

# Paste this in an IPython shell to be 100 % certain the CUDA kernels are used:

# ```python

import sys
sys.path.append('/data_sata/ReID_Group/ReID_Group/KANTransfarmers/rational_kat_cu')

# import torch
# from kat_rational import KAT_Group       # same import the model uses

# kat = KAT_Group(mode='swish').cuda()
# inp = torch.randn(1024, 1024, device='cuda')

# # warm-up
# for _ in range(10):
#     kat(inp)

# torch.cuda.synchronize()
# import time, contextlib
# with contextlib.ExitStack() as stack:
#     start = time.time()
#     for _ in range(100):
#         kat(inp)
#     torch.cuda.synchronize()
#     print("elapsed ms:", (time.time() - start) * 1000 / 100)
# gelu_vs_kat.py
# import torch, time, contextlib, gc
# from kat_rational import KAT_Group

# gelu = torch.nn.GELU().cuda()
# kat  = KAT_Group(mode='swish').cuda()

# inp = torch.randn(64, 197, 768, device='cuda')
# for _ in range(20): gelu(inp); kat(inp)          # warm-up

# def bench(mod):
#     torch.cuda.synchronize(); t0=time.time()
#     for _ in range(100): mod(inp)
#     torch.cuda.synchronize(); return (time.time()-t0)*10

# print('GELU 100 iters  :', bench(gelu), 'ms')
# print('KAT  100 iters  :', bench(kat ), 'ms')


import os
import zipfile
import shutil

def create_zip_file():
    # Define the files to include
    files_to_zip = [
        'model/backbones/kat_pytorch_org.py',
        'model/backbones/vit_pytorch.py',
        'model/make_model.py',
        'configs/Market/kat_market1.yml', 
        'configs/Market/vit_base.yml',
        'processor/processor_org.py',
        'katransformer.py',
        'train.py',
        'test.py'
    ]
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create zip file
    zip_path = os.path.join(output_dir, 'kat_files_24.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if os.path.exists(file_path):
                # Get just the filename without path
                filename = os.path.basename(file_path)
                # Add file to zip
                zipf.write(file_path, filename)
                print(f"Added {filename} to zip")
            else:
                print(f"Warning: {file_path} not found")
                
    print(f"\nZip file created at: {zip_path}")
    
if __name__ == "__main__":
    create_zip_file()
