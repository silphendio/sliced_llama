#!/usr/bin/env python3
from subprocess import run
import sys
import os

# create venv, try different python commands
python_cmds = ["python", "python3", "py"]
print("creating virtual environment...")
for cmd in python_cmds:
    if os.path.isdir(".venv"):
        break
    try:
        run([cmd, "-m", "venv", ".venv"])
    except: pass
if not os.path.isdir(".venv"):
    print("failed to create virtual environment.")


# determine platform
exl2_version = "0.1.5"
torch_str = "torch2.3.1"
torch_str_fa_linux = "torch2.3"

fa_version = "2.5.9.post1"
fa_gpu_str = ""

py_str = "cp" + ''.join(sys.version.split(".")[:2])

gpu_str = "" # CUDA or ROCm versions
os_str = "" #"win_amd64" or "linux_x86_64"

# check for CUDA or ROCm
try:
    a = run("nvidia-smi", capture_output=True)
    if b'CUDA Version: 12' in a.stdout:
        gpu_str = "cu121"
        fa_gpu_str = "cu122"
    elif b'CUDA Version: 11' in a.stdout:
        gpu_str = "cu118"
except:
    try:
        a = run("rocminfo", capture_output=True)
        major_version = int(a.stdout.decode('utf-8').split('Runtime Version')[1].strip().split('.')[0])
        
        if major_version == 5:
            gpu_str = "rocm5.6"
        elif major_version >= 6:
            gpu_str = "rocm6.0"
    except:
        print("no CUDA or ROCm found")

# determine OS
if sys.platform == 'win32':
    os_str = "win_amd64"
if sys.platform == 'linux':
    os_str = "linux_x86_64"


## specify requirenments
requirements = [
    "fastapi",
    "sse-starlette",
    "uvicorn",
    "tokenizers",
    "setuptools",
]

# exllamav2
if os_str and gpu_str:
    requirements.append(f"https://github.com/turboderp/exllamav2/releases/download/v{exl2_version}/exllamav2-{exl2_version}+{gpu_str}.{torch_str}-{py_str}-{py_str}-{os_str}.whl")
else:
    requirements.append("exllamav2")

# flash_attn
if gpu_str.startswith("cu") :
    if os_str == "win_amd64":
        requirements.append(f"https://github.com/bdashore3/flash-attention/releases/download/v{fa_version}/flash_attn-{fa_version}+{fa_gpu_str}{torch_str}cxx11abiFALSE-{py_str}-{py_str}-{os_str}.whl")

    # linux
    if os_str == "linux_x86_64":
        requirements.append(f"https://github.com/bdashore3/flash-attention/releases/download/v{fa_version}/flash_attn-{fa_version}+{fa_gpu_str}{torch_str_fa_linux}cxx11abiFALSE-{py_str}-{py_str}-{os_str}.whl")
# there is no fallback for flash attention, because it's optional


# install requirements

pip_cmd = "./.venv/Scripts/pip" if sys.platform == 'win32' else "./.venv/bin/pip"

# pytorch
if gpu_str.startswith("cu") or os_str == "linux_x86_64":
    run([pip_cmd, "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{gpu_str}"])

for req in requirements:
    run([pip_cmd, "install", req])
