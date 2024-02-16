#!/usr/bin/env python3

from subprocess import run
import sys
import os

# create venv, try different python commands
if not os.path.isdir(".venv"):
    print("creating virtual environment...")
    try:
        run(["python", "-m", "venv", ".venv"])
    except: pass
if not os.path.isdir(".venv"):
    try:
        run(["python3", "-m", "venv", ".venv"])
    except: pass
if not os.path.isdir(".venv"):
    print("failed to create virtual environment.")


requirements_file = "requirements.txt"

# TODO: switch to conda to make sure that there's always a compatble python version
if sys.version.startswith("3.11") or sys.version.startswith("3.10"):
    try:
        a = run("nvidia-smi", capture_output=True)
        if b'CUDA Version: 12' in a.stdout:
            requirements_file = "requirements-cu122.txt"
            print("found CUDA 12")
        elif b'CUDA Version: 11' in a.stdout:
            requirements_file = "requirements-cu118.txt"
            print("found CUDA 11")
    except:
        print("no CUDA found")
        try:
            run("rocminfo", capture_output=True)
            # TODO: check version
            print("ROCm found")
        except:
            print("no ROCm found")

if sys.platform == 'win32':
    run(["./.venv/Scripts/pip", "install", "-r", requirements_file])
else:
    run(["./.venv/bin/pip", "install", "-r", requirements_file])
