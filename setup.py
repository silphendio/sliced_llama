#!/usr/bin/env python3

from subprocess import run
import sys
import os

if not os.path.isdir(".venv"):
    run(["python3", "-m", "venv", ".venv"])

requirements_file = "requirements.txt"

try:
    a = run(["nvcc", "--version"], capture_output=True)
    if b'release 12' in a.stdout:
        requirements_file = "requirements-cu122.txt"
        print("found CUDA 12")
    elif b'release 11' in a.stdout:
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


run(["./.venv/bin/pip", "install", "-r", requirements_file])
run(["echo", "install", "-r", requirements_file])
