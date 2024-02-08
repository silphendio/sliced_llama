# sliced_llama
Simple LLM inference server

## Features:
- partly OpenAI-compatible API (this is a work in progress)
- Layer Slicing: Basically instant Franken-self-merges. You don't even need to reload the model (just the cache).
- Top Logprobs: See the top probabilities for each chosen token. This might help with adjusting sampler parameters.
- Text Completion WebUI

## Installation:
##### Linux:
- Make sure `python` and `git` is installed. You need at least version 3.11
- run
```bash
git clone --depth=1 https://github.com/silphendio/sliced_llama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
On Windows it's the same thing. Just install `python` and `git` from the Windows Store, if you haven't already, then open `powershell` and use
```cmd
git clone --depth=1 https://github.com/silphendio/sliced_llama
python3 -m venv .venv
venv\Scripts\activate.ps1
pip install -r requirements.txt
```
DISCLAIMER: I haven't tested it on windows at all.

## Usage:
to run it, do:
```bash
source .venv/bin/activate # if it isn't already activated
python sliced_llama_server.py
```
This starts the inference server and the webUI. There, you can load models, adjust parameters and do inference.

You can also connecct the server to a different UI, like [Mikupad](https://github.com/lmg-anon/mikupad) or [SillyTavern](https://github.com/SillyTavern/SillyTavern).

For SillyTavern, select chat completion, and use `http://0.0.0.0:57593/v1` as costum endpoint. 

## TODO / missing features:
- configuration file
- OpenAI API:
  - chat completion currently only supports ChatML and only works with streaming
  - `presency_penalty` and `frequency_penalty` aren't supported
  - authentication
  - usage statistics
