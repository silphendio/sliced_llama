# sliced_llama
Simple LLM inference server
## Installation:
- Make sure `python` is installed.
- run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Usage:
to run it, do:
```bashsource .venv/bin/activate # if it isn't already activated
python sliced_llama_server.py
```
This starts the inference server and the webUI. In the webUI, you can adjust parameters and load a model.
You can also connecct the inference server wiwth a different UI, like Mikupad or SillyTavern.

For SillyTavern, select chat completion, and use `http://0.0.0.0:57593/v1` as costum endpoint.
