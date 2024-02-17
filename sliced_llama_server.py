#!/usr/bin/env -S sh -c '"`dirname $0`/.venv/bin/python" "$0" "$@"'

# run `uvicorn main:app --reload`
import argparse
import webbrowser
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware


from fastapi.staticfiles import StaticFiles

from api import *
import os

from sliced_llama import SlicedLLama
from sliced_llama_exl2 import SlicedLLamaExl2
import serve_completions as compl
import serve_chat_completions as chat_compl

import sys
basedir_path = os.path.abspath(os.path.dirname(sys.argv[0]))


llm : SlicedLLama = SlicedLLamaExl2()
server_host = '0.0.0.0'
server_port = 7777

compl.llm = llm
chat_compl.llm = llm
chat_compl.template_folders = [os.path.join(basedir_path, "chat_templates")]


# server stuff
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(compl.router)
app.include_router(chat_compl.router)
        

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
        {
            "id": llm.model_name,
            "object": "model",
            "created": 0, # too lazy to look up a real number
            "owned_by": "dunno"
        },
        ]
    }


# tabbyAPI stuff
@app.post('/v1/token/encode')
def tabby_encode_token(req: TabbyTokenEncodeRequest) -> list[int]:
    return llm.tokenize(req.text)

@app.post('/v1/token/decode')
def tabby_decode_token(req: TabbyTokenDecodeRequest) -> str:
    return llm.decode_tokens(req.tokens)

# TODO
@app.get('/v1/internal/model/info')
def get_model_info() -> ModelInfo:
    print(llm)
    model_card = ModelInfo()
    model_card.id = llm.model_name
    model_card.available_layers = llm.available_layers
    return model_card

@app.post('/v1/model/load')
def load_model(req: LoadModelRequest) -> ModelLoadResponse:
    if req.name:
        try:
            llm.load_model(req.name, req.cache_mode, max_seq_len=req.max_seq_len)

            # use folder name as model name
            if req.name[-1] in "/\\":
                req.name = req.name[:-1]
            llm.model_name = os.path.basename(req.name)
            return ModelLoadResponse(success=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    if req.layer_list:
        try:
            llm.rearrange_layers(req.layer_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# layer slicing
@app.post('/v1/rearrange_layers')
def rearrange_layers(req: LayerRearrangeRequest):
    try:
        llm.rearrange_layers(req.layer_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# from koboldcpp
@app.get('api/extra/tokencount')
def kobold_cpp_tokencount(req):
    """for compatability reasons only. Please don't use this"""
    return { 'value': llm.tokenizer.count_tokens(req.prompt)}


# Barebones WebUI
app.mount("/", StaticFiles(directory= os.path.join(basedir_path, "webui"), html=True))



def create_layer_list(slices_str: str) -> list[int]:
    try:
        layers = []
        for s in slices_str.split(','):
            a,b = s.split('-')
            layers += list(range(int(a), int(b)))
        return layers
    
    except ValueError:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="path to exl2 folder")
    parser.add_argument("-c", "--cache-mode", help="FP8 or FP16", default="FP16")
    parser.add_argument("-p", "--port", help="A unique number below 65535", type=int, default=57593)
    parser.add_argument("-s", "--slices", help="layers to use, e.g.: '0-24, 8-32'")
    parser.add_argument("-l", "--max-seq-len", "--context-size", help="context size, a lower number saves VRAM", type=int)
    args = parser.parse_args()

    if args.model != None:
        load_model(LoadModelRequest(name=args.model, cache_mode=args.cache_mode, max_seq_len=args.max_seq_len))
        if args.slices != None:
            layer_list = create_layer_list(args.slices)
            llm.rearrange_layers(layer_list)


    server_port = args.port

    webbrowser.open(server_host + ":" + str(server_port))

    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
        log_level="info",
    )
