import asyncio
import json
from fastapi.encoders import jsonable_encoder
from sse_starlette import EventSourceResponse
from api import *
from sliced_llama import SlicedLLama
from fastapi import APIRouter

llm : SlicedLLama

router = APIRouter()

@router.post("/v1/completions", response_model=CompletionChoice)
async def oai_completions(req: CompletionRequest)  -> CompletionChoice | EventSourceResponse:
    print("request received: ", req)

    # update gen settings (TODO: do this properly?)
    llm.gen_settings.__dict__ = dict(llm.gen_settings.__dict__,  **req.model_dump())

    # handle parameters
    stop_str = req.stop
    if stop_str is str: stop_str = [stop_str]
    if stop_str is None: stop_str = []


    if req.stream:
        async def event_generator():
            ret_logprobs = CompletionLogProbs() if req.logprobs and req.logprobs > 0 else None
            text = ''
            finish_reason = None
            if not req.max_tokens: req.max_tokens = 99999999
            llm.start_stream(req.prompt, stop_strings=stop_str, logprobs=req.logprobs)
            try:
                for i in range(req.max_tokens):
                    ret = llm.stream_next_token()
                    text += ret.chunk
                    print(ret.chunk, end='')
                    if ret.eos:
                        finish_reason = "stop"
                        break

                    # log probs
                    if ret_logprobs:
                        ret_logprobs = CompletionLogProbs(
                            text_offset = [0],
                            token_logprobs = [ret.logprob],
                            tokens = [ret.chunk],
                            top_logprobs = [ret.top_logprobs],
                        )

                    stream_chunk_data = Completion(
                        model = llm.model_name,
                        choices = [CompletionChoice(finish_reason=finish_reason, text=ret.chunk, logprobs=ret_logprobs)]
                        )
                    yield json.dumps(jsonable_encoder(stream_chunk_data)) # remove class names, fix braces and quotes
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                print(" - STREAM CANCELLED")

        return EventSourceResponse(event_generator())
    
    else: # not streaming
        text = ''
        finish_reason = "length"
        ret_logprobs = CompletionLogProbs() if req.logprobs and req.logprobs > 0 else None
        if not req.max_tokens: req.max_tokens = 99999999
        llm.start_stream(req.prompt)
        for _ in range(req.max_tokens):
            ret = llm.stream_next_token()

            # logprobs
            if ret_logprobs:
                ret_logprobs.text_offset += [len(text)],
                ret_logprobs.token_logprobs += [ret.logprob],
                ret_logprobs.tokens = [ret.chunk],
                ret_logprobs.top_logprobs = [ret.top_logprobs],

            text += ret.chunk
            if ret.eos:
                finish_reason = "stop"
                break
            
             # logprobs

        return Completion(
            model = llm.model_name,
            choices = [CompletionChoice(finish_reason=finish_reason, text=text, logprobs=ret_logprobs)]
            )
