import asyncio
import json
from fastapi.encoders import jsonable_encoder
from sse_starlette import EventSourceResponse
from api import *
from sliced_llama import SlicedLLama
from fastapi import APIRouter

import jinja2

llm : SlicedLLama

jinja2_env = jinja2.Environment()
chatml_template_string = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
chatml_template = jinja2_env.from_string(chatml_template_string)
chatml_start_str = '<|im_start|>assisstant\n'
chatml_stop_str = '<|im_end|>'

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletion)
async def oai_chat_completions(req: ChatRequest)  -> ChatCompletion | EventSourceResponse:
    print("request received: ", req.model_dump())
    print("----------------------------------")

    prompt = chatml_template.render(req.model_dump())
    prompt += chatml_start_str
    print(prompt)
    print("#######################")
    print("prompt in ChatML format:\n" + prompt + "\n")

    # update gen settings (TODO: do this properly?)
    llm.gen_settings.__dict__ = dict(llm.gen_settings.__dict__,  **req.__dict__)

    # handle parameters
    stop_str = req.stop
    if stop_str is str: stop_str = [stop_str]
    if stop_str is None: stop_str = []
    stop_str += [chatml_stop_str]


    if req.stream:
        async def event_generator():
            ret_logprobs = ChatCompletionLogProbs() if req.logprobs and req.logprobs > 0 else None
            text = ''
            finish_reason = None
            if not req.max_tokens: req.max_tokens = 99999999
            llm.start_stream(prompt, stop_strings=stop_str, logprobs=req.logprobs)
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
                        ret_logprobs.content = ChatCompletionLogProb(
                            text_offset = [0],
                            logprob = ret.logprob,
                            token = ret.chunk,
                            top_logprobs = [ TopLogprob(token=tok, logprob=lp) for tok,lp in ret.top_logprobs.items()],
                        )

                    stream_chunk_data = ChatCompletionChunk(
                        model = llm.model_name,
                        choices = [ChatCompletionChunkChoice(
                            finish_reason=finish_reason,
                            delta=ChatMessage(content=ret.chunk),
                            logprobs=ret_logprobs
                        )]
                    )
                    yield json.dumps(jsonable_encoder(stream_chunk_data)) # remove class names, fix braces and quotes
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                print(" - STREAM CANCELLED")

        return EventSourceResponse(event_generator())
    
    else: # not streaming
        raise NotImplementedError # TODO
