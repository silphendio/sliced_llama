import gc
from exllamav2 import *
from exllamav2.generator import *
from exllamav2.module import ExLlamaV2Module
import sys, torch
from sliced_llama import SlicedLLama, StreamTokenReturn
from typing import Generator
import os
from copy import copy
import math

from urllib.parse import urlparse # to allow loading from path urls

#from streaming2 import ExLlamaV2StreamingGenerator2


class SlicedLLamaExl2(SlicedLLama):
    config: ExLlamaV2Config
    model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    generator: ExLlamaV2StreamingGenerator
    gen_settings: ExLlamaV2Sampler.Settings
    is_streaming: bool = False
    original_modules: list[ExLlamaV2Module] # for repeated layer rearranging
    available_layers: int = 0
    cache : ExLlamaV2Cache | ExLlamaV2Cache_8bit
    
    # extra config
    n_logprobs: int = 0

    def __init__(self):
        pass

    def load_model(self, model_path: str, cache_type: str = "FP16", max_seq_len : int|None = None):
        if hasattr(self, 'model'):
            self.model.modules = self.original_modules
            self.model.unload()
        self.cache = None

        try: model_path = urlparse(model_path).path
        except: pass
        
        self.config = ExLlamaV2Config()
        self.config.model_dir = model_path
        self.config.prepare()
        if max_seq_len:
            self.config.max_seq_len = max_seq_len
        self.model = ExLlamaV2(self.config)
        self.original_modules = self.model.modules
        self.available_layers = self.config.num_hidden_layers

        if cache_type == "FP16":
            self.cache = ExLlamaV2Cache(self.model, lazy = True)
        else:
            self.cache = ExLlamaV2Cache_8bit(self.model, lazy = True)
        
        self.model.load_autosplit(self.cache)


        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.gen_settings = ExLlamaV2Sampler.Settings()


    def rearrange_layers(self, layer_list: list[int]):
        if len(layer_list) < 1: raise ValueError
        if any([x < 0 or x >= self.available_layers for x in layer_list]):
            print("Layer index out of range!")
            raise ValueError
        # modules arangement: [embedding, [...layers], rms-norm, head]
        # where each layer is attention, mlp
        self.model.modules = self.original_modules[:1]
        for i, idx in enumerate(layer_list):
            self.model.modules += [copy(self.original_modules[idx*2 + 1])]
            self.model.modules[-1].layer_idx = i # use different cache for copied layer
            self.model.modules += [copy(self.original_modules[idx*2 + 2])]
        self.model.modules += self.original_modules[-2:]
        self.model.head_layer_idx = len(self.model.modules) -1
        self.model.config.num_hidden_layers = len(layer_list)
        self.model.last_kv_layer_idx = len(self.model.modules) -4

        # reload cache
        print("deleting old cache...")
        Cache = type(self.cache)
        del self.cache
        print("creating new cache...")
        self.model.cache_map = {}
        self.model.set_cache_map()
        self.cache = Cache(self.model)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
        print("layers sucessfully rearranged!")
        ids = [id(x) for x in self.cache.key_states]

    
    def start_stream(self, prompt: str, stop_strings: list[str]=[], logprobs = 0, **kwargs):
        print(self.gen_settings.__dict__)
        text_ids = self.tokenizer.encode(prompt, add_bos = True)
        #max_length = min(max_length, self.model.config.max_seq_len - text_ids.size(0)) # TODO: rope scaling??

        self.generator.set_stop_conditions(stop_strings)
        self.generator.begin_stream(text_ids, self.gen_settings)
        self.is_streaming = True
        self.n_logprobs = logprobs if logprobs else 0
        self.generator.return_logits = True
            
    def stream_next_token(self) -> StreamTokenReturn:
        ret = StreamTokenReturn()
        if not self.is_streaming:
            return ret
        chunk, eos, token, logits = self.generator.stream()
        ret.eos = eos
        ret.chunk = chunk

        if self.n_logprobs > 0:
            ret.top_logprobs = {}
            if token.numel() == 0:
                return ret
            
            token = token.flatten()[0]

            logprobs = logits[0].log_softmax(-1)
            ret.logprob = float(logprobs[token])
            top_logprobs = logprobs.topk(self.n_logprobs)

            for (prob, tok) in zip(*top_logprobs):
                tok_str = self.tokenizer.decode(tok.reshape(1,1))[0]
                # sometimes, two tokens have the same string
                if tok_str in ret.top_logprobs: # adding logits together is complicated
                    ret.top_logprobs[tok_str] = math.log(math.exp(float(prob)) +
                                                         math.exp(ret.top_logprobs[tok_str]))
                else:
                    ret.top_logprobs[tok_str] = float(prob)
            
        return ret
    
    def stop_stream(self):
        self.is_streaming = False

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_bos = True).flatten().tolist()

    def generate_response(self, max_length: int = 9999999, stop_strings: list = []) -> str:
        text = ""
        self.start_stream()
        for i in range(max_length):
            chunk, eos, _ = self.generator.stream()
            text += chunk
            if eos: return text
        return text


    def create_embeddings(self, text) -> list[float]:
        ids = self.tokenize(text)
        return my_sliced_llama.model.forward(ids, return_last_state=True)[1].flatten().tolist()

    def decode_tokens(self, tokens: list[int], add_bos_token: bool=True,
                 encode_special_tokens: bool=True, decode_special_tokens: bool=True) -> str:
        self.tokenizer.decode(torch.tensor(tokens))

# test code

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
    my_sliced_llama : SlicedLLama = SlicedLLamaExl2()

    my_sliced_llama.load_model('TinyLlama-1.1B-Chat-v1.0-3.0bpw-h6-exl2')
    print("load #1")
    import time
    time.sleep(3.0)
    my_sliced_llama.load_model('TinyLlama-1.1B-Chat-v1.0-3.0bpw-h6-exl2')
    print("load #2")

    layer_list = create_layer_list('0-14, 8-22')

    finish_reason = "length"
    text = ""
    max_tokens = 512
    my_sliced_llama.start_stream("Once upon a time")
    for i in range(max_tokens):
        chunk, eos, _ = my_sliced_llama.stream_next_token()
        text += chunk
        print(chunk, end='')
        if eos:
            finish_reason = "stop"
            break

    print("--------------------")

    layer_list = create_layer_list('0-22')

    finish_reason = "length"
    text = ""
    max_tokens = 512
    my_sliced_llama.start_stream("Once upon a time")
    for i in range(max_tokens):
        chunk, eos, _ = my_sliced_llama.stream_next_token()
        text += chunk
        print(chunk, end='')
        if eos:
            finish_reason = "stop"
            break

    print("--------------------")


    

