from typing import Generator
import os

import api


class StreamTokenReturn:
    chunk: str = ""
    eos: bool = True
    logprob: float = 0.0
    top_logprobs: dict[str, float] = {}

from exllamav2.generator import ExLlamaV2Sampler

GenSettings = ExLlamaV2Sampler.Settings

class SlicedLLama:
    model_name: str = ""
    layers: int = 0
    is_streaming: bool
    gen_settings : GenSettings

    def __init__(self):
        raise NotImplementedError

    def load_model(path: str, cache_type: str = "FP16", max_seq_len : int|None = None):
        raise NotImplementedError

    def unload_model():
        raise NotImplementedError

    def rearrange_layers(self, layer_list: list[int]):
        raise NotImplementedError

    def start_stream(self, prompt: str, stop_strings: list[str]=[], logprobs = 0, **kwargs):
        raise NotImplementedError

    def stream_next_token(self) -> StreamTokenReturn:
        raise NotImplementedError
    
    def stop_stream(self):
        raise NotImplementedError


    def generate_response(self, max_length: int = 9999999, stop_strings: list[str] = []) -> str:
        raise NotImplementedError
    
    def tokenize(self, text: str, add_bos_token: bool=True,
                 encode_special_tokens: bool=True, decode_special_tokens: bool=True) -> list[int]:
        raise NotImplementedError

    def decode_tokens(self, tokens: list[int], add_bos_token: bool=True,
                 encode_special_tokens: bool=True, decode_special_tokens: bool=True) -> str:
        raise NotImplementedError

    def create_embeddings(self, text) -> list[float]:
        raise NotImplementedError

    def count_tokens(self, text) -> int:
        return len(self.tokenize(text))
