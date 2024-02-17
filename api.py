import time
from pydantic import BaseModel, Field


class GenerationInput(BaseModel):
    prompt: str


# OpenAI API



# OpenAI Chat API

class ChatMessage(BaseModel):
    content: str | None = None
    # tool_calls # not implemented
    role : str = "assistant"
    name : str | None = None
    

class ChatRequest(BaseModel):
    messages: list[ChatMessage] = []
    model: str
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] = {}
    logprobs: bool = False
    top_logprobs: int = 0
    max_tokens: int | None = None
    n: int = 1
    presence_penalty: float | None = None
    #response_format
    seed: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    #tools
    #tool_choice
    #user

## OAI chat completions
class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None

class ChatCompletionLogProb(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob]

class ChatCompletionLogProbs(BaseModel):
    content: list[ChatCompletionLogProb] = []

class ChatCompletionChoice(BaseModel):
    finish_reason: str | None = Field(default=None, description='"stop", "Length", or null')
    index : int = Field(default=0, description="n > 1 (multiple choices).")
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None

class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ChatCompletion(BaseModel):
    id: str = "1234"
    choices: list[ChatCompletionChoice]
    created: int = 1234
    model: str
    system_fingerprint: str = "N.A."
    object: str = "chat.completion"
    usage: CompletionUsage | None # TODO

class ChatCompletionChunkChoice(BaseModel):
    finish_reason: str | None = Field(default=None, description='"stop", "Length", or null')
    index : int = Field(default=0, description="n > 1 (multiple choices).")
    delta: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None


class ChatCompletionChunk(BaseModel):
    id: str = "1234"
    choices: list[ChatCompletionChunkChoice]
    created: int = 1234
    model: str
    system_fingerprint: str = "N.A."
    object: str = "chat.completion"


## OAI completions
class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    stream: bool = Field(default = False)
    best_of: int | None = None
    echo: bool = False
    logit_bias: dict[int,int] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int = Field(default=1, description="This will be ignored. Batch processing is not yet implemented.")
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    suffix: str = ""
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None



class CompletionLogProbs(BaseModel):
    text_offset: list[int] = []
    token_logprobs: list[float] = []
    tokens: list[str] = []
    top_logprobs: list[dict[str, float]] | None = []

class CompletionChoice(BaseModel):
    finish_reason: str | None = 'length' # or 'stop
    index: int = 0
    logprobs: CompletionLogProbs | None = None
    text: str = ""

class Completion(BaseModel):
    id: str = "1234"
    choices: list[CompletionChoice]
    created: int = 1234
    model: str
    system_fingerprint: str = "this fingerprint does not exist"
    object: str = "chat.completion"
    usage: CompletionUsage | None = None # TODO: add this & make it mandatory!

# embeddings

class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = 'float'
    user: str = "user1234"

class Embeddings(BaseModel):
    index: int = 0
    embedding: list
    object: str = 'embedding'

class PromptRequest:
    prompt: str
class ValueResponse(BaseModel):
    value: int

# tabbyAPI stuff
class TabbyTokenEncodeRequest(BaseModel):
    add_bos_token: bool = True
    encode_special_tokens: bool = True
    decode_special_tokens: bool = True
    text: str

class TabbyTokenDecodeRequest(BaseModel):
    add_bos_token: bool = True
    encode_special_tokens: bool = True
    decode_special_tokens: bool = True
    tokens: list[int]

class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: int | None = None
    rope_scale: float = 1.0
    rope_alpha: float = 1.0
    cache_mode: str = "FP16"
    prompt_template: str | None = None
    num_experts_per_token: int | None = None
    #use_cfg: Optional[bool] = None
    #draft: Optional["ModelCard"] = None



class ModelInfo(BaseModel):
    """Represents a single model card."""

    id: str = "not available"
    object: str = "model"
    created: int = 1234 # no idea what time this represents
    owned_by: str = "no idea"
    #logging: Optional[LogPreferences] = None
    parameters: ModelCardParameters | None = None
    available_layers: int | None = None
    used_layers: list[int] | None = None




class LoadModelRequest(BaseModel):
    name: str | None = Field(default=None, description = "This is actually the path to the folder containing the safetensor and config files. The awkward name is for compatibility with TabbyAPI.")
    cache_mode: str = Field(default="FP8", description="use FP8 to save memory, or FP16 for better accuracy")
    max_seq_len: int | None = None
    layer_list: list[int] | None = None



class LayerRearrangeRequest(BaseModel):
    layer_list: list[int]

# other stuff
class ModelLoadResponse(BaseModel):
    success: bool = True

class LayerRearrangeResponse(BaseModel):
    success: bool = True

class TemplatesResponse(BaseModel):
    object: str = "list"
    data: list[str] = []

class LoadTemplateRequest(BaseModel):
    name: str