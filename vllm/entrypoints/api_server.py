import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_token_enforcer_tokenizer_data
from typing import Union, List, Optional
from pydantic import BaseModel

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int

class ChatCompletionRequest(BaseModel):
    prompt: str
    stream: bool = False
    prefix_pos: Optional[int] = None
    temperature: float = 1.0


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: ChatCompletionRequest) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # request_dict = await request.json()
    # prompt = request_dict.pop("prompt")
    # prefix_pos = request_dict.pop("prefix_pos", None)
    # stream = request_dict.pop("stream", False)
    prompt = request.prompt
    prefix_pos = request.prefix_pos
    stream = request.stream
    sampling_params = SamplingParams()
    request_id = random_uuid()

    # tokenizer_data = build_vllm_token_enforcer_tokenizer_data(engine)

    # def build_vllm_token_enforcer_tokenizer_data(llm: Union[vllm.LLM, PreTrainedTokenizerBase]) -> TokenEnforcerTokenizerData:
    #     tokenizer = llm.get_tokenizer() if isinstance(llm, vllm.LLM) else llm
    #     # In some vLLM versions the tokenizer is wrapped in a TokenizerGroup
    #     if tokenizer.__class__.__name__ == 'TokenizerGroup':
    #         tokenizer = tokenizer.tokenizer  # noqa
    #     return build_token_enforcer_tokenizer_data(tokenizer)

    tokenizer = engine.engine.tokenizer.tokenizer
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    parser = JsonSchemaParser(AnswerFormat.schema())
    logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
    sampling_params.logits_processors = [logits_processor]

    results_generator = engine.generate(prompt,
                                    sampling_params,
                                    request_id,
                                    prefix_pos=prefix_pos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        # if await request.is_disconnected():
        #     # Abort the request if the client disconnects.
        #     await engine.abort(request_id)
        #     return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    print(ret)
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
