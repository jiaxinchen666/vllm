import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from typing import Union, List, Optional
from pydantic import BaseModel

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'

class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int

question = 'Please give me information about Michael Jordan. You MUST answer using the following json schema: '
question_with_schema = f'{question}{AnswerFormat.schema_json()}'
prompt = get_prompt(question_with_schema)

# def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
#     """Create a list of test prompts with their sampling parameters."""
#     return [
#         ("A robot may not injure a human being",
#          SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
#         ("To be or not to be,",
#          SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
#         ("What is the meaning of life?",
#          SamplingParams(n=2,
#                         best_of=5,
#                         temperature=0.8,
#                         top_p=0.95,
#                         frequency_penalty=0.1)),
#         ("It is only with the heart that one can see rightly",
#          SamplingParams(n=3, best_of=3, use_beam_search=True,
#                         temperature=0.0)),
#     ]

DEFAULT_MAX_NEW_TOKENS = 100
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def process_requests(engine: LLMEngine,
                     test_prompts: str):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(engine)

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            # prompt, sampling_params = test_prompts.pop(0)
            sampling_params = SamplingParams()
            sampling_params.max_tokens = DEFAULT_MAX_NEW_TOKENS
            if parser:
                logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
                sampling_params.logits_processors = [logits_processor]

            engine.add_request(str(request_id), test_prompts, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)

    test_prompts = prompt
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = "/root/Baichuan2-13B-Chat"
    args.trust_remote_code=True
    main(args)