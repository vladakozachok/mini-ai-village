import os

from openai import OpenAI
import village.config as cfg

MAX_TOOL_CALLS = 1
API_KEY = os.environ.get('OPEN_API_KEY')

client = OpenAI(api_key = API_KEY)

def generate_response(
        *, 
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int = cfg.MAX_OUTPUT_TOKENS,
        temperature: float = cfg.TEMPERATURE, 
):
    response = client.responses.create(
        model = model, 
        instructions = instructions,
        input = input_text,
        max_output_tokens = max_output_tokens,
        temperature = temperature,
    )

    if not response.output_text:
        raise ValueError(f"Missing response output text. Got {response}")

    return response.output_text 