import os
import village.config as cfg

from openai import OpenAI

MAX_TOOL_CALLS = 1

API_KEY = os.environ.get('DEEPSEEK_API_KEY')

client = OpenAI(
    api_key=API_KEY, 
    base_url="https://api.deepseek.com"
    )

def generate_response(
        *, 
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int = cfg.MAX_OUTPUT_TOKENS,
        temperature: float = cfg.TEMPERATURE, 
):

    response = client.chat.completions.create(
        model = model, 
        messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_text}
        ],
        max_tokens = max_output_tokens,
        temperature = temperature,
    )

    if not response.choices[0].message.content:
        raise ValueError(f"Missing response output text. Got {response}")

    return response.choices[0].message.content