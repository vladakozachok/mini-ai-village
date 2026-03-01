import os
import village.config as cfg

from openai import OpenAI

MAX_TOOL_CALLS = 1


def _get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY environment variable.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

def generate_response(
        *, 
        model: str,
        instructions: str,
        input_text: str,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_output_tokens: int = cfg.MAX_OUTPUT_TOKENS,
        temperature: float = cfg.TEMPERATURE, 
):
    client = _get_client()

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
