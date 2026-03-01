import os

from openai import OpenAI
import village.config as cfg

MAX_TOOL_CALLS = 1


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)

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
    if image_b64:
        user_content = [
            {
                "type": "input_text", 
                "text": input_text},
            {
                "type": "input_image",
                "image_url": f"data:{image_media_type};base64,{image_b64}",
            },
        ]
        request_input = [{"role": "user", "content": user_content}]
    else:
        request_input = input_text

    request = {
        "model": model,
        "instructions": instructions,
        "input": request_input,
        "max_output_tokens": max_output_tokens,
    }

    if model.startswith("gpt-5"):
        request["text"] = {"verbosity": cfg.OPENAI_TEXT_VERBOSITY}
        request["reasoning"] = {"effort": cfg.OPENAI_REASONING_EFFORT}
    elif model.startswith("gpt-4o"):
        request["temperature"] = temperature

    response = client.responses.create(**request)

    if not response.output_text:
        raise ValueError(f"Missing response output text. Got {response}")
    
    return response.output_text
