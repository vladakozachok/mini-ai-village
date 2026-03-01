import os
import base64

from google import genai
from google.genai import types

import village.config as cfg


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    
    return genai.Client(api_key=api_key)

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
    contents: str | list = input_text
    if image_b64:
        contents = [
            input_text,
            types.Part.from_bytes(
                data=base64.b64decode(image_b64),
                mime_type=image_media_type,
            ),
        ]

    response = client.models.generate_content(
        model=model, 
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    if not response.text:
        raise ValueError(f"Missing Gemini response text. Got {response}")

    return response.text
