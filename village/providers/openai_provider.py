import os

from openai import OpenAI
import village.config as cfg

MAX_TOOL_CALLS = 1
API_KEY = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key = API_KEY)

def generate_response(
        *, 
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int = cfg.MAX_OUTPUT_TOKENS,
        temperature: float = cfg.TEMPERATURE,
):
    request = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
    }

    if model.startswith("gpt-5"):
        request["text"] = {"verbosity": cfg.OPENAI_TEXT_VERBOSITY}
        request["reasoning"] = {"effort": cfg.OPENAI_REASONING_EFFORT}
    else:
        request["temperature"] = temperature

    response = client.responses.create(**request)

    if response.output_text:
        return response.output_text

    incomplete_reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
    if incomplete_reason == "max_output_tokens":
        retry_request = dict(request)
        retry_request["max_output_tokens"] = min(cfg.MAX_OUTPUT_TOKENS_RETRY, max_output_tokens * 2)
        retry_response = client.responses.create(**retry_request)
        if retry_response.output_text:
            return retry_response.output_text
        raise ValueError(
            "Model response was truncated at max_output_tokens and retry also returned no output text. "
            f"Retry tokens: {retry_request['max_output_tokens']}. Response: {retry_response}"
        )

    raise ValueError(f"Missing response output text. Got {response}")
