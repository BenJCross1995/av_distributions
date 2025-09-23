import os
import json

from typing import Optional, Any
from openai import OpenAI

# My own modules
from read_and_write_docs import read_jsonl

def initialise_client(credential_loc: Optional[str] = None, env_var: str = "OPENAI_API_KEY") -> OpenAI:
    """
    If `credential_loc` is provided, use your existing `read_jsonl(credential_loc)` to find `env_var`
    (default 'OPENAI_API_KEY') and initialise the OpenAI client with it.
    Otherwise, try the environment variable. If neither yields a key, raise RuntimeError.
    Returns: OpenAI client.
    """
    key: Optional[str] = None

    if credential_loc:
        with open(credential_loc) as f:
            credentials = json.load(f)
            key = credentials['OPENAI_API_KEY']
        if not key:
            raise RuntimeError(f"'{env_var}' not found in {credential_loc}")
    else:
        # fall back to environment
        env_val = os.environ.get(env_var)
        key = env_val.strip()

    if not key:
        raise RuntimeError(f"Missing API key. Set {env_var} in your environment or provide `credential_loc` with that field.")

    return OpenAI(api_key=key)

def llm(
    system_prompt: str,
    prompt: str,
    client: OpenAI,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 256,
    temperature: float = 0.7,
    **create_kwargs: Any,   # <— extras forwarded to the API call
) -> str:
    """
    Basic LLM call: returns only the generated text.
    If no client is provided, it will create one via `initialise_client()`.
    """

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        **create_kwargs, 
    )

    return resp or ""