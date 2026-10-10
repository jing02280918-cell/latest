import logging
import os

import dotenv
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("together").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logger = logging.getLogger("query_functions")

dotenv.load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

anthropic_client = None
together_client = None
gemini_client = None

try:
    openai_client = OpenAI(api_key=OPENAI_KEY)
except Exception as e:
    logger.warning(f"Error initializing OpenAI client: {e}")
    openai_client = None


def _get_anthropic_client():
    global anthropic_client
    if anthropic_client is not None:
        return anthropic_client
    try:
        import anthropic

        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)  # type: ignore[misc]
    except Exception as e:
        logger.warning(f"Error initializing Anthropic client: {e}")
        anthropic_client = None  # type: ignore[misc]
    return anthropic_client


def _get_together_client():
    global together_client
    if together_client is not None:
        return together_client
    try:
        from together import Together

        together_client = Together(api_key=TOGETHER_KEY)  # type: ignore[misc]
    except Exception as e:
        logger.warning(f"Error initializing Together client: {e}")
        together_client = None  # type: ignore[misc]
    return together_client


def _get_gemini_client():
    global gemini_client
    if gemini_client is not None:
        return gemini_client
    try:
        from google import genai

        gemini_client = genai.Client(api_key=GEMINI_KEY)  # type: ignore[misc]
    except Exception as e:
        logger.warning(f"Error initializing Gemini client: {e}")
        gemini_client = None  # type: ignore[misc]
    return gemini_client


def query_openai(
    prompt,
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=2048,
    system_prompt=None,
    message_history=[],
    **kwargs,
):
    messages = []
    if system_prompt:
        if message_history and message_history[0]["role"] == "system":
            logger.warning(
                "System prompt is provided but the first message in the message history is also a system message. This is likely an error. Ignoring the system prompt."
            )
        else:
            messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})
    messages = message_history + messages

    assert openai_client is not None

    if model in ["o1", "o1-mini", "o3-mini"]:
        chat_completion = openai_client.chat.completions.create(
            messages=messages,
            model=model,
        )
    elif model in ["o3-2025-04-16", "gpt-5-2025-08-07"]:
        chat_completion = openai_client.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=max_tokens,
        )
    else:
        chat_completion = openai_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            safety_identifier="user_123456",
        )
    return chat_completion.choices[0].message.content


def query_anthropic(
    prompt,
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=2048,
    system_prompt=None,
    message_history=[],
    **kwargs,
):
    messages_params = {
        "model": model,
        # "temperature": temperature,
        "messages": message_history + [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    if system_prompt:
        messages_params["system"] = system_prompt

    client = _get_anthropic_client()
    assert client is not None
    response = client.messages.create(**messages_params)
    try:
        return response.content[0].text
    except Exception as e:
        raise Exception("content filtering policy blocked the response")


def query_together(
    prompt,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=2048,
    system_prompt=None,
    message_history=[],
    **kwargs,
):
    messages = []
    if system_prompt:
        if message_history and message_history[0]["role"] == "system":
            logger.warning(
                "System prompt is provided but the first message in the message history is also a system message. This is likely an error. Ignoring the system prompt."
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages = message_history + messages
    client = _get_together_client()
    assert client is not None
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response_text = response.choices[0].message.content

    cleaned_response = response_text
    cleaned_response = _remove_thinking_tags(response_text)
    return cleaned_response


def _remove_thinking_tags(response: str) -> str:
    if not response or "<think>" not in response:
        return response

    cleaned = response
    think_start = cleaned.find("<think>")
    think_end = cleaned.find("</think>")
    while think_start != -1 and think_end != -1 and think_end > think_start:
        before_think = cleaned[:think_start].strip()
        after_think = cleaned[think_end + 8 :].strip()
        cleaned = before_think + " " + after_think

        # Look for more <think> sections
        think_start = cleaned.find("<think>")
        think_end = cleaned.find("</think>")

    return cleaned.strip()


def query_embeddings(
    text,
    model="Qwen/Qwen3-Embedding-0.6B",
    batch_size=100,
    port=8003,
):
    single_input = False
    if isinstance(text, str):
        text = [text]
        single_input = True

    try:
        vllm_client = OpenAI(
            api_key="token-abc12", base_url=f"http://localhost:{port}/v1"
        )
    except Exception as e:
        logger.warning(f"Error initializing vLLM client: {e}")
        vllm_client = None

    all_embeddings = []

    assert vllm_client is not None

    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]

        response = vllm_client.embeddings.create(
            input=batch,
            model=model,
        )

        batch_embeddings = []
        for item in response.data:
            batch_embeddings.append(item.embedding)

        all_embeddings.extend(batch_embeddings)

    if single_input and all_embeddings:
        return all_embeddings[0]

    return all_embeddings


def query_gemini(
    prompt,
    model="gemini-2.5-pro-preview-03-25",
    temperature=0,
    max_tokens=2048,
    system_prompt=None,
    message_history=[],
):
    from google.genai import types

    client = _get_gemini_client()
    contents = []

    if system_prompt:
        contents.append(
            types.Content(
                role="system", parts=[types.Part.from_text(text=system_prompt)]
            )
        )

    for msg in message_history:
        if msg.get("content"):
            role = msg["role"]
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role, parts=[types.Part.from_text(text=msg["content"])]
                )
            )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    assert client is not None

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text


def _dashscope_extra_body(model: str) -> dict | None:
    model_lc = model.lower()
    if model.startswith("ZHIPU/") or model_lc.startswith("zhipu/"):
        effort = os.getenv("VDB_GLM_REASONING_EFFORT", "low")
        return {"enable_thinking": True, "reasoning_effort": effort}
    return None


def query_dashscope(
    prompt,
    model="qwen3.7-plus",
    temperature=0.0,
    max_tokens=4096,
    system_prompt=None,
    message_history=[],
    **kwargs,
):
    del kwargs
    dotenv.load_dotenv("/root/vdriftbench/.env")
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("VDB_API_KEY", "EMPTY")
    base_url = os.getenv(
        "VDB_TARGET_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    timeout_s = 300.0 if (model.startswith("ZHIPU/") or model.lower().startswith("zhipu/")) else 120.0
    timeout_s = float(os.getenv("VDB_TARGET_TIMEOUT_S", str(timeout_s)))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    messages = []
    if system_prompt:
        if message_history and message_history[0]["role"] == "system":
            logger.warning(
                "System prompt is provided but the first message in the message history is also a system message. Ignoring the system prompt."
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages = message_history + messages
    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    extra_body = _dashscope_extra_body(model)
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message
    text = msg.content or ""
    if not text.strip():
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            text = reasoning
    return text


def query_deepseek(
    prompt,
    model="deepseek-v4-pro",
    temperature=0.0,
    max_tokens=4096,
    system_prompt=None,
    message_history=[],
    **kwargs,
):
    del kwargs
    dotenv.load_dotenv("/root/vdriftbench/.env")
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("VDB_API_KEY", "EMPTY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = []
    if system_prompt:
        if message_history and message_history[0]["role"] == "system":
            logger.warning(
                "System prompt is provided but the first message in the message history is also a system message. Ignoring the system prompt."
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages = message_history + messages
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def query_vllm_server(
    prompt,
    model="Qwen/QwQ-32B",
    temperature=0.0,
    max_tokens=2048,
    system_prompt=None,
    port=8000,
    message_history=[],
    seed=None,
    **kwargs,
):
    del kwargs
    # vLLM always available or hosted at http://localhost:8000
    # Completions API is the same as OpenAI's API /v1/chat/completions

    try:
        vllm_client = OpenAI(
            api_key="token-abc12", base_url=f"http://localhost:{port}/v1"
        )
    except Exception as e:
        logger.warning(f"Error initializing vLLM client: {e}")
        vllm_client = None

    messages = []
    if system_prompt:
        if message_history and message_history[0]["role"] == "system":
            logger.warning(
                "System prompt is provided but the first message in the message history is also a system message. This is likely an error. Ignoring the system prompt."
            )
        else:
            messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})
    messages = message_history + messages

    assert vllm_client is not None

    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        create_kwargs["seed"] = seed
    response = vllm_client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content


def query_vllm_embed_server(
    prompt,
    model="Qwen/Qwen3-Embedding-0.6B",
    batch_size=100,
    port=8003,
):
    try:
        vllm_client = OpenAI(
            api_key="token-abc12", base_url=f"http://localhost:{port}/v1"
        )
    except Exception as e:
        logger.warning(f"Error initializing vLLM client: {e}")
        vllm_client = None
        
    single_input = False
    if isinstance(prompt, str):
        text = [prompt]
        single_input = True
    else:
        text = prompt

    all_embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]

        response = vllm_client.embeddings.create(
            input=batch,
            model=model,
        )

        batch_embeddings = []
        for item in response.data:
            batch_embeddings.append(item.embedding)

        all_embeddings.extend(batch_embeddings)

    if single_input and all_embeddings:
        return all_embeddings[0]

    return all_embeddings
