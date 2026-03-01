# -*- coding: utf-8 -*-
"""
统一 LLM 调用层：支持 Gemini，可选 OpenAI 兼容接口。
入参：provider、model、api_key、system_prompt、user_message；可选 base_url（OpenAI 兼容）。
"""
from __future__ import annotations

import time
from typing import Optional

SUPPORTED_PROVIDERS = ("gemini", "openai")
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview"
)
GEMINI_MAX_OUTPUT_TOKENS_DEFAULT = 16384
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2.0


def call_llm(
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    *,
    images: Optional[list] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
    max_output_tokens: Optional[int] = None,
) -> str:
    """
    调用大模型，返回回复文本。

    Args:
        provider: 供应商，当前仅支持 "gemini"。
        model: 模型名，如 gemini-2.0-flash。
        api_key: 对应供应商的 API Key。
        system_prompt: 系统提示词（Alpha 投委会等）。
        user_message: 用户消息（拼装好的 OHLCV 等）。
        images: 可选图片列表（PIL Image 或 bytes），仅部分模型支持。
        base_url: 仅 OpenAI 兼容时使用，Gemini 忽略。
        timeout: 请求超时秒数。

    Returns:
        模型回复的纯文本。

    Raises:
        ValueError: provider 不支持或参数无效。
        RuntimeError: 调用失败或返回为空。
    """
    if not api_key or not api_key.strip():
        raise ValueError("API Key 未配置，请先在设置页录入。")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"不支持的供应商: {provider}，当前仅支持: {SUPPORTED_PROVIDERS}")

    if provider == "gemini":
        return _call_gemini(
            model=model,
            api_key=api_key.strip(),
            system_prompt=system_prompt,
            user_message=user_message,
            images=images,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return _call_openai_compatible(
            model=model,
            api_key=api_key.strip(),
            system_prompt=system_prompt,
            user_message=user_message,
            timeout=timeout,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"未实现的供应商: {provider}")


def _call_gemini(
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    images: Optional[list],
    timeout: int,
    max_output_tokens: Optional[int],
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
    )
    resolved_max_tokens = (
        int(max_output_tokens)
        if max_output_tokens is not None
        else GEMINI_MAX_OUTPUT_TOKENS_DEFAULT
    )
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max(1024, resolved_max_tokens),
    }

    contents = [user_message]
    if images:
        contents.extend(images)

    last_err: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = generative_model.generate_content(
                contents,
                generation_config=generation_config,
                request_options={"timeout": timeout},
            )
            if response is None:
                raise RuntimeError("Gemini 返回空响应")

            text = getattr(response, "text", None) or ""
            if not text and getattr(response, "candidates", None):
                parts = []
                for c in response.candidates:
                    content = getattr(c, "content", None)
                    if not content:
                        continue
                    for p in getattr(content, "parts", []) or []:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
                text = "".join(parts).strip()

            if not text:
                raise RuntimeError("Gemini 返回内容为空")

            finish_reason = ""
            if getattr(response, "candidates", None):
                finish_reason = str(getattr(response.candidates[0], "finish_reason", "") or "")
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            total_tokens = getattr(usage, "total_token_count", None) if usage else None
            print(
                "[llm] gemini model={} finish_reason={} prompt_tokens={} completion_tokens={} total_tokens={} max_output_tokens={}".format(
                    model,
                    finish_reason or "unknown",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    generation_config["max_output_tokens"],
                )
            )
            return text
        except Exception as e:
            last_err = e
            if attempt >= GEMINI_MAX_RETRIES:
                break
            sleep_s = GEMINI_RETRY_DELAY * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30.0)
            print(f"[llm] gemini attempt {attempt}/{GEMINI_MAX_RETRIES} failed: {e}; retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Gemini 调用失败: {last_err}")


def _call_openai_compatible(
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    timeout: int,
    base_url: Optional[str],
    max_output_tokens: Optional[int],
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        timeout=timeout,
    )

    # Responses API（优先），失败再回退 Chat Completions
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_output_tokens=max(1024, int(max_output_tokens or 4096)),
        )
        text = getattr(resp, "output_text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception as e:
        print(f"[llm] openai responses fallback to chat.completions: {e}")

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max(1024, int(max_output_tokens or 4096)),
        temperature=0.4,
    )
    text = (chat.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI 返回内容为空")
    return text
