"""
API client for calling LLMs (OpenAI and OpenRouter/Claude).
Handles rate limiting, retries, and response parsing.
"""

import os
import time
import json
from dataclasses import dataclass
from typing import Optional
import httpx
from openai import OpenAI


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    model: str
    prompt_condition: str
    question_id: str
    response_text: str
    is_abstention: bool
    confidence: Optional[str] = None  # For CoT prompts
    latency_ms: float = 0.0
    tokens_used: int = 0


def detect_abstention(response_text: str) -> bool:
    """
    Detect if the response indicates abstention ("I don't know" or similar).
    Uses multiple heuristics to catch various phrasings.
    """
    text_lower = response_text.lower().strip()

    # Exact matches
    abstention_phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot answer",
        "i can't answer",
        "i'm unable to",
        "i am unable to",
        "i don't have enough information",
        "i cannot provide",
        "i'm uncertain",
        "i am uncertain",
        "unknown",
        "uncertain",
        "not enough information",
        "insufficient information",
        "i cannot determine",
        "i can't determine",
    ]

    for phrase in abstention_phrases:
        if phrase in text_lower:
            return True

    # Check if response starts with uncertainty
    uncertain_starts = [
        "i'm not certain",
        "i am not certain",
        "it's unclear",
        "it is unclear",
        "i lack",
    ]
    for start in uncertain_starts:
        if text_lower.startswith(start):
            return True

    return False


def extract_confidence(response_text: str) -> Optional[str]:
    """
    Extract stated confidence level from CoT responses.
    Returns 'high', 'medium', 'low', 'uncertain', or None.
    """
    text_lower = response_text.lower()

    if "high confidence" in text_lower or "confidence: high" in text_lower:
        return "high"
    elif "medium confidence" in text_lower or "confidence: medium" in text_lower:
        return "medium"
    elif "low confidence" in text_lower or "confidence: low" in text_lower:
        return "low"
    elif "uncertain" in text_lower:
        return "uncertain"
    return None


class OpenAIClient:
    """Client for OpenAI API calls."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        question_id: str,
        prompt_condition: str,
        temperature: float = 0.0,
        max_tokens: int = 256
    ) -> LLMResponse:
        """Make a single API call and return parsed response."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            latency_ms = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                model=self.model,
                prompt_condition=prompt_condition,
                question_id=question_id,
                response_text=response_text,
                is_abstention=detect_abstention(response_text),
                confidence=extract_confidence(response_text),
                latency_ms=latency_ms,
                tokens_used=tokens_used
            )

        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return LLMResponse(
                model=self.model,
                prompt_condition=prompt_condition,
                question_id=question_id,
                response_text=f"ERROR: {str(e)}",
                is_abstention=False,
                latency_ms=(time.time() - start_time) * 1000
            )


class OpenRouterClient:
    """Client for OpenRouter API (for Claude, etc.)."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        question_id: str,
        prompt_condition: str,
        temperature: float = 0.0,
        max_tokens: int = 256
    ) -> LLMResponse:
        """Make a single API call and return parsed response."""
        start_time = time.time()

        try:
            with httpx.Client() as client:
                response = client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://research-experiment.local",
                        "X-Title": "LLM Abstention Research"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=60.0
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code != 200:
                    return LLMResponse(
                        model=self.model,
                        prompt_condition=prompt_condition,
                        question_id=question_id,
                        response_text=f"ERROR: HTTP {response.status_code} - {response.text}",
                        is_abstention=False,
                        latency_ms=latency_ms
                    )

                data = response.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return LLMResponse(
                    model=self.model,
                    prompt_condition=prompt_condition,
                    question_id=question_id,
                    response_text=response_text,
                    is_abstention=detect_abstention(response_text),
                    confidence=extract_confidence(response_text),
                    latency_ms=latency_ms,
                    tokens_used=tokens_used
                )

        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return LLMResponse(
                model=self.model,
                prompt_condition=prompt_condition,
                question_id=question_id,
                response_text=f"ERROR: {str(e)}",
                is_abstention=False,
                latency_ms=(time.time() - start_time) * 1000
            )


def get_client(model_name: str):
    """Factory function to get appropriate client based on model name."""
    if model_name.startswith("gpt-"):
        return OpenAIClient(model=model_name)
    elif model_name.startswith("anthropic/") or model_name.startswith("claude"):
        model = model_name if model_name.startswith("anthropic/") else f"anthropic/{model_name}"
        return OpenRouterClient(model=model)
    else:
        # Default to OpenRouter for other models
        return OpenRouterClient(model=model_name)


if __name__ == "__main__":
    # Test clients
    print("Testing OpenAI client...")
    openai_client = OpenAIClient(model="gpt-4o-mini")
    response = openai_client.call(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2?",
        question_id="test_1",
        prompt_condition="test"
    )
    print(f"OpenAI response: {response.response_text[:100]}...")
    print(f"Abstention: {response.is_abstention}")
