"""
Paper Mind - LLM Interface (Step 5)
-----------------------------------
Abstracts LLM inference across:
1. Groq API (Primary cloud LLM)
2. Ollama API (Secondary local LLM - e.g. gemma3, llama3)
3. Deterministic Fallback Heuristic (Clean verbatim evidence with explicit flags)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
import dotenv

dotenv.load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:latest")

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM Client providing traceable LLM completions with transparent fallbacks.
    """
    def __init__(self):
        self.groq_key = GROQ_API_KEY
        self.groq_model = GROQ_MODEL
        self.ollama_url = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL
        self._groq_disabled = False  # Cache invalid key/network failures

    def _call_groq(self, messages: List[Dict[str, str]], temperature: float = 0.1, json_mode: bool = False, max_tokens: int = 512) -> Optional[str]:
        if not self.groq_key or self._groq_disabled:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                elif resp.status_code in (401, 403, 404):
                    logger.warning(f"Groq API returned status {resp.status_code}. Disabling Groq for session.")
                    self._groq_disabled = True
                    return None
                else:
                    return None
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}. Disabling Groq for session.")
            self._groq_disabled = True
            return None

    def _call_ollama(self, messages: List[Dict[str, str]], temperature: float = 0.1, json_mode: bool = False, max_tokens: int = 512) -> Optional[str]:
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if json_mode:
            payload["format"] = "json"

        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(f"{self.ollama_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    return resp.json()["message"]["content"]
                else:
                    logger.warning(f"Ollama returned status {resp.status_code}: {resp.text}")
                    return None
        except Exception as e:
            logger.warning(f"Ollama API call failed: {e}")
            return None

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Executes completion attempting: Groq -> Ollama -> Deterministic Fallback.
        Returns: { "content": str, "backend": str, "fallback_used": bool }
        """
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)

        # 1. Try Groq (Primary Cloud)
        groq_resp = self._call_groq(formatted_messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)
        if groq_resp:
            return {
                "content": groq_resp,
                "backend": f"groq:{self.groq_model}",
                "fallback_used": False
            }

        # 2. Try Ollama (Secondary Local)
        ollama_resp = self._call_ollama(formatted_messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)
        if ollama_resp:
            return {
                "content": ollama_resp,
                "backend": f"ollama:{self.ollama_model}",
                "fallback_used": True
            }

        # 3. Deterministic Fallback Notice
        return {
            "content": "",
            "backend": "none",
            "fallback_used": True,
            "error": "Both Groq and Ollama backends are unavailable."
        }


# Global instance
llm = LLMClient()
