import requests
import json
from typing import Optional, Dict, Any
from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.2:latest",
        temperature: float = 0.7,
        host: str = "localhost",
        port: int = 11434
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"

        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            if model not in available_models:
                print(f"Warning: Model '{model}' not found. Available models: {available_models}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url}: {e}")

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60  # Increase timeout for longer generations
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Ollama response: {e}")

    def generate_with_image(self, prompt: str, image_path: str) -> str:
        raise NotImplementedError("Ollama image generation not implemented yet")
