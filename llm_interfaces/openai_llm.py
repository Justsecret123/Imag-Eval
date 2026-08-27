from langchain_openai import AzureChatOpenAI
from typing import Optional, Dict, Any
from .base_llm import BaseLLM
from pydantic import SecretStr
import os

class OpenaiLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 1
    ):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("OPENAI_API_VERSION")
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=SecretStr(self.api_key) if self.api_key else None,
            api_version=self.api_version,
            azure_deployment=model,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt"""
        response = self.llm.invoke(prompt)
        return response.content

    def generate_with_image(self, prompt: str, image_path: str) -> str:
        raise NotImplementedError("TODO")
