from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # Environment Server Settings (Internal)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 7860
    # The URL where the environment server (FastAPI) is running
    ENV_URL: str = "http://127.0.0.1:7860"
    
    # LLM Settings (Injected by Validator)
    # The validator uses API_BASE_URL, MODEL_NAME, and HF_TOKEN
    API_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME: str = "meta/llama-3.1-8b-instruct"
    HF_TOKEN: str = "dummy"

    @property
    def api_key(self):
        return self.HF_TOKEN
    
    # Environment Settings
    BENCHMARK_NAME: str = "cloud-cost-optimizer"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
