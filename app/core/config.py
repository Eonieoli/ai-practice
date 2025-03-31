# app/core/config.py
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Image Analysis API"
    
    # CORS 설정
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # AI 모델 설정
    MODEL_NAME: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    MODEL_DOWNLOAD_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "downloads")
    TEMP_IMAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "temp")
    
    # 이미지 평가 카테고리
    EVALUATION_CATEGORIES: List[str] = [
        "composition",      # 구도
        "sharpness",        # 선명도
        "noise",            # 노이즈
        "exposure",         # 노출
        "color_harmony",    # 색감
        "aesthetics"        # 심미성
    ]
    
    # 다양한 평가 프롬프트 템플릿
    EVALUATION_PROMPT: str = """
    You are an expert photographer and image analyst. Please evaluate the given image in the following categories, 
    providing a score from 1 to 10 for each category and a brief explanation:
    
    - Composition (구도): How well elements are arranged in the frame
    - Sharpness (선명도): The clarity and detail of the image
    - Noise (노이즈): The presence of unwanted artifacts or grain
    - Exposure (노출): How well-balanced the light is
    - Color Harmony (색감): How well colors complement each other
    - Aesthetics (심미성): Overall visual appeal
    
    Return your evaluation in JSON format with the following structure:
    {
        "composition": {"score": X, "explanation": "..."},
        "sharpness": {"score": X, "explanation": "..."},
        "noise": {"score": X, "explanation": "..."},
        "exposure": {"score": X, "explanation": "..."},
        "color_harmony": {"score": X, "explanation": "..."},
        "aesthetics": {"score": X, "explanation": "..."},
        "overall_score": X,
        "overall_feedback": "..."
    }
    """
    
    # S3 설정 (이미지 다운로드에 필요)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_REGION: Optional[str] = None
    
    # 모델 관련 설정
    DEVICE: str = "cuda"  # 'cuda' 또는 'cpu'
    BATCH_SIZE: int = 1
    MAX_NEW_TOKENS: int = 512
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# 필요한 디렉토리 생성
os.makedirs(settings.MODEL_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(settings.TEMP_IMAGE_DIR, exist_ok=True)