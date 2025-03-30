# app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리 경로
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    애플리케이션 설정
    """
    # 기본 정보
    PROJECT_NAME: str = "AI Image Analysis Server"
    PROJECT_DESCRIPTION: str = "Analyze images using multi-modal LLaVA-NeXT model"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # 모델 설정
    MODEL_DIR: Path = ROOT_DIR / "models" / "downloads"
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llava-next-vicuna-7b")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cuda")  # "cuda" 또는 "cpu"
    
    # 임시 파일 저장 경로
    TEMP_DIR: Path = ROOT_DIR / "temp"
    
    # AWS S3 설정
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# 설정 인스턴스 생성
settings = Settings()

# 필요한 디렉토리가 없으면 생성
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)