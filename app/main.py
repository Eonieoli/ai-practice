# app/main.py
from fastapi import FastAPI
from app.api.v1.router import api_router
from app.core.config import settings
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
)

# API 라우터 등록
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """
    앱 시작 시 실행되는 이벤트
    """
    logger.info("Starting AI server...")
    # 이곳에 모델 로딩 등의 초기화 코드를 추가할 수 있습니다

@app.on_event("shutdown")
async def shutdown_event():
    """
    앱 종료 시 실행되는 이벤트
    """
    logger.info("Shutting down AI server...")
    # 이곳에 리소스 정리 코드를 추가할 수 있습니다