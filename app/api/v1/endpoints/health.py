# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import time

router = APIRouter()


class HealthResponse(BaseModel):
    """
    헬스 체크 응답 스키마
    """
    status: str
    version: str
    timestamp: float


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    서버 상태 확인 엔드포인트
    """
    from app.core.config import settings
    
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": time.time()
    }