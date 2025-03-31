# app/api/v1/router.py
from fastapi import APIRouter
import logging

from app.api.v1.endpoints import health, analysis

logger = logging.getLogger(__name__)

# API v1 메인 라우터 생성
router = APIRouter()

# 상태 확인 라우터 추가
router.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

# 이미지 분석 라우터 추가
router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}},
)

logger.info("API v1 router configured")