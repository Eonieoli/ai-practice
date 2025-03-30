# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import analysis, health

# API 라우터 생성
api_router = APIRouter()

# 각 엔드포인트 라우터 포함
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])