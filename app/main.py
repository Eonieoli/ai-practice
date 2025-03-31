# app/main.py
import logging
import time
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.v1.router import router as api_v1_router
from app.utils.helper import setup_logging, create_error_response
from app.services.ai_service import get_ai_service

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI 이미지 분석 API - 멀티모달 LLM을 활용한 이미지 평가 서비스",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 처리 시간 미들웨어
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        process_time = time.time() - start_time
        error_response = JSONResponse(
            status_code=500,
            content=create_error_response(
                error_message="Internal server error",
                error_code="SERVER_ERROR",
                details={"error": str(e)}
            )
        )
        error_response.headers["X-Process-Time"] = str(process_time)
        return error_response

# API v1 라우터 등록
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# 루트 경로 처리
@app.get("/")
async def root():
    """API 루트 경로: API 정보 및 상태 반환"""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "멀티모달 LLM을 활용한 이미지 평가 API",
        "status": "active",
        "docs_url": "/docs",
        "api_url": settings.API_V1_STR
    }

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    logger.info("Starting up AI Image Analysis API")
    
    try:
        # AI 서비스 인스턴스 초기화
        ai_service = get_ai_service()
        
        # 모델 사전 로드 (선택적)
        if settings.PRELOAD_MODEL:
            logger.info("Preloading AI model...")
            await asyncio.create_task(ai_service.force_model_reload())
            logger.info("AI model preloaded successfully")
        
        # 임시 파일 클린업
        from app.utils.helper import cleanup_temp_files
        cleanup_temp_files()
        
        logger.info("API startup completed successfully")
    
    except Exception as e:
        logger.error(f"Error during API startup: {str(e)}", exc_info=True)
        # 애플리케이션을 중단하지 않고 오류 로깅만 수행

# 애플리케이션 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트 핸들러"""
    logger.info("Shutting down AI Image Analysis API")
    
    try:
        # AI 서비스 정리
        ai_service = get_ai_service()
        await ai_service.cleanup()
        
        # 임시 파일 정리
        from app.utils.helper import cleanup_temp_files
        cleanup_temp_files(max_age_hours=0)  # 모든 임시 파일 정리
        
        logger.info("API shutdown completed successfully")
    
    except Exception as e:
        logger.error(f"Error during API shutdown: {str(e)}", exc_info=True)

# 개발 환경에서 직접 실행할 때 사용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)