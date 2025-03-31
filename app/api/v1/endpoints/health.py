# app/api/v1/endpoints/health.py
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, Optional

from app.schemas.requests import HealthCheckRequest
from app.schemas.responses import HealthCheckResponse, ErrorResponse
from app.services.ai_service import get_ai_service, AIService
from app.utils.helper import create_error_response

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", response_model=HealthCheckResponse, responses={500: {"model": ErrorResponse}})
async def health_check(
    background_tasks: BackgroundTasks,
    check_model: Optional[bool] = Query(False, description="AI 모델 상태를 확인할지 여부")
) -> Dict[str, Any]:
    """
    서버 상태 확인 엔드포인트.
    
    - 시스템 리소스 상태 반환
    - 선택적으로 AI 모델 상태 확인
    """
    try:
        logger.info(f"Health check requested. check_model={check_model}")
        
        # AI 서비스 인스턴스 가져오기
        service = get_ai_service()
        
        # 상태 확인 실행
        health_data = await service.get_system_health(check_model=check_model)
        
        # 임시 파일 정리 태스크 백그라운드로 실행
        if check_model:
            from app.utils.helper import cleanup_temp_files
            background_tasks.add_task(cleanup_temp_files)
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                error_message="Health check failed",
                error_code="HEALTH_CHECK_ERROR",
                details={"error": str(e)}
            )
        )

@router.post("/reload", responses={500: {"model": ErrorResponse}})
async def force_reload_model() -> Dict[str, Any]:
    """
    AI 모델 강제 리로드 엔드포인트.
    
    - 문제 해결이나 모델 업데이트 후 사용
    - 모델을 메모리에서 언로드하고 다시 로드
    """
    try:
        logger.info("Force model reload requested")
        
        # AI 서비스 인스턴스 가져오기
        service = get_ai_service()
        
        # 모델 리로드 실행
        result = await service.force_model_reload()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in force reload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                error_message="Model reload failed",
                error_code="MODEL_RELOAD_ERROR",
                details={"error": str(e)}
            )
        )