# app/api/v1/endpoints/analysis.py
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional

from app.schemas.requests import ImageAnalysisRequest
from app.schemas.responses import ImageAnalysisResponse, ErrorResponse
from app.services.ai_service import get_ai_service, AIService
from app.utils.helper import create_error_response, is_valid_url, validate_categories, format_exception

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/analyze", 
    response_model=ImageAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def analyze_image(
    request: ImageAnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    이미지 평가 분석 엔드포인트.
    
    - S3 또는 다른 URL의 이미지 분석
    - 구도, 선명도, 노이즈, 노출, 색감, 심미성 등 평가
    - 종합 점수 및 피드백 제공
    
    Parameters:
    - **image_url**: 분석할 이미지의 URL
    - **custom_prompt** (선택 사항): 기본 프롬프트 대신 사용할 사용자 정의 프롬프트
    - **categories** (선택 사항): 평가할 특정 카테고리 목록
    """
    try:
        start_time = time.time()
        logger.info(f"Image analysis requested for URL: {request.image_url}")
        
        # URL 유효성 검사
        if not is_valid_url(str(request.image_url)):
            logger.error(f"Invalid URL provided: {request.image_url}")
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    error_message="Invalid image URL provided",
                    error_code="INVALID_URL",
                    details={"url": str(request.image_url)}
                )
            )
        
        # 카테고리 유효성 검사 (제공된 경우)
        validated_categories = None
        if request.categories:
            validated_categories = validate_categories(request.categories)
            if not validated_categories and request.categories:
                logger.warning(
                    f"None of the provided categories are valid: {request.categories}"
                )
        
        # AI 서비스 인스턴스 가져오기
        service = get_ai_service()
        
        # 이미지 분석 실행
        result = await service.analyze_image(
            image_url=str(request.image_url),
            custom_prompt=request.custom_prompt,
            categories=validated_categories
        )
        
        # 임시 파일 정리를 백그라운드 태스크로 실행
        from app.utils.helper import cleanup_temp_files
        background_tasks.add_task(cleanup_temp_files)
        
        # 처리 시간 기록
        elapsed_time = time.time() - start_time
        logger.info(f"Image analysis completed in {elapsed_time:.2f} seconds")
        
        return result
        
    except HTTPException:
        # 이미 생성된 HTTP 예외는 그대로 전달
        raise
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        
        # 오류 코드 및 상태 코드 결정
        error_code = "ANALYSIS_ERROR"
        status_code = 500
        
        # 오류 유형에 따른 구체적인 오류 코드 및 상태 코드 할당
        error_type = type(e).__name__
        if error_type == "ConnectionError" or error_type == "TimeoutError":
            error_code = "IMAGE_DOWNLOAD_ERROR"
            status_code = 404
        elif error_type == "ValueError" and "Invalid token" in str(e):
            error_code = "MODEL_ERROR"
        elif error_type == "JSONDecodeError":
            error_code = "RESPONSE_PARSE_ERROR"
        
        # HTTP 예외 발생
        raise HTTPException(
            status_code=status_code,
            detail=create_error_response(
                error_message=f"Image analysis failed: {str(e)}",
                error_code=error_code,
                details=format_exception(e)
            )
        )

@router.post(
    "/batch-analyze",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def batch_analyze_images(
    images: List[ImageAnalysisRequest],
    background_tasks: BackgroundTasks
) -> List[Dict[str, Any]]:
    """
    여러 이미지의 일괄 분석 엔드포인트.
    
    - 여러 이미지를 동시에 분석
    - 각 이미지에 대한 분석 결과 목록 반환
    - 일부 이미지 분석 실패 시에도 가능한 결과 반환
    
    참고: 이 엔드포인트는 많은 수의 이미지를 처리할 때 시간이 오래 걸릴 수 있습니다.
    """
    if not images:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                error_message="No images provided for batch analysis",
                error_code="EMPTY_BATCH",
                details={"image_count": 0}
            )
        )
    
    # 최대 배치 크기 제한
    max_batch_size = 10
    if len(images) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                error_message=f"Batch size exceeds maximum limit of {max_batch_size}",
                error_code="BATCH_TOO_LARGE",
                details={"image_count": len(images), "max_allowed": max_batch_size}
            )
        )
    
    logger.info(f"Batch analysis requested for {len(images)} images")
    
    # AI 서비스 인스턴스 가져오기
    service = get_ai_service()
    
    results = []
    for idx, img_request in enumerate(images):
        try:
            # URL 유효성 검사
            if not is_valid_url(str(img_request.image_url)):
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": "Invalid image URL",
                    "code": "INVALID_URL"
                })
                continue
            
            # 카테고리 유효성 검사 (제공된 경우)
            validated_categories = None
            if img_request.categories:
                validated_categories = validate_categories(img_request.categories)
            
            # 이미지 분석 실행
            result = await service.analyze_image(
                image_url=str(img_request.image_url),
                custom_prompt=img_request.custom_prompt,
                categories=validated_categories
            )
            
            # 결과에 인덱스와 상태 추가
            result["index"] = idx
            result["status"] = "success"
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error analyzing image at index {idx}: {str(e)}")
            results.append({
                "index": idx,
                "status": "error",
                "error": str(e),
                "code": "ANALYSIS_ERROR"
            })
    
    # 임시 파일 정리를 백그라운드 태스크로 실행
    from app.utils.helper import cleanup_temp_files
    background_tasks.add_task(cleanup_temp_files)
    
    return results