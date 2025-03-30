# app/api/v1/endpoints/analysis.py
from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import Dict, Any

from app.schemas.requests import ImageAnalysisRequest
from app.schemas.responses import ImageAnalysisResponse
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """
    이미지 분석 엔드포인트
    
    Args:
        request: 이미지 URL을 포함한 요청
        
    Returns:
        ImageAnalysisResponse: 이미지 분석 결과
    """
    try:
        # AIService 인스턴스 생성
        ai_service = AIService()
        
        # URL로부터 이미지 다운로드 및 분석
        downloaded_image_path = await ai_service.download_image(str(request.image_url))
        
        if not downloaded_image_path:
            raise HTTPException(status_code=400, detail="이미지를 다운로드할 수 없습니다")
        
        # 이미지 분석
        analysis_result = await ai_service.analyze_image_from_file(downloaded_image_path)
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Error processing image analysis request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류가 발생했습니다: {str(e)}")


@router.post("/url", response_model=ImageAnalysisResponse)
async def analyze_image_direct_url(request: ImageAnalysisRequest):
    """
    이미지 URL 직접 분석 엔드포인트
    - 이미지를 임시 파일로 저장하지 않고 직접 분석
    
    Args:
        request: 이미지 URL을 포함한 요청
        
    Returns:
        ImageAnalysisResponse: 이미지 분석 결과
    """
    try:
        # AIService 인스턴스 생성
        ai_service = AIService()
        
        # URL로부터 이미지 직접 분석
        analysis_result = ai_service.analyze_image_from_url(str(request.image_url))
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Error processing direct image analysis request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류가 발생했습니다: {str(e)}")