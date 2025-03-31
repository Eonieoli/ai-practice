# app/schemas/requests.py
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl

class ImageAnalysisRequest(BaseModel):
    """이미지 분석 요청 스키마"""
    
    image_url: HttpUrl = Field(
        ..., 
        description="S3에 업로드된 이미지의 URL"
    )
    
    custom_prompt: Optional[str] = Field(
        None,
        description="기본 프롬프트 대신 사용할 사용자 정의 프롬프트 (선택 사항)"
    )
    
    categories: Optional[List[str]] = Field(
        None,
        description="평가할 특정 카테고리 목록 (선택 사항, 제공되지 않을 경우 모든 카테고리 평가)"
    )

class HealthCheckRequest(BaseModel):
    """상태 확인 요청 스키마"""
    
    check_model: bool = Field(
        False,
        description="AI 모델 로드 상태를 확인할지 여부"
    )