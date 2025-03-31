# app/schemas/responses.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class CategoryEvaluation(BaseModel):
    """단일 카테고리 평가 결과"""
    
    score: float = Field(
        ...,
        description="1부터 10까지의 점수",
        ge=1.0,
        le=10.0
    )
    
    explanation: str = Field(
        ...,
        description="평가에 대한 설명"
    )

class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    
    request_id: str = Field(..., description="요청 식별자")
    
    # 각 카테고리별 평가 결과
    composition: Optional[CategoryEvaluation] = Field(None, description="구도 평가")
    sharpness: Optional[CategoryEvaluation] = Field(None, description="선명도 평가")
    noise: Optional[CategoryEvaluation] = Field(None, description="노이즈 평가")
    exposure: Optional[CategoryEvaluation] = Field(None, description="노출 평가")
    color_harmony: Optional[CategoryEvaluation] = Field(None, description="색감 평가")
    aesthetics: Optional[CategoryEvaluation] = Field(None, description="심미성 평가")
    
    # 종합 평가
    overall_score: float = Field(
        ...,
        description="종합 점수 (1-10)",
        ge=1.0,
        le=10.0
    )
    
    overall_feedback: str = Field(..., description="종합 피드백")
    
    # 메타데이터
    processing_time: float = Field(..., description="처리 시간 (초)")
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 생성 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "f58d7a6c-c7c0-4b7c-8f0f-3c9d4c8f0abc",
                "composition": {
                    "score": 8.5,
                    "explanation": "이미지는 3분할 법칙을 잘 따르고 있으며, 주요 피사체가 잘 배치되어 있습니다."
                },
                "sharpness": {
                    "score": 7.8,
                    "explanation": "대부분의 이미지가 선명하지만, 일부 영역에서 약간의 초점 흐림이 있습니다."
                },
                "noise": {
                    "score": 9.0,
                    "explanation": "노이즈가 매우 적으며, 깨끗한 이미지 품질을 보여줍니다."
                },
                "exposure": {
                    "score": 8.2,
                    "explanation": "전반적으로 노출이 좋으나, 일부 하이라이트 영역에서 약간의 과노출이 있습니다."
                },
                "color_harmony": {
                    "score": 9.5,
                    "explanation": "색상 조합이 조화롭고 시각적으로 매력적입니다."
                },
                "aesthetics": {
                    "score": 8.8,
                    "explanation": "전체적으로 아름다운 이미지이며, 시각적 매력이 높습니다."
                },
                "overall_score": 8.6,
                "overall_feedback": "이 이미지는 특히 색감과 노이즈 측면에서 뛰어나며, 전체적으로 잘 균형잡힌 사진입니다. 약간의 노출 조정과 선명도 향상으로 더 개선될 수 있습니다.",
                "processing_time": 1.24,
                "timestamp": "2023-10-15T14:23:45.123456"
            }
        }

class ErrorResponse(BaseModel):
    """오류 응답 스키마"""
    
    error: str = Field(..., description="오류 메시지")
    code: str = Field(..., description="오류 코드")
    details: Optional[Dict[str, Any]] = Field(None, description="추가 오류 세부 정보")
    timestamp: datetime = Field(default_factory=datetime.now, description="오류 발생 시간")

class HealthCheckResponse(BaseModel):
    """상태 확인 응답 스키마"""
    
    status: str = Field(..., description="서비스 상태 (ok, error)")
    version: str = Field(..., description="API 버전")
    
    # 추가 상태 정보
    system_info: Dict[str, Any] = Field(
        ..., 
        description="시스템 정보 (메모리 사용량, CPU 사용량 등)"
    )
    
    model_status: Optional[Dict[str, Any]] = Field(
        None,
        description="AI 모델 상태 정보 (check_model이 True인 경우만 포함)"
    )
    
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 생성 시간")