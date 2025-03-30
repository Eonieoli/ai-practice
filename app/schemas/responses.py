# app/schemas/responses.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ScoreDetail(BaseModel):
    """
    각 평가 항목의 상세 점수 및 설명
    """
    score: float = Field(..., ge=0, le=10, description="0~10 사이의 점수")
    explanation: str = Field(..., description="점수에 대한 설명")
    improvement_tips: Optional[List[str]] = Field(None, description="개선을 위한 팁")

class ImageAnalysisResponse(BaseModel):
    """
    이미지 분석 응답 스키마
    """
    # 각 평가 항목별 점수 및 설명
    composition: ScoreDetail
    sharpness: ScoreDetail
    noise: ScoreDetail
    exposure: ScoreDetail
    color_harmony: ScoreDetail
    aesthetics: ScoreDetail
    
    # 종합 평가
    overall_score: float = Field(..., ge=0, le=10, description="전체 평가 점수 (0~10)")
    overall_assessment: str = Field(..., description="종합적인 평가 설명")
    
    # 태그 및 추가 정보
    tags: List[str] = Field(..., description="이미지에서 감지된 주요 요소")
    
    # 메타 정보
    processing_time_ms: float = Field(..., description="처리 소요 시간 (밀리초)")