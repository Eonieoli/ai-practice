# tests/test_ai.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.requests import ImageAnalysisRequest
from app.services.ai_service import AIService
from unittest.mock import patch, MagicMock
import json

client = TestClient(app)

# 테스트용 데이터
SAMPLE_IMAGE_URL = "https://example.com/sample.jpg"
SAMPLE_ANALYSIS_RESULT = {
    "composition": {"score": 7.5, "explanation": "좋은 구도", "improvement_tips": ["중앙 배치 개선"]},
    "sharpness": {"score": 8.0, "explanation": "선명한 이미지", "improvement_tips": ["미세 조정 필요"]},
    "noise": {"score": 7.0, "explanation": "약간의 노이즈", "improvement_tips": ["ISO 값 낮추기"]},
    "exposure": {"score": 8.5, "explanation": "적절한 노출", "improvement_tips": ["하이라이트 보존"]},
    "color_harmony": {"score": 9.0, "explanation": "조화로운 색감", "improvement_tips": ["색온도 미세 조정"]},
    "aesthetics": {"score": 8.0, "explanation": "심미적으로 아름다움", "improvement_tips": ["구도 개선"]},
    "overall_score": 8.0,
    "overall_assessment": "전반적으로 훌륭한 사진입니다.",
    "tags": ["자연", "풍경", "하늘"],
    "processing_time_ms": 1234.56
}


def test_health_endpoint():
    """
    헬스 체크 엔드포인트 테스트
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


@patch("app.services.ai_service.AIService.download_image")
@patch("app.services.ai_service.AIService.analyze_image_from_file")
def test_analyze_image_endpoint(mock_analyze, mock_download):
    """
    이미지 분석 엔드포인트 테스트
    """
    # Mock 설정
    mock_download.return_value = "/temp/mock_image.jpg"
    mock_analyze.return_value = SAMPLE_ANALYSIS_RESULT
    
    # 요청 생성
    request_data = {"image_url": SAMPLE_IMAGE_URL}
    
    # 엔드포인트 호출
    response = client.post("/api/v1/analysis", json=request_data)
    
    # 결과 검증
    assert response.status_code == 200
    data = response.json()
    assert data["overall_score"] == SAMPLE_ANALYSIS_RESULT["overall_score"]
    assert len(data["tags"]) == len(SAMPLE_ANALYSIS_RESULT["tags"])
    
    # Mock 함수 호출 검증
    mock_download.assert_called_once_with(SAMPLE_IMAGE_URL)
    mock_analyze.assert_called_once()


@patch("app.services.ai_service.AIService.analyze_image_from_url")
def test_analyze_image_direct_url_endpoint(mock_analyze_url):
    """
    URL 직접 분석 엔드포인트 테스트
    """
    # Mock 설정
    mock_analyze_url.return_value = SAMPLE_ANALYSIS_RESULT
    
    # 요청 생성
    request_data = {"image_url": SAMPLE_IMAGE_URL}
    
    # 엔드포인트 호출
    response = client.post("/api/v1/analysis/url", json=request_data)
    
    # 결과 검증
    assert response.status_code == 200
    data = response.json()
    assert data["overall_score"] == SAMPLE_ANALYSIS_RESULT["overall_score"]
    assert len(data["tags"]) == len(SAMPLE_ANALYSIS_RESULT["tags"])
    
    # Mock 함수 호출 검증
    mock_analyze_url.assert_called_once_with(SAMPLE_IMAGE_URL)


@patch("app.models.ai_model.LlavaNextModel.analyze_image")
def test_ai_service(mock_analyze_image):
    """
    AI 서비스 테스트
    """
    # Mock 설정
    mock_analyze_image.return_value = SAMPLE_ANALYSIS_RESULT
    
    # 서비스 인스턴스 생성
    service = AIService()
    
    # PIL Image Mock 생성
    mock_image = MagicMock()
    
    # 분석 실행
    result = service.model.analyze_image(mock_image)
    
    # 결과 검증
    assert result == SAMPLE_ANALYSIS_RESULT
    mock_analyze_image.assert_called_once_with(mock_image)