# tests/test_ai.py
import os
import sys
import pytest
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
import json
import logging

# 로깅 비활성화
logging.disable(logging.CRITICAL)

# 시스템 경로에 프로젝트 루트 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.models.ai_model import AIImageAnalyzer
from app.services.ai_service import AIService

# 테스트 클라이언트 생성
client = TestClient(app)

class TestAIModel(unittest.TestCase):
    """AI 모델 테스트 클래스"""
    
    def setUp(self):
        """테스트 셋업"""
        # 테스트용 분석 결과
        self.mock_analysis_result = {
            "composition": {"score": 8.5, "explanation": "Good composition with rule of thirds."},
            "sharpness": {"score": 7.2, "explanation": "Generally sharp with some soft areas."},
            "noise": {"score": 9.0, "explanation": "Very low noise levels."},
            "exposure": {"score": 8.0, "explanation": "Well-exposed image with good dynamic range."},
            "color_harmony": {"score": 8.8, "explanation": "Excellent color balance and harmony."},
            "aesthetics": {"score": 8.5, "explanation": "Visually appealing image overall."},
            "overall_score": 8.3,
            "overall_feedback": "This is a high-quality image with good technical aspects.",
            "processing_time": 1.23
        }
    
    @patch('app.models.ai_model.AIImageAnalyzer.load_model')
    @patch('app.models.ai_model.AIImageAnalyzer.download_image')
    @patch('app.models.ai_model.requests.get')
    @patch('app.models.ai_model.AutoProcessor.from_pretrained')
    @patch('app.models.ai_model.AutoModelForCausalLM.from_pretrained')
    def test_analyze_image(self, mock_model, mock_processor, mock_requests_get, mock_download, mock_load):
        """이미지 분석 기능 테스트"""
        # 모의 객체 설정
        mock_model.return_value.generate.return_value = [MagicMock()]
        mock_processor.return_value.decode.return_value = json.dumps(self.mock_analysis_result)
        
        # 테스트 이미지 생성
        test_image = Image.new('RGB', (100, 100), color='red')
        mock_download.return_value = test_image
        
        # 모델 인스턴스 생성 및 분석 실행
        analyzer = AIImageAnalyzer()
        analyzer.processor = mock_processor.return_value
        analyzer.model = mock_model.return_value
        analyzer.is_model_loaded = True
        
        result = analyzer.analyze_image("http://example.com/test.jpg")
        
        # 결과 검증
        self.assertIsNotNone(result)
        self.assertIn("overall_score", result)
        self.assertIn("processing_time", result)

    @patch('app.models.ai_model.AIImageAnalyzer.download_image')
    def test_download_image_exception(self, mock_download):
        """이미지 다운로드 예외 처리 테스트"""
        # 예외 발생 설정
        mock_download.side_effect = Exception("Download failed")
        
        # 모델 인스턴스 생성
        analyzer = AIImageAnalyzer()
        
        # 예외가 제대로 발생하는지 확인
        with self.assertRaises(Exception):
            analyzer.analyze_image("http://example.com/test.jpg")

@pytest.mark.asyncio
class TestAIService:
    """AI 서비스 테스트 클래스"""

    @pytest.fixture
    def mock_ai_service(self):
        """AI 서비스 모의 객체 생성"""
        with patch('app.services.ai_service.get_analyzer') as mock_get_analyzer:
            # 분석기 모의 객체 설정
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = {
                "composition": {"score": 8.5, "explanation": "Good composition."},
                "overall_score": 8.3,
                "processing_time": 1.23
            }
            mock_analyzer.get_model_info.return_value = {
                "model_name": "test-model",
                "is_loaded": True
            }
            mock_get_analyzer.return_value = mock_analyzer
            
            # 서비스 인스턴스 생성
            service = AIService()
            yield service
    
    async def test_analyze_image(self, mock_ai_service):
        """이미지 분석 서비스 테스트"""
        # 분석 실행
        result = await mock_ai_service.analyze_image("http://example.com/test.jpg")
        
        # 결과 검증
        assert "request_id" in result
        assert "overall_score" in result
        assert result["overall_score"] == 8.3
    
    async def test_get_system_health(self, mock_ai_service):
        """시스템 상태 확인 테스트"""
        # 상태 확인 실행
        result = await mock_ai_service.get_system_health(check_model=True)
        
        # 결과 검증
        assert "status" in result
        assert result["status"] == "ok"
        assert "system_info" in result
        assert "model_status" in result

class TestAPIEndpoints:
    """API 엔드포인트 테스트 클래스"""
    
    @patch('app.api.v1.endpoints.analysis.get_ai_service')
    def test_analyze_image_endpoint(self, mock_get_service):
        """이미지 분석 엔드포인트 테스트"""
        # 모의 서비스 설정
        mock_service = AsyncMock()
        mock_service.analyze_image.return_value = {
            "request_id": "test-id",
            "composition": {"score": 8.5, "explanation": "Good composition."},
            "overall_score": 8.3,
            "overall_feedback": "Great image.",
            "processing_time": 1.23,
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_get_service.return_value = mock_service
        
        # API 요청 실행
        response = client.post(
            "/api/v1/analysis/analyze",
            json={"image_url": "https://example.com/test.jpg"}
        )
        
        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "overall_score" in data
        assert data["overall_score"] == 8.3
    
    @patch('app.api.v1.endpoints.health.get_ai_service')
    def test_health_check_endpoint(self, mock_get_service):
        """상태 확인 엔드포인트 테스트"""
        # 모의 서비스 설정
        mock_service = AsyncMock()
        mock_service.get_system_health.return_value = {
            "status": "ok",
            "version": "1.0.0",
            "system_info": {
                "cpu_usage_percent": 5.0,
                "memory_usage_percent": 30.0
            },
            "timestamp": "2023-01-01T00:00:00"
        }
        mock_get_service.return_value = mock_service
        
        # API 요청 실행
        response = client.get("/api/v1/health/")
        
        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "system_info" in data
    
    def test_root_endpoint(self):
        """루트 엔드포인트 테스트"""
        # API 요청 실행
        response = client.get("/")
        
        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "active"

if __name__ == "__main__":
    unittest.main()