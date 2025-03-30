# app/services/ai_service.py
import os
import time
import logging
import requests
from io import BytesIO
from PIL import Image
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from app.models.ai_model import LlavaNextModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIService:
    """
    AI 분석 서비스
    """
    def __init__(self):
        self.model = LlavaNextModel()
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """
        모델이 로드되어 있는지 확인하고, 로드되지 않았다면 로드
        """
        if not self.model.model_loaded:
            self.model.load_model()
    
    async def download_image(self, image_url: str) -> Optional[Path]:
        """
        URL에서 이미지 다운로드
        
        Args:
            image_url: 이미지 URL
            
        Returns:
            Optional[Path]: 다운로드된 이미지 파일 경로, 실패 시 None
        """
        try:
            logger.info(f"Downloading image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 임시 파일 경로 생성
            temp_file_path = settings.TEMP_DIR / f"{uuid.uuid4()}.jpg"
            
            # 이미지 저장
            with open(temp_file_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Image downloaded to: {temp_file_path}")
            return temp_file_path
        
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return None
    
    def analyze_image_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        URL로부터 이미지를 다운로드하고 분석
        
        Args:
            image_url: 이미지 URL
            
        Returns:
            Dict: 분석 결과
        """
        start_time = time.time()
        
        try:
            # 이미지 다운로드
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # PIL 이미지로 변환
            image = Image.open(BytesIO(response.content))
            
            # 이미지 분석
            analysis_result = self.model.analyze_image(image)
            
            # 처리 시간 추가
            processing_time_ms = (time.time() - start_time) * 1000
            analysis_result["processing_time_ms"] = processing_time_ms
            
            logger.info(f"Image analysis completed in {processing_time_ms:.2f}ms")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            
            # 에러 발생 시 기본 응답 반환
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            return {
                "composition": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "sharpness": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "noise": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "exposure": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "color_harmony": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "aesthetics": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "overall_score": 0.0,
                "overall_assessment": f"분석 중 오류가 발생했습니다: {str(e)}",
                "tags": ["error"],
                "processing_time_ms": processing_time_ms
            }
    
    async def analyze_image_from_file(self, image_path: Path) -> Dict[str, Any]:
        """
        파일 경로로부터 이미지를 로드하고 분석
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict: 분석 결과
        """
        start_time = time.time()
        
        try:
            # 이미지 로드
            image = Image.open(image_path)
            
            # 이미지 분석
            analysis_result = self.model.analyze_image(image)
            
            # 처리 시간 추가
            processing_time_ms = (time.time() - start_time) * 1000
            analysis_result["processing_time_ms"] = processing_time_ms
            
            logger.info(f"Image analysis completed in {processing_time_ms:.2f}ms")
            
            # 임시 파일 삭제
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Temporary file deleted: {image_path}")
            
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            
            # 임시 파일 삭제 시도
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Temporary file deleted: {image_path}")
            except Exception as del_err:
                logger.warning(f"Error deleting temporary file: {str(del_err)}")
            
            # 에러 발생 시 기본 응답 반환
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            return {
                "composition": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "sharpness": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "noise": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "exposure": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "color_harmony": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "aesthetics": {"score": 0.0, "explanation": "분석 실패", "improvement_tips": ["오류로 인해 분석할 수 없습니다"]},
                "overall_score": 0.0,
                "overall_assessment": f"분석 중 오류가 발생했습니다: {str(e)}",
                "tags": ["error"],
                "processing_time_ms": processing_time_ms
            }