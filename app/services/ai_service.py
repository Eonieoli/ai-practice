# app/services/ai_service.py
import os
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
import json
import psutil
from datetime import datetime

from app.core.config import settings
from app.models.ai_model import get_analyzer

logger = logging.getLogger(__name__)

class AIService:
    """
    AI 서비스 클래스: AI 모델과 API 엔드포인트 간의 중간 계층
    비즈니스 로직, 캐싱, 로깅, 오류 처리 등을 담당
    """
    
    def __init__(self):
        """AIService 초기화"""
        self.analyzer = get_analyzer()
        self._request_counter = 0
        self._cache = {}  # 간단한 메모리 내 캐시
        logger.info("AIService initialized")
    
    async def analyze_image(
        self, 
        image_url: str, 
        custom_prompt: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        이미지 분석 서비스
        
        Args:
            image_url: 분석할 이미지의 URL
            custom_prompt: 사용자 정의 프롬프트 (선택 사항)
            categories: 평가할 카테고리 목록 (선택 사항)
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 요청 ID 생성
            request_id = str(uuid.uuid4())
            self._request_counter += 1
            
            logger.info(f"Image analysis request received. ID: {request_id}, URL: {image_url}")
            
            # 동일한 이미지 + 프롬프트 조합에 대한 캐싱 키 생성
            cache_key = f"{image_url}_{custom_prompt}_{','.join(categories) if categories else 'all'}"
            
            # 캐시 확인 (선택적으로 비활성화 가능)
            if cache_key in self._cache and settings.ENABLE_CACHING:
                logger.info(f"Cache hit for request {request_id}")
                cached_result = self._cache[cache_key].copy()
                cached_result["request_id"] = request_id
                cached_result["cached"] = True
                return cached_result
            
            # AI 모델 분석 실행
            result = self.analyzer.analyze_image(
                image_url=image_url,
                custom_prompt=custom_prompt,
                categories=categories
            )
            
            # 결과에 요청 ID 추가
            result["request_id"] = request_id
            
            # 결과 캐싱 (메모리 관리를 위해 캐시 크기 제한)
            if settings.ENABLE_CACHING:
                if len(self._cache) >= 100:  # 최대 100개 항목 캐싱
                    # 가장 오래된 항목 제거
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                
                self._cache[cache_key] = result.copy()
            
            logger.info(f"Image analysis completed successfully for request {request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_image service: {str(e)}", exc_info=True)
            raise
    
    async def get_system_health(self, check_model: bool = False) -> Dict[str, Any]:
        """
        시스템 상태 정보 제공
        
        Args:
            check_model: AI 모델 로드 상태도 확인할지 여부
            
        Returns:
            시스템 및 모델 상태 정보
        """
        try:
            logger.info("Health check requested")
            
            # 기본 시스템 정보 수집
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            health_info = {
                "status": "ok",
                "version": "1.0.0",  # API 버전
                "system_info": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "uptime_seconds": time.time() - psutil.boot_time(),
                    "request_count": self._request_counter
                },
                "timestamp": datetime.now()
            }
            
            # AI 모델 상태 확인 (요청 시)
            if check_model:
                logger.info("Model status check requested")
                
                model_info = self.analyzer.get_model_info()
                health_info["model_status"] = model_info
                
                # 모델 로드 확인
                if not self.analyzer.is_model_loaded:
                    try:
                        logger.info("Loading model for health check")
                        self.analyzer.load_model()
                        health_info["model_status"]["loaded_for_check"] = True
                    except Exception as model_error:
                        logger.error(f"Error loading model: {str(model_error)}")
                        health_info["status"] = "error"
                        health_info["model_status"]["load_error"] = str(model_error)
            
            logger.info("Health check completed")
            return health_info
            
        except Exception as e:
            logger.error(f"Error in get_system_health service: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "version": "1.0.0",
                "system_info": {
                    "error": str(e)
                },
                "timestamp": datetime.now()
            }
    
    async def force_model_reload(self) -> Dict[str, Any]:
        """
        AI 모델 강제 리로드
        
        Returns:
            리로드 결과 정보
        """
        try:
            logger.info("Force model reload requested")
            
            start_time = time.time()
            
            # 기존 모델 언로드
            if self.analyzer.is_model_loaded:
                self.analyzer.unload_model()
            
            # 모델 새로 로드
            self.analyzer.load_model(force_reload=True)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Force model reload completed in {elapsed_time:.2f} seconds")
            
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "reload_time_seconds": elapsed_time,
                "model_info": self.analyzer.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Error in force_model_reload service: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error reloading model: {str(e)}",
                "error_details": str(e)
            }
    
    async def cleanup(self) -> None:
        """
        서비스 종료 시 정리 작업
        """
        try:
            logger.info("Performing service cleanup")
            
            # 캐시 정리
            self._cache.clear()
            
            # 모델 언로드
            if self.analyzer.is_model_loaded:
                self.analyzer.unload_model()
            
            logger.info("Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during service cleanup: {str(e)}", exc_info=True)


# 싱글톤 인스턴스
_service_instance = None

def get_ai_service() -> AIService:
    """싱글톤 AIService 인스턴스 반환"""
    global _service_instance
    if _service_instance is None:
        _service_instance = AIService()
    return _service_instance