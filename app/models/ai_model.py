# app/models/ai_model.py
import os
import json
import time
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from PIL import Image
import requests
from io import BytesIO
import re
import numpy as np

# 멀티모달 LLM 관련 라이브러리
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

from app.core.config import settings

logger = logging.getLogger(__name__)

class AIImageAnalyzer:
    """
    LLaVA-NeXT 멀티모달 LLM을 사용한 이미지 분석 클래스
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        AIImageAnalyzer 클래스 초기화
        
        Args:
            model_name: 사용할 모델 이름
            device: 추론에 사용할 장치 ('cuda' 또는 'cpu')
        """
        self.model_name = model_name or settings.MODEL_NAME
        self.device_name = device or settings.DEVICE
        self.device = torch.device(self.device_name if torch.cuda.is_available() and self.device_name == 'cuda' else 'cpu')
        
        # 모델과 프로세서 초기화
        self.model = None
        self.processor = None
        
        # 모델 로드 상태 추적
        self.is_model_loaded = False
        self.last_load_time = None
        
        logger.info(f"AIImageAnalyzer initialized. Model: {self.model_name}, Device: {self.device}")
    
    def load_model(self, force_reload: bool = False) -> None:
        """
        모델 및 프로세서 로드
        
        Args:
            force_reload: 이미 로드된 모델이라도 강제로 다시 로드할지 여부
        """
        if self.is_model_loaded and not force_reload:
            logger.info("Model already loaded. Skipping.")
            return
        
        logger.info(f"Loading model {self.model_name}...")
        load_start_time = time.time()
        
        try:
            # 4비트 양자화 설정 (VRAM 사용량 감소)
            if self.device_name == 'cuda':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device_name == 'cuda' else None,
                device_map="auto" if self.device_name == 'cuda' else None,
                quantization_config=bnb_config if self.device_name == 'cuda' else None,
            )
            
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # 모델 로드 상태 업데이트
            self.is_model_loaded = True
            self.last_load_time = time.time()
            
            load_time = time.time() - load_start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def download_image(self, image_url: str) -> Image.Image:
        """
        URL에서 이미지 다운로드
        
        Args:
            image_url: 다운로드할 이미지 URL
            
        Returns:
            PIL Image 객체
        """
        try:
            logger.info(f"Downloading image from {image_url}")
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            # 이미지 로드
            img = Image.open(BytesIO(response.content))
            
            # RGBA 이미지를 RGB로 변환 (알파 채널 제거)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 알파 채널을 마스크로 사용
                img = background
            
            logger.info(f"Image downloaded successfully. Size: {img.size}, Mode: {img.mode}")
            return img
            
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise
    
    def analyze_image(
        self, 
        image_url: str, 
        custom_prompt: Optional[str] = None, 
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        이미지 분석 실행
        
        Args:
            image_url: 분석할 이미지 URL
            custom_prompt: 사용자 정의 프롬프트 (선택 사항)
            categories: 평가할 카테고리 목록 (선택 사항)
            
        Returns:
            분석 결과 딕셔너리
        """
        # 모델이 로드되지 않았다면 로드
        if not self.is_model_loaded:
            self.load_model()
        
        # 시작 시간 기록
        start_time = time.time()
        
        try:
            # 이미지 다운로드
            image = self.download_image(image_url)
            
            # 평가 프롬프트 준비
            prompt = custom_prompt or settings.EVALUATION_PROMPT
            
            # 특정 카테고리만 평가하도록 프롬프트 수정
            if categories:
                valid_categories = [cat for cat in categories if cat in settings.EVALUATION_CATEGORIES]
                
                if not valid_categories:
                    logger.warning(f"No valid categories provided. Using all categories.")
                else:
                    # 프롬프트에서 선택한 카테고리만 언급하도록 수정
                    category_text = "\n".join([
                        f"- {cat.replace('_', ' ').title()}: " + 
                        next((line.split(':')[1].strip() for line in prompt.split('\n') 
                             if line.strip().startswith(f"- {cat.replace('_', ' ').title()}")), "")
                        for cat in valid_categories
                    ])
                    
                    # 새 프롬프트 생성
                    prompt_parts = prompt.split("Return your evaluation")
                    if len(prompt_parts) >= 2:
                        base_prompt = prompt_parts[0]
                        json_instructions = prompt_parts[1]
                        
                        # 새 카테고리 목록으로 프롬프트 재구성
                        new_prompt = (
                            f"{base_prompt.split('in the following categories')[0]} "
                            f"in the following categories, providing a score from 1 to 10 for each category "
                            f"and a brief explanation:\n\n{category_text}\n\n"
                            f"Return your evaluation{json_instructions}"
                        )
                        prompt = new_prompt
            
            logger.info(f"Prepared evaluation prompt with {len(prompt)} characters")
            
            # 입력 생성
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": settings.MAX_NEW_TOKENS,
                "do_sample": False,  # greedy decoding
                "temperature": 0.1,  # 결정적인 출력을 위해 낮은 온도 사용
            }
            
            # 추론 실행
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 출력 디코딩
            decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
            
            # 프롬프트 제거하여 응답만 추출
            response = decoded_output[len(prompt):]
            
            # JSON 데이터 추출 시도
            try:
                # JSON 형식 문자열 찾기
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    # JSON을 찾지 못한 경우 텍스트 파싱 시도
                    logger.warning("JSON not found in model output. Attempting to parse manually.")
                    result = self._parse_evaluation_text(response)
            except Exception as json_error:
                logger.error(f"Error parsing JSON from model output: {str(json_error)}")
                logger.info(f"Raw model output: {response}")
                result = self._parse_evaluation_text(response)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            logger.info(f"Image analysis completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
    
    def _parse_evaluation_text(self, text: str) -> Dict[str, Any]:
        """
        모델이 JSON을 반환하지 않을 경우 텍스트에서 평가 결과 파싱
        
        Args:
            text: 모델이 생성한 텍스트 응답
            
        Returns:
            파싱된 평가 결과 딕셔너리
        """
        result = {}
        categories = settings.EVALUATION_CATEGORIES
        scores = []
        
        for category in categories:
            # 카테고리 이름 변환 (snake_case -> Title Case)
            category_title = category.replace('_', ' ').title()
            
            # 정규식으로 카테고리 점수 찾기 시도
            score_pattern = rf'{category_title}[^\d]*(\d+(?:\.\d+)?)'
            score_match = re.search(score_pattern, text, re.IGNORECASE)
            
            # 설명 찾기 시도
            explanation_pattern = rf'{category_title}[^\d]*\d+(?:\.\d+)?[^\n\.]*(.*?)(?=\n\n|\n[A-Z]|$)'
            explanation_match = re.search(explanation_pattern, text, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                score = float(score_match.group(1))
                scores.append(score)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."
                
                result[category] = {
                    "score": min(max(score, 1.0), 10.0),  # 1-10 범위로 제한
                    "explanation": explanation
                }
            else:
                logger.warning(f"Could not find score for {category}")
        
        # 전체 점수와 피드백
        if scores:
            result["overall_score"] = round(sum(scores) / len(scores), 1)
        else:
            result["overall_score"] = 5.0
        
        # 전체 피드백 찾기 시도
        feedback_patterns = [
            r"(?:Overall|Summary|Conclusion)(?:[^\.]*)\.(.*?)(?=\n\n|\n[A-Z]|$)",
            r"(?:종합|요약|결론)(?:[^\.]*)\.(.*?)(?=\n\n|\n[가-힣]|$)"
        ]
        
        for pattern in feedback_patterns:
            feedback_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if feedback_match:
                result["overall_feedback"] = feedback_match.group(1).strip()
                break
        else:
            # 패턴 일치 항목을 찾지 못한 경우 마지막 단락 사용
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            result["overall_feedback"] = paragraphs[-1] if paragraphs else "No overall feedback provided."
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 상태 및 정보를 포함하는 딕셔너리
        """
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_loaded": self.is_model_loaded,
            "last_load_time": self.last_load_time
        }
        
        # 모델이 로드된 경우 추가 정보 포함
        if self.is_model_loaded and self.model:
            # 모델 메모리 사용량 추정
            if self.device_name == 'cuda' and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB 단위
                info["memory"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2)
                }
            
            # 모델 크기 (파라미터 수)
            param_count = sum(p.numel() for p in self.model.parameters())
            info["parameters"] = param_count
            info["parameters_millions"] = round(param_count / 1_000_000, 2)
        
        return info
    
    def unload_model(self) -> None:
        """모델 언로드 (메모리 해제)"""
        if self.is_model_loaded:
            logger.info("Unloading model from memory")
            
            # 모델과 프로세서 참조 제거
            del self.model
            del self.processor
            
            # CUDA 캐시 정리 (CUDA 사용 시)
            if self.device_name == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 상태 업데이트
            self.model = None
            self.processor = None
            self.is_model_loaded = False
            
            logger.info("Model unloaded successfully")


# 싱글톤 인스턴스
_analyzer_instance = None

def get_analyzer() -> AIImageAnalyzer:
    """싱글톤 AIImageAnalyzer 인스턴스 반환"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AIImageAnalyzer()
    return _analyzer_instance