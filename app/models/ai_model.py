# app/models/ai_model.py
import os
import logging
import torch
from pathlib import Path
from app.core.config import settings
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import json

logger = logging.getLogger(__name__)

class LlavaNextModel:
    """
    LLaVA-NeXT 모델 래퍼 클래스
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = settings.MODEL_DEVICE
        self.model_loaded = False
        
    def load_model(self) -> None:
        """
        LLaVA-NeXT 모델 로드
        """
        try:
            logger.info(f"Loading LLaVA-NeXT model on {self.device}...")
            
            # 모델이 GPU를 사용할 수 있는지 확인
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA is not available. Switching to CPU.")
                self.device = "cpu"
            
            # transformers에서 모델 및 프로세서 로드
            from transformers import AutoProcessor, LlavaNextForConditionalGeneration
            
            model_name_or_path = settings.MODEL_NAME
            
            # 로컬 경로에 모델이 있는지 확인
            local_model_path = settings.MODEL_DIR / model_name_or_path
            model_path = str(local_model_path) if local_model_path.exists() else model_name_or_path
            
            # 모델 및 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            
            self.model_loaded = True
            logger.info("LLaVA-NeXT model loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        이미지 분석 및 평가 수행
        
        Args:
            image: PIL Image 객체
            
        Returns:
            Dict: 분석 결과
        """
        if not self.model_loaded:
            self.load_model()
        
        # 평가 기준 프롬프트 생성
        prompt = """
        당신은 사진의 전문적인 평가자입니다. 다음 6가지 측면에서 이 사진을 분석하고 평가해주세요:
        1. 구도(composition): 프레이밍, 구성 요소의 배치, 시각적 균형 등을 평가
        2. 선명도(sharpness): 이미지의 초점과 세부 묘사의 선명함을 평가
        3. 노이즈(noise): 이미지의 입자감이나 디지털 노이즈 수준을 평가
        4. 노출(exposure): 밝기, 대비, 하이라이트와 그림자의 디테일을 평가
        5. 색감(color harmony): 색상의 조화, 채도, 색온도의 적절성을 평가
        6. 심미성(aesthetics): 전반적인 시각적 매력과 예술적 가치를 평가
        
        각 항목에 대해 0-10점 척도로 점수를 매기고, 왜 그런 점수를 주었는지 설명해주세요.
        또한 각 항목별로 개선할 수 있는 팁을 제공해주세요.
        
        마지막으로 전체 평가 점수와 종합적인 평가를 제공하고, 이미지에서 감지된 주요 요소들을 태그로 알려주세요.
        
        JSON 형식으로 응답해주세요. 다음과 같은 구조여야 합니다:
        {
            "composition": {"score": float, "explanation": string, "improvement_tips": [string]},
            "sharpness": {"score": float, "explanation": string, "improvement_tips": [string]},
            "noise": {"score": float, "explanation": string, "improvement_tips": [string]},
            "exposure": {"score": float, "explanation": string, "improvement_tips": [string]},
            "color_harmony": {"score": float, "explanation": string, "improvement_tips": [string]},
            "aesthetics": {"score": float, "explanation": string, "improvement_tips": [string]},
            "overall_score": float,
            "overall_assessment": string,
            "tags": [string]
        }
        """
        
        try:
            # 이미지 및 프롬프트 처리
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # 추론 실행
            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
            
            # 출력 디코딩
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # JSON 추출 (모델이 프롬프트와 함께 응답할 수 있으므로 JSON 부분만 추출)
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = generated_text[json_start:json_end]
                try:
                    analysis_result = json.loads(json_str)
                    return analysis_result
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 텍스트 응답을 구조화된 형식으로 변환 시도
                    logger.warning("Failed to parse JSON from model output, attempting to structure it manually")
                    return self._structure_text_response(generated_text)
            else:
                logger.warning("No JSON found in model output, attempting to structure it manually")
                return self._structure_text_response(generated_text)
                
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise
    
    def _structure_text_response(self, text: str) -> Dict[str, Any]:
        """
        모델이 JSON 형식으로 응답하지 않을 경우 텍스트 응답을 구조화
        
        Args:
            text: 모델 출력 텍스트
            
        Returns:
            Dict: 구조화된 결과
        """
        # 기본 응답 템플릿
        result = {
            "composition": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "sharpness": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "noise": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "exposure": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "color_harmony": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "aesthetics": {"score": 5.0, "explanation": "분석 실패", "improvement_tips": ["자동 분석 실패"]},
            "overall_score": 5.0,
            "overall_assessment": "자동 분석에 실패했습니다. 모델 응답을 구조화할 수 없습니다.",
            "tags": ["분석실패"]
        }
        
        # 텍스트에서 각 항목에 대한 정보 추출 시도
        categories = ["composition", "sharpness", "noise", "exposure", "color_harmony", "aesthetics"]
        
        for category in categories:
            # 카테고리 관련 텍스트 찾기
            cat_index = text.lower().find(category.lower())
            if cat_index != -1:
                # 다음 카테고리 시작 위치 또는 텍스트 끝
                next_cat_index = len(text)
                for next_cat in categories:
                    if next_cat != category:
                        next_idx = text.lower().find(next_cat.lower(), cat_index + len(category))
                        if next_idx != -1 and next_idx < next_cat_index:
                            next_cat_index = next_idx
                
                # 카테고리 관련 텍스트 추출
                category_text = text[cat_index:next_cat_index].strip()
                
                # 점수 추출 시도
                import re
                score_match = re.search(r'(\d+(\.\d+)?)/10|score:?\s*(\d+(\.\d+)?)', category_text, re.IGNORECASE)
                if score_match:
                    score_str = score_match.group(1) if score_match.group(1) else score_match.group(3)
                    try:
                        result[category]["score"] = float(score_str)
                    except (ValueError, TypeError):
                        pass
                
                # 설명 추출
                explanation_text = category_text.replace(category, '', 1).strip()
                if explanation_text:
                    result[category]["explanation"] = explanation_text[:200]  # 너무 길지 않게 자름
        
        # 전체 평가 점수 추출 시도
        overall_match = re.search(r'overall.*?(\d+(\.\d+)?)/10|overall.*?score:?\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
        if overall_match:
            overall_str = overall_match.group(1) if overall_match.group(1) else overall_match.group(3)
            try:
                result["overall_score"] = float(overall_str)
            except (ValueError, TypeError):
                pass
        
        # 전체 평가 텍스트 추출 시도
        overall_idx = text.lower().find('overall')
        if overall_idx != -1:
            result["overall_assessment"] = text[overall_idx:overall_idx+200].strip()
        
        return result