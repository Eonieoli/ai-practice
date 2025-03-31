# scripts/preload_models.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모델 사전 다운로드 스크립트
AI 서버 실행 전에 필요한 모델 파일을 미리 다운로드합니다.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# 설정 로드
from app.core.config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('model-downloader')

def download_model(model_name=None, force=False):
    """
    Hugging Face 모델 다운로드
    
    Args:
        model_name: 다운로드할 모델 이름
        force: 이미 다운로드된 모델도 강제로 다시 다운로드할지 여부
    """
    start_time = time.time()
    model_name = model_name or settings.MODEL_NAME
    
    logger.info(f"모델 다운로드 시작: {model_name}")
    
    try:
        # 모델 다운로드 경로 설정
        os.makedirs(settings.MODEL_DOWNLOAD_PATH, exist_ok=True)
        
        # 모델 프로세서 다운로드
        from transformers import AutoProcessor
        logger.info(f"프로세서 다운로드 중: {model_name}")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=settings.MODEL_DOWNLOAD_PATH,
            local_files_only=not force
        )
        logger.info("프로세서 다운로드 완료")
        
        # 모델 다운로드
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        logger.info(f"모델 다운로드 중: {model_name}")
        
        # 양자화 설정 (선택적)
        use_quantization = False
        if use_quantization:
            import torch
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=settings.MODEL_DOWNLOAD_PATH,
            local_files_only=not force,
            quantization_config=bnb_config if use_quantization else None,
        )
        logger.info("모델 다운로드 완료")
        
        # 모델 크기 확인
        model_size_params = sum(p.numel() for p in model.parameters())
        logger.info(f"모델 크기: {model_size_params:,} 파라미터")
        
        # 모델 파일 경로 확인
        model_path = os.path.join(settings.MODEL_DOWNLOAD_PATH, model_name.split('/')[-1])
        if os.path.exists(model_path):
            logger.info(f"모델 파일 경로: {model_path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"모델 다운로드 완료. 소요 시간: {elapsed_time:.2f}초")
        
        return True
    
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Hugging Face 모델 다운로더')
    parser.add_argument('--model', type=str, default=None, help='다운로드할 모델 이름')
    parser.add_argument('--force', action='store_true', help='이미 다운로드된 모델도 강제로 다시 다운로드')
    args = parser.parse_args()
    
    download_model(args.model, args.force)

if __name__ == "__main__":
    main()