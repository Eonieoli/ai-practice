# app/utils/helper.py
import os
import json
import time
import logging
import urllib.parse
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import tempfile
import aiohttp
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = None) -> None:
    """
    애플리케이션의 로깅 설정을 구성합니다.
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, (log_level or settings.LOG_LEVEL).upper())
    
    # 기본 로거 설정
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 외부 라이브러리 로거 레벨 조정
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")

def create_error_response(error_message: str, error_code: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    표준화된 오류 응답 생성
    
    Args:
        error_message: 오류 메시지
        error_code: 오류 코드
        details: 추가 오류 세부 정보
        
    Returns:
        오류 응답 딕셔너리
    """
    return {
        "error": error_message,
        "code": error_code,
        "details": details,
        "timestamp": datetime.now()
    }

def is_valid_url(url: str) -> bool:
    """
    URL이 유효한지 확인
    
    Args:
        url: 확인할 URL 문자열
        
    Returns:
        URL이 유효하면 True, 그렇지 않으면 False
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_categories(categories: List[str]) -> List[str]:
    """
    유효한 평가 카테고리 목록만 필터링
    
    Args:
        categories: 확인할 카테고리 목록
        
    Returns:
        유효한 카테고리만 포함된 목록
    """
    valid_categories = []
    
    for category in categories:
        # 카테고리 이름 정규화 (공백 제거, 소문자 변환, 언더스코어로 치환)
        normalized = re.sub(r'\s+', '_', category.strip().lower())
        
        if normalized in settings.EVALUATION_CATEGORIES:
            valid_categories.append(normalized)
        else:
            logger.warning(f"Invalid category provided: {category}")
    
    return valid_categories

async def download_file_to_temp(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    URL에서 파일을 다운로드하여 임시 파일로 저장
    
    Args:
        url: 다운로드할 파일의 URL
        timeout: 다운로드 타임아웃 (초)
        
    Returns:
        (임시 파일 경로, 파일 확장자) 튜플
    """
    try:
        # URL에서 파일 확장자 추출 시도
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1]
        
        if not ext:
            ext = '.jpg'  # 기본 확장자
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
            dir=settings.TEMP_IMAGE_DIR
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # 파일 다운로드
        logger.info(f"Downloading file from {url} to {temp_path}")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download file: HTTP {response.status}")
                
                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
        
        download_time = time.time() - start_time
        logger.info(f"Download completed in {download_time:.2f} seconds")
        
        return temp_path, ext
        
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

async def get_s3_file_url(bucket: str, file_key: str, expires_in: int = 3600) -> str:
    """
    S3 파일에 대한 서명된 URL 생성
    
    Args:
        bucket: S3 버킷 이름
        file_key: S3 파일 키
        expires_in: URL 만료 시간 (초)
        
    Returns:
        서명된 URL 문자열
    """
    try:
        # AWS 설정 확인
        if not all([settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, settings.AWS_S3_REGION]):
            raise ValueError("AWS credentials not configured")
        
        # S3 클라이언트 생성
        s3_client = boto3.client(
            's3',
            region_name=settings.AWS_S3_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        
        # 서명된 URL 생성
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': file_key},
            ExpiresIn=expires_in
        )
        
        return url
        
    except Exception as e:
        logger.error(f"Error generating S3 signed URL: {str(e)}")
        raise

def cleanup_temp_files(max_age_hours: int = 24) -> int:
    """
    오래된 임시 파일 정리
    
    Args:
        max_age_hours: 파일을 보존할 최대 시간 (시간)
        
    Returns:
        삭제된 파일 수
    """
    try:
        temp_dir = settings.TEMP_IMAGE_DIR
        if not os.path.exists(temp_dir):
            logger.warning(f"Temp directory does not exist: {temp_dir}")
            return 0
        
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            # 파일인지 확인하고 마지막 수정 시간 가져오기
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                
                # 파일이 너무 오래된 경우 삭제
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting temp file {file_path}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} temporary files")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")
        return 0

def format_exception(e: Exception) -> Dict[str, Any]:
    """
    예외를 구조화된 딕셔너리로 포맷팅
    
    Args:
        e: 예외 객체
        
    Returns:
        예외 정보를 담은 딕셔너리
    """
    return {
        "type": e.__class__.__name__,
        "message": str(e),
        "details": {
            "module": e.__class__.__module__,
            "args": [str(arg) for arg in e.args]
        }
    }