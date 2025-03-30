# app/utils/helper.py
import os
import uuid
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_s3_client():
    """
    AWS S3 클라이언트 생성
    
    Returns:
        boto3.client: S3 클라이언트
    """
    return boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

def download_from_s3(bucket_name: str, object_key: str, local_path: Optional[str] = None) -> Optional[str]:
    """
    S3에서 파일 다운로드
    
    Args:
        bucket_name: S3 버킷 이름
        object_key: 객체 키
        local_path: 로컬 저장 경로 (없으면 임시 파일 생성)
        
    Returns:
        Optional[str]: 다운로드된 파일 경로, 실패 시 None
    """
    if not local_path:
        local_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4()}-{os.path.basename(object_key)}")
    
    try:
        s3_client = get_s3_client()
        s3_client.download_file(bucket_name, object_key, local_path)
        logger.info(f"Downloaded {bucket_name}/{object_key} to {local_path}")
        return local_path
    except ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        return None

def upload_to_s3(local_path: str, bucket_name: str, object_key: Optional[str] = None) -> Optional[str]:
    """
    S3에 파일 업로드
    
    Args:
        local_path: 로컬 파일 경로
        bucket_name: S3 버킷 이름
        object_key: 객체 키 (없으면 파일명 사용)
        
    Returns:
        Optional[str]: 업로드된 객체의 URL, 실패 시 None
    """
    if not object_key:
        object_key = os.path.basename(local_path)
    
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(local_path, bucket_name, object_key)
        
        # 객체 URL 생성
        object_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{object_key}"
        logger.info(f"Uploaded {local_path} to {object_url}")
        
        return object_url
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None

def clean_temp_files(age_hours: int = 24) -> int:
    """
    오래된 임시 파일 정리
    
    Args:
        age_hours: 이 시간(시간) 이상 지난 파일 삭제
        
    Returns:
        int: 삭제된 파일 수
    """
    import time
    from pathlib import Path
    
    temp_dir = Path(settings.TEMP_DIR)
    current_time = time.time()
    age_seconds = age_hours * 3600
    deleted_count = 0
    
    try:
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > age_seconds:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old temp file: {file_path}")
        
        logger.info(f"Cleaned {deleted_count} temporary files older than {age_hours} hours")
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}")
        return -1