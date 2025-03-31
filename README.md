# AI 이미지 분석 서버

멀티모달 LLM(LLaVA-NeXT)을 활용한 이미지 평가 및 분석 서버입니다.

## 주요 기능

- 이미지 품질 평가 (구도, 선명도, 노이즈, 노출, 색감, 심미성)
- 종합 점수 및 피드백 제공
- AWS S3와 통합된 이미지 처리
- 확장 가능한 REST API
- Docker 기반 배포 지원

## 기술 스택

- **FastAPI**: 고성능 API 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **LLaVA-NeXT**: 멀티모달 대규모 언어 모델
- **Docker & Docker Compose**: 컨테이너화 및 배포
- **Prometheus & Grafana**: 모니터링 및 시각화

## 설치 및 실행

### 필수 요구사항

- Python 3.10 이상
- Docker 및 Docker Compose (선택사항)
- (GPU 사용 시) NVIDIA CUDA 11.7 이상

### 로컬 설치

1. 저장소 복제

```bash
git clone https://github.com/yourusername/ai-image-analysis-server.git
cd ai-image-analysis-server
```

2. 패키지 설치

```bash
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv pillow requests aiohttp boto3 psutil pytest pytest-asyncio transformers accelerate bitsandbytes torch torchvision torchaudio sentencepiece protobuf
```

3. 환경 변수 설정

`.env` 파일을 편집하여 필요한 환경 변수를 설정합니다.

4. 서버 실행

```bash
python -m app.main
```

### Docker 실행

```bash
# 이미지 빌드 및 서버 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f ai-server

# 서버 중지
docker-compose down
```

## API 사용법

### 이미지 분석

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/analyze" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://your-bucket.s3.amazonaws.com/image.jpg"}'
```

### 서버 상태 확인

```bash
curl "http://localhost:8000/api/v1/health/"
```

## 모니터링

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (기본 계정 admin/admin)

## 프로젝트 구조

```
ai-server/
│
├── app/                     # 애플리케이션 코드
│   ├── api/                 # API 관련 코드
│   ├── core/                # 핵심 설정 및 유틸리티
│   ├── models/              # 모델 정의
│   ├── services/            # 비즈니스 로직
│   ├── schemas/             # 스키마 정의
│   └── utils/               # 유틸리티 함수
│
├── models/                  # 모델 파일 저장
├── data/                    # 데이터 저장
├── temp/                    # 임시 파일 저장
├── monitoring/              # 모니터링 설정
├── tests/                   # 테스트 코드
│
├── .env                     # 환경 변수
├── docker-compose.yml       # Docker Compose 설정
└── Dockerfile               # Docker 이미지 빌드 설정
```

## 개발 및 기여

1. 포크 및 클론
2. 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 요청

## 라이선스

MIT License