# AI 기반 검색엔진 (환경 변수 지원)

```
    _____ _____   _____                     _       ______             _            
   |  _  |_   _| /  ___|                   | |      |  ___|           (_)           
   | | | | | |   \ `--.  ___  __ _ _ __ ___| |__    | |__ _ __   __ _ _ _ __   ___ 
   | | | | | |    `--. \/ _ \/ _` | '__/ __| '_ \   |  __| '_ \ / _` | | '_ \ / _ \
   | |/ / _| |_  /\__/ /  __/ (_| | | | (__| | | |  | |__| | | | (_| | | | | |  __/
   |___/  \___/  \____/ \___|\__,_|_|  \___|_| |_|  \____/_| |_|\__, |_|_| |_|\___|
                                                                 __/ |            
                                                                |___/             
```

환경 변수를 통한 유연한 설정을 지원하는 AI 기반 검색엔진입니다.

## 🚀 주요 기능

- **환경 변수 설정**: API 키, 모델, 호스트 등 모든 설정을 환경 변수로 관리
- **멀티 AI 지원**: Anthropic Claude + Ollama DeepSeek 모델 지원
- **하이브리드 검색**: 키워드 검색 + 벡터 검색 결합
- **실시간 웹 검색**: Brave Search API 통합
- **모듈화 설계**: 각 구성 요소의 독립적 활성화/비활성화

## 📋 시스템 요구사항

- Python 3.9+
- 8GB RAM (벡터 검색 사용시)
- 2GB 저장공간 (모델 캐시용)

## 🛠️ 설치 및 설정

### 1. 저장소 클론 및 의존성 설치

```bash
git clone <repository-url>
cd AI-based-searchEngine

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

#### 방법 1: 환경 변수 직접 설정

```bash
# API 키 설정
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export BRAVE_API_KEY="your_brave_api_key"

# 모델 설정 (벡터 검색 비활성화시 "none" 설정)
export VECTOR_MODEL_NAME="bert-base-uncased"  # 또는 "none"
export CACHE_DIR="./cache"

# 서버 설정
export HOST_IP="localhost"
export HOST_PORT="8000"

# 검색 설정
export MAX_SEARCH_RESULTS="10"
```

#### 방법 2: 설정 파일 사용

```bash
# config.env 파일 편집
cp config.env config_local.env
nano config_local.env

# 설정 적용
source config_local.env
```

#### 방법 3: Python에서 직접 설정

```python
import os

# 프로그램 실행 전에 설정
os.environ["ANTHROPIC_API_KEY"] = "your_key"
os.environ["BRAVE_API_KEY"] = "your_key"
os.environ["VECTOR_MODEL_NAME"] = "none"  # 벡터 검색 비활성화

from ai_search_engine import AISearchEngine
```

## 🔑 API 키 설정

### Anthropic API 키 (Claude AI)

1. [Anthropic Console](https://console.anthropic.com/)에서 계정 생성
2. API 키 생성 후 `ANTHROPIC_API_KEY` 환경 변수에 설정

### Brave Search API 키

1. [Brave Search API](https://api.search.brave.com/)에서 계정 생성  
2. API 키 생성 후 `BRAVE_API_KEY` 환경 변수에 설정

## 📊 환경 변수 상세 설정

| 변수명 | 설명 | 기본값 | 예시 |
|--------|------|--------|------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API 키 | "" | sk-ant-api03-... |
| `BRAVE_API_KEY` | Brave Search API 키 | "" | BSAxxxxx... |
| `VECTOR_MODEL_NAME` | BERT 벡터 모델명 | bert-base-uncased | bert-base-uncased, none |
| `HOST_IP` | 서버 호스트 IP | localhost | 0.0.0.0, 192.168.1.100 |
| `HOST_PORT` | 서버 포트 | 8000 | 8080, 3000 |
| `MAX_SEARCH_RESULTS` | 최대 검색 결과 수 | 10 | 5, 20 |
| `CACHE_DIR` | 모델 캐시 디렉토리 | ./cache | /tmp/models |

## 🚀 실행 방법

### 1. 기본 실행 (대화형)

```bash
python ai_search_engine.py
```

### 2. 벡터 검색 비활성화 실행

```bash
export VECTOR_MODEL_NAME="none"
python ai_search_engine.py
```

### 3. 프로그래밍 방식 사용

```python
from ai_search_engine import AISearchEngine

# 환경 변수에서 자동 로드
engine = AISearchEngine()

# 또는 직접 설정
engine = AISearchEngine(
    anthropic_api_key="your_key",
    brave_api_key="your_key",
    vector_model="none"  # 벡터 검색 비활성화
)

# 검색 및 답변 생성
result = engine.search_and_answer("인공지능의 최신 동향은?")
print(result['answer'])
```

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Search Engine                        │
├─────────────────────────────────────────────────────────────┤
│  Config (Environment Variables)                            │
│  ├── API Keys (Anthropic, Brave)                          │
│  ├── Model Settings (Vector, Cache)                       │
│  └── Server Settings (Host, Port)                         │
├─────────────────────────────────────────────────────────────┤
│  Core Components                                           │
│  ├── Query Processing & Generation                        │
│  ├── Web Search (Brave API)                              │
│  ├── Content Extraction (newspaper3k)                     │
│  ├── Vector Search (BERT + FAISS) [Optional]             │
│  └── Answer Generation (Claude AI)                        │
├─────────────────────────────────────────────────────────────┤
│  Output                                                    │
│  ├── Structured Answer                                     │
│  ├── Source Citations                                      │
│  ├── Related Queries                                       │
│  └── Performance Metrics                                   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 설정 모드

### 최소 모드 (API 키 없음)
- 벡터 검색: 비활성화
- 웹 검색: 샘플 데이터 사용
- AI 답변: 단순 요약 제공

```bash
export VECTOR_MODEL_NAME="none"
# API 키 설정 안함
python ai_search_engine.py
```

### 기본 모드 (API 키 있음)
- 벡터 검색: 비활성화 
- 웹 검색: Brave API 사용
- AI 답변: Claude AI 사용

```bash
export VECTOR_MODEL_NAME="none"
export ANTHROPIC_API_KEY="your_key"
export BRAVE_API_KEY="your_key"
python ai_search_engine.py
```

### 풀 모드 (모든 기능)
- 벡터 검색: BERT 모델 사용
- 웹 검색: Brave API 사용  
- AI 답변: Claude AI 사용

```bash
export VECTOR_MODEL_NAME="bert-base-uncased"
export ANTHROPIC_API_KEY="your_key"
export BRAVE_API_KEY="your_key"
python ai_search_engine.py
```

## 📈 성능 최적화

### 메모리 사용량 최적화

```bash
# 벡터 검색 비활성화로 메모리 절약
export VECTOR_MODEL_NAME="none"

# 검색 결과 수 제한
export MAX_SEARCH_RESULTS="5"
```

### 캐시 설정

```bash
# SSD에 캐시 저장
export CACHE_DIR="/path/to/fast/storage/cache"

# 캐시 디렉토리 생성
mkdir -p $CACHE_DIR
```

## 🐛 문제 해결

### 1. 벡터 모델 다운로드 오류

```bash
# 벡터 검색 비활성화
export VECTOR_MODEL_NAME="none"
```

### 2. API 키 오류

```bash
# 환경 변수 확인
echo $ANTHROPIC_API_KEY
echo $BRAVE_API_KEY

# 설정 확인
python -c "from ai_search_engine import config; print(config.__dict__)"
```

### 3. 메모리 부족 오류

```bash
# 가벼운 모델 사용
export VECTOR_MODEL_NAME="distilbert-base-uncased"

# 또는 벡터 검색 비활성화
export VECTOR_MODEL_NAME="none"
```

## 📝 예시 코드

### 환경 변수 설정 예시

```bash
#!/bin/bash
# setup_env.sh

export ANTHROPIC_API_KEY="sk-ant-api03-..."
export BRAVE_API_KEY="BSA..."
export VECTOR_MODEL_NAME="none"
export HOST_IP="0.0.0.0"
export HOST_PORT="8080"
export MAX_SEARCH_RESULTS="15"
export CACHE_DIR="./models_cache"

echo "환경 변수 설정 완료"
python ai_search_engine.py
```

### Python 프로그래밍 예시

```python
import os
from ai_search_engine import AISearchEngine

# 환경 변수 설정
os.environ.update({
    "ANTHROPIC_API_KEY": "your_key",
    "BRAVE_API_KEY": "your_key", 
    "VECTOR_MODEL_NAME": "none",
    "MAX_SEARCH_RESULTS": "5"
})

# 검색엔진 초기화
engine = AISearchEngine()

# 검색 및 답변
queries = [
    "2025년 AI 트렌드는?",
    "파이썬 최신 기능은?",
    "기후변화 대응 방안은?"
]

for query in queries:
    result = engine.search_and_answer(query)
    print(f"Q: {query}")
    print(f"A: {result['answer']}")
    print(f"시간: {result['processing_time']}초")
    print("-" * 50)
```

## 🔗 관련 링크

- [Anthropic API 문서](https://docs.anthropic.com/)
- [Brave Search API 문서](https://api.search.brave.com/app/documentation)
- [BERT 모델 허브](https://huggingface.co/models?library=transformers&search=bert)
- [FAISS 문서](https://faiss.ai/)

## 📄 라이선스

MIT License

## 🤝 기여

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

**환경 변수 설정을 통해 유연하고 확장 가능한 AI 검색엔진을 구축하세요!** 🚀 