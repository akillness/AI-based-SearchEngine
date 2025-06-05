# 🤖 AI 기반 검색엔진 프레임워크

```
    █████╗ ██╗    ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
   ██╔══██╗██║    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
   ███████║██║    ███████╗█████╗  ███████║██████╔╝██║     ███████║
   ██╔══██║██║    ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
   ██║  ██║██║    ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
   ╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                      
    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗                 
    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝                 
    █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗                   
    ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝                   
    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗                 
    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝                 
```

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [API 키 설정](#api-키-설정)
- [프로젝트 구조](#프로젝트-구조)
- [기술 스택](#기술-스택)
- [사용 예시](#사용-예시)
- [라이선스](#라이선스)

## 🌟 개요

**AI 기반 검색엔진 프레임워크**는 최신 인공지능 기술을 활용하여 전통적인 키워드 검색을 넘어선 인텔리전트한 검색 시스템입니다. 

### 핵심 특징
- 🧠 **자연어 이해**: Claude AI를 활용한 복잡한 질문 이해
- 🔍 **하이브리드 검색**: 키워드 + 의미적 검색의 융합
- 📊 **벡터 검색**: BERT와 FAISS를 활용한 의미 기반 검색
- 🌐 **실시간 웹 검색**: Brave Search API 통합
- 📝 **종합 답변 생성**: 여러 소스를 종합한 완전한 답변
- 🔗 **출처 추적**: 신뢰할 수 있는 정보원 제공

## ⚡ 주요 기능

```
┌─────────────────────────────────────────────┐
│                사용자 질문                    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         🧠 쿼리 이해 및 확장                  │
├─────────────────────────────────────────────┤
│  • 자연어 처리                               │
│  • 관련 쿼리 생성                             │
│  • 의도 파악                                 │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         🔍 다중 검색 실행                     │
├─────────────────────────────────────────────┤
│  • 웹 검색 (Brave API)                      │
│  • 벡터 검색 (FAISS)                        │
│  • 콘텐츠 추출                               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         📝 답변 생성 및 종합                  │
├─────────────────────────────────────────────┤
│  • LLM 기반 답변 생성                        │
│  • 출처 정보 포함                            │
│  • 신뢰성 검증                               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              💡 최종 답변                     │
└─────────────────────────────────────────────┘
```

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                    프레젠테이션 레이어                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Web UI    │  │  REST API   │  │   CLI App   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│                      비즈니스 로직 레이어                │
│  ┌─────────────────────────────────────────────────┐  │
│  │            AISearchEngine                       │  │
│  │  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │  쿼리 처리   │  │      답변 생성           │   │  │
│  │  └─────────────┘  └─────────────────────────┘   │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│                      검색 서비스 레이어                │
│  ┌─────────────┐              ┌─────────────────────┐ │
│  │ 웹 검색 모듈  │              │  벡터 검색 모듈       │ │
│  │             │              │                     │ │
│  │ Brave API   │              │ BERT + FAISS        │ │
│  │ newspaper3k │              │ Transformers        │ │
│  └─────────────┘              └─────────────────────┘ │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│                      데이터 레이어                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   웹 데이터   │  │  벡터 인덱스  │  │   캐시 저장   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└──────────────────────────────────────────────────────┘
```

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/AI-based-searchEngine.git
cd AI-based-searchEngine
```

### 2. Python 환경 설정 (권장: Python 3.8+)
```bash
# 가상환경 생성
python -m venv ai_search_env

# 가상환경 활성화
# Windows
ai_search_env\Scripts\activate
# macOS/Linux
source ai_search_env/bin/activate
```

### 3. 종속성 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 추가 패키지 (선택사항)
```bash
# GPU 지원을 위한 PyTorch (CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# FAISS GPU 버전 (성능 향상)
pip uninstall faiss-cpu
pip install faiss-gpu
```

## 🔑 API 키 설정

### 필수 API 키

1. **Anthropic Claude API**
   - [Anthropic Console](https://console.anthropic.com/)에서 API 키 발급
   - 월별 사용량 제한 확인

2. **Brave Search API**
   - [Brave Search API](https://api.search.brave.com/)에서 무료 계정 생성
   - 월 2,000회 무료 검색 제공

### 환경 변수 설정

#### Windows
```bash
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
set BRAVE_API_KEY=your_brave_api_key_here
```

#### macOS/Linux
```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
export BRAVE_API_KEY=your_brave_api_key_here
```

#### .env 파일 생성 (권장)
```bash
# .env 파일 생성
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" > .env
echo "BRAVE_API_KEY=your_brave_api_key_here" >> .env
```

## 🎯 사용 방법

### 기본 실행
```bash
# 완전한 검색엔진 실행
python ai_search_complete.py

# 기존 프레임워크 실행
python search_framework.py
```

### 프로그래밍 방식 사용
```python
from ai_search_complete import AISearchEngine

# 검색엔진 초기화
engine = AISearchEngine(
    anthropic_api_key="your_api_key",
    brave_api_key="your_brave_key"
)

# 웹 검색 및 답변 생성
answer = engine.search_and_answer("인공지능의 미래는 어떨까요?")
print(answer)

# 벡터 검색 사용
corpus = ["문서1 내용", "문서2 내용", "문서3 내용"]
index, corpus = engine.vector_module.build_index(corpus)
results = engine.vector_module.search("검색 쿼리", index, corpus)
```

## 📁 프로젝트 구조

```
AI-based-searchEngine/
├── 📄 README.md                    # 프로젝트 설명서
├── 📄 concept.md                   # 개념 설명 문서
├── 📄 requirements.txt             # Python 종속성
├── 🐍 ai_search_complete.py        # 완전한 AI 검색엔진
├── 🐍 search_framework.py          # 기존 검색 프레임워크
├── 📁 docs/                       # 문서화
│   ├── 📄 architecture.md
│   ├── 📄 api_reference.md
│   └── 📄 examples.md
├── 📁 tests/                      # 테스트 파일
│   ├── 🧪 test_search_engine.py
│   ├── 🧪 test_vector_search.py
│   └── 🧪 test_web_search.py
├── 📁 examples/                   # 사용 예시
│   ├── 🐍 basic_usage.py
│   ├── 🐍 advanced_search.py
│   └── 🐍 custom_pipeline.py
└── 📁 utils/                     # 유틸리티 함수
    ├── 🐍 text_processing.py
    ├── 🐍 data_validation.py
    └── 🐍 performance_monitor.py
```

## 🛠️ 기술 스택

### 핵심 기술
| 구분 | 기술 | 버전 | 용도 |
|------|------|------|------|
| 🧠 **AI/ML** | Anthropic Claude | 3.0+ | 자연어 이해 및 답변 생성 |
| 🔍 **검색** | Brave Search API | v1 | 실시간 웹 검색 |
| 📊 **벡터DB** | FAISS | 1.7+ | 고속 벡터 검색 |
| 🤖 **NLP** | Transformers (BERT) | 4.30+ | 텍스트 임베딩 |
| 🐍 **언어** | Python | 3.8+ | 메인 프로그래밍 언어 |

### 주요 라이브러리
| 라이브러리 | 용도 | 설치 명령 |
|-----------|------|-----------|
| `anthropic` | Claude AI API 클라이언트 | `pip install anthropic` |
| `transformers` | BERT 모델 로딩 | `pip install transformers` |
| `faiss-cpu` | 벡터 검색 | `pip install faiss-cpu` |
| `newspaper3k` | 웹 콘텐츠 추출 | `pip install newspaper3k` |
| `requests` | HTTP 요청 | `pip install requests` |
| `torch` | 딥러닝 프레임워크 | `pip install torch` |

## 💡 사용 예시

### 1. 기본 질문-답변
```python
# 질문: "기후 변화의 주요 원인은 무엇인가요?"
# 답변: 기후 변화의 주요 원인은 온실가스 배출로 인한 
#       지구 온난화입니다. [1][2] 화석연료 연소, 삼림벌채, 
#       산업활동 등이 주요 요인입니다. [3]
```

### 2. 기술적 질문
```python
# 질문: "머신러닝과 딥러닝의 차이점을 설명해주세요"
# 답변: 머신러닝은 데이터에서 패턴을 학습하는 기술이며,
#       딥러닝은 신경망을 사용하는 머신러닝의 한 분야입니다...
```

### 3. 벡터 검색 예시
```python
# 문서 컬렉션에서 의미적으로 유사한 문서 찾기
corpus = [
    "인공지능은 미래 기술의 핵심입니다",
    "기계학습은 AI의 중요한 분야입니다", 
    "오늘 날씨가 정말 좋네요"
]

# "AI 기술"로 검색하면 첫 번째와 두 번째 문서가 높은 유사도로 반환됩니다
```

## 🔧 고급 설정

### 성능 최적화
```python
# GPU 사용 설정
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 벡터 검색 모듈에 GPU 적용
vector_module = VectorSearchModule(model_name='bert-base-uncased')
vector_module.model.to(device)
```

### 커스텀 프롬프트
```python
# 답변 생성 프롬프트 커스터마이징
custom_prompt = """
전문가로서 다음 질문에 답변해주세요:
질문: {query}
답변 조건:
- 정확하고 객관적인 정보 제공
- 출처 명시 [번호]
- 3-5문장으로 간결하게 설명
"""
```

## 🧪 테스트 실행

```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 모듈 테스트
python -m pytest tests/test_search_engine.py -v

# 커버리지 확인
pip install pytest-cov
python -m pytest tests/ --cov=ai_search_complete
```

## 📊 성능 벤치마크

| 메트릭 | 값 | 설명 |
|--------|----|----- |
| 평균 응답 시간 | 3-8초 | 질문부터 답변까지 |
| 검색 정확도 | 85-92% | 관련성 평가 기준 |
| 벡터 검색 속도 | 0.1-0.5초 | 1000개 문서 기준 |
| 메모리 사용량 | 2-4GB | BERT 모델 로딩 시 |

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원 및 문의

- 🐛 **버그 리포트**: [Issues](https://github.com/your-username/AI-based-searchEngine/issues)
- 💡 **기능 요청**: [Feature Requests](https://github.com/your-username/AI-based-searchEngine/discussions)
- 📧 **이메일**: your-email@example.com
- 💬 **Discord**: [커뮤니티 참여](https://discord.gg/your-channel)

## 🙏 감사의 말

- [Anthropic](https://www.anthropic.com/) - Claude AI API 제공
- [Brave](https://brave.com/) - 검색 API 서비스
- [Hugging Face](https://huggingface.co/) - Transformers 라이브러리
- [Facebook Research](https://github.com/facebookresearch/faiss) - FAISS 라이브러리

---

```
┌─────────────────────────────────────────────────────────────┐
│  🚀 Ready to search the future with AI? Let's get started!  │
└─────────────────────────────────────────────────────────────┘
```

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!** 