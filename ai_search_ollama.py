"""
Ollama DeepSeek 기반 AI 검색엔진 프레임워크
===========================================

이 모듈은 Ollama DeepSeek 모델과 Brave Search API를 활용하여
로컬에서 실행되는 인텔리전트한 웹 검색 및 답변 생성 시스템을 구현합니다.
BERT와 FAISS를 활용한 벡터 기반 검색도 지원합니다.
"""

import os
import json
import requests
import asyncio
import httpx
from newspaper import Article
import numpy as np
from transformers import BertTokenizer, BertModel
import faiss
import torch
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과를 나타내는 데이터 클래스"""
    id: int
    title: str
    url: str
    content: str
    description: str = ""
    score: float = 0.0

@dataclass
class VectorSearchResult:
    """벡터 검색 결과를 나타내는 데이터 클래스"""
    document: str
    distance: float
    similarity_score: float = 0.0

class OllamaClient:
    """Ollama 서버와 통신하는 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:14b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
    def is_available(self) -> bool:
        """Ollama 서버 가용성 확인"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """모델 다운로드"""
        try:
            logger.info(f"모델 다운로드 시작: {model_name}")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if data.get('status'):
                        logger.info(f"다운로드 상태: {data['status']}")
                    if data.get('error'):
                        logger.error(f"다운로드 오류: {data['error']}")
                        return False
            
            logger.info(f"모델 다운로드 완료: {model_name}")
            return True
        except Exception as e:
            logger.error(f"모델 다운로드 실패: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """텍스트 생성"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '').strip()
            else:
                logger.error(f"생성 요청 실패: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return ""
    
    async def generate_async(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """비동기 텍스트 생성"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '').strip()
                else:
                    logger.error(f"비동기 생성 요청 실패: {response.status_code}")
                    return ""
                    
        except Exception as e:
            logger.error(f"비동기 텍스트 생성 실패: {e}")
            return ""

class VectorSearchModule:
    """BERT와 FAISS를 활용한 벡터 검색 모듈"""
    
    def __init__(self, model_name='bert-base-uncased'):
        """벡터 검색 모듈 초기화"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="./cache")
            self.model = BertModel.from_pretrained(model_name, cache_dir="./cache")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"벡터 검색 모듈 초기화 완료: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"벡터 검색 모듈 초기화 실패: {e}")
            self.tokenizer = None
            self.model = None

    def text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """텍스트를 벡터로 변환"""
        if not self.tokenizer or not self.model:
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return vector
        except Exception as e:
            logger.error(f"텍스트 벡터화 실패: {e}")
            return None

    def build_index(self, corpus: List[str]) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
        """텍스트 코퍼스에 대해 FAISS 인덱스 구축"""
        if not corpus:
            return None, None
        
        vectors = []
        for i, doc in enumerate(corpus):
            if i % 10 == 0:
                logger.info(f"벡터화 진행: {i+1}/{len(corpus)}")
            vec = self.text_to_vector(doc)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return None, None

        vectors_np = np.vstack(vectors)
        dimension = vectors_np.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors_np)
        
        logger.info(f"FAISS 인덱스 구축 완료: {index.ntotal}개 벡터")
        return index, corpus

    def search(self, query: str, index: faiss.Index, corpus: List[str], k: int = 3) -> List[VectorSearchResult]:
        """벡터 검색 실행"""
        if not index or not corpus:
            return []
        
        query_vector = self.text_to_vector(query)
        if query_vector is None:
            return []
        
        distances, indices = index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if 0 <= idx < len(corpus):
                distance = float(distances[0][i])
                similarity = 1.0 / (1.0 + distance)
                
                results.append(VectorSearchResult(
                    document=corpus[idx],
                    distance=distance,
                    similarity_score=similarity
                ))
        
        return results

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 유사도 계산"""
        vec1 = self.text_to_vector(text1)
        vec2 = self.text_to_vector(text2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # 코사인 유사도 계산
        dot_product = np.dot(vec1.flatten(), vec2.flatten())
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

class OllamaSearchEngine:
    """Ollama DeepSeek 기반 AI 검색엔진 메인 클래스"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", 
                 model: str = "deepseek-r1:14b", 
                 brave_api_key: str = None):
        """검색엔진 초기화"""
        self.ollama_client = OllamaClient(ollama_url, model)
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.vector_module = VectorSearchModule()
        
        # Ollama 서버 확인 및 모델 준비
        self._check_ollama_setup()

    def _check_ollama_setup(self):
        """Ollama 서버 및 모델 설정 확인"""
        if not self.ollama_client.is_available():
            logger.error("Ollama 서버에 연결할 수 없습니다. http://localhost:11434에서 Ollama가 실행 중인지 확인하세요.")
            return
        
        available_models = self.ollama_client.list_models()
        if self.ollama_client.model not in available_models:
            logger.warning(f"모델 {self.ollama_client.model}이 없습니다. 다운로드를 시도합니다...")
            if not self.ollama_client.pull_model(self.ollama_client.model):
                logger.error("모델 다운로드에 실패했습니다.")
                # 대체 모델 시도
                for alt_model in ["deepseek-coder:1.3b", "llama2:7b", "codellama:7b"]:
                    if alt_model in available_models:
                        logger.info(f"대체 모델 사용: {alt_model}")
                        self.ollama_client.model = alt_model
                        break
        else:
            logger.info(f"모델 {self.ollama_client.model} 사용 준비 완료")

    def generate_related_queries(self, query: str) -> List[str]:
        """DeepSeek을 사용하여 관련 쿼리 생성"""
        prompt = f"""다음 질문과 관련된 검색 쿼리 5개를 생성해주세요: "{query}"

각 쿼리는 원래 질문의 다른 측면이나 관련 개념을 다뤄야 합니다.
JSON 배열 형식으로만 응답해주세요.

예시 형식:
["관련 질문 1", "관련 질문 2", "관련 질문 3", "관련 질문 4", "관련 질문 5"]

응답:"""

        try:
            response = self.ollama_client.generate(prompt, max_tokens=300, temperature=0.7)
            
            # JSON 파싱 시도
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                related_queries = json.loads(json_str)
                
                if isinstance(related_queries, list):
                    return [query] + related_queries[:4]  # 원본 포함 최대 5개
            
            logger.warning("관련 쿼리 JSON 파싱 실패, 원본 쿼리만 사용")
            return [query]
            
        except Exception as e:
            logger.error(f"관련 쿼리 생성 실패: {e}")
            return [query]

    def get_search_results(self, query: str) -> List[Dict]:
        """Brave Search API를 사용하여 검색 결과 가져오기"""
        if not self.brave_api_key:
            logger.warning("Brave API 키가 설정되지 않았습니다.")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": 5}
        headers = {"X-Subscription-Token": self.brave_api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("web", {}).get("results", [])
        except Exception as e:
            logger.error(f"검색 결과 가져오기 실패: {e}")
            return []

    def get_url_content(self, url: str) -> str:
        """URL의 콘텐츠 추출"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"URL 콘텐츠 추출 실패 {url}: {e}")
            return ""

    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """DeepSeek을 사용하여 답변 생성"""
        if not search_results:
            return "충분한 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
        
        context_parts = []
        for result in search_results:
            content_snippet = result.content[:500] + "..." if len(result.content) > 500 else result.content
            context_parts.append(f"[{result.id}] {result.title}: {content_snippet}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""다음 검색된 정보를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공해주세요.

검색된 정보:
{context}

사용자 질문: {query}

답변 조건:
1. 제공된 정보만을 바탕으로 답변하세요
2. 정보가 불충분하거나 확실하지 않으면 그렇게 명시하세요  
3. 답변에 사용한 정보의 출처를 [번호] 형식으로 인용하세요
4. 명확하고 구조화된 답변을 제공하세요
5. 한국어로 답변해주세요

답변:"""

        try:
            response = self.ollama_client.generate(prompt, max_tokens=1000, temperature=0.3)
            return response if response else "답변 생성에 실패했습니다."
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

    def search_and_answer(self, query: str) -> Dict[str, Any]:
        """웹 검색 및 DeepSeek 답변 생성 파이프라인"""
        logger.info(f"검색 쿼리: {query}")
        start_time = time.time()
        
        # 1. 관련 쿼리 생성
        logger.info("관련 쿼리 생성 중...")
        related_queries = self.generate_related_queries(query)
        logger.info(f"생성된 관련 쿼리: {related_queries}")
        
        # 2. 검색 실행
        all_results = []
        unique_urls = set()
        
        for search_query in related_queries[:3]:  # 최대 3개 쿼리
            logger.info(f"검색 중: {search_query}")
            results = self.get_search_results(search_query)
            
            for result in results:
                url = result.get("url")
                if url and url not in unique_urls:
                    unique_urls.add(url)
                    logger.info(f"콘텐츠 추출 중: {url}")
                    content = self.get_url_content(url)
                    
                    if content:
                        all_results.append(SearchResult(
                            id=len(all_results) + 1,
                            title=result.get("title", ""),
                            url=url,
                            content=content,
                            description=result.get("description", "")
                        ))
                
                if len(all_results) >= 5:  # 최대 5개 결과
                    break
            
            if len(all_results) >= 5:
                break
        
        logger.info(f"수집된 결과: {len(all_results)}개")
        
        # 3. 답변 생성
        logger.info("답변 생성 중...")
        answer = self.generate_response(query, all_results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "id": r.id,
                    "title": r.title,
                    "url": r.url,
                    "description": r.description
                } for r in all_results
            ],
            "related_queries": related_queries[1:],  # 원본 쿼리 제외
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "model_used": self.ollama_client.model
        }
        
        return result

def setup_ollama():
    """Ollama 설치 및 설정 가이드"""
    print("""
🔧 Ollama 설치 가이드
====================

1. Ollama 설치:
   - macOS: brew install ollama
   - Linux: curl -fsSL https://ollama.com/install.sh | sh
   - Windows: https://ollama.com/download 에서 다운로드

2. Ollama 서버 시작:
   ollama serve

3. DeepSeek 모델 다운로드:
   ollama pull deepseek-coder:6.7b
   
   또는 더 작은 모델:
   ollama pull deepseek-coder:1.3b

4. 모델 확인:
   ollama list

설치 후 이 스크립트를 다시 실행하세요.
""")

def main():
    """메인 함수 - 검색엔진 데모"""
    print("🤖 Ollama DeepSeek 기반 AI 검색엔진")
    print("=" * 50)
    
    # Ollama 클라이언트 테스트
    ollama_client = OllamaClient()
    
    if not ollama_client.is_available():
        print("❌ Ollama 서버에 연결할 수 없습니다.")
        setup_ollama()
        return
    
    print("✅ Ollama 서버 연결 성공")
    
    # 사용 가능한 모델 확인
    models = ollama_client.list_models()
    print(f"📋 사용 가능한 모델: {models}")
    
    # 검색엔진 초기화
    engine = OllamaSearchEngine()
    
    # 웹 검색 데모
    print("\n--- 웹 검색 + DeepSeek 답변 데모 ---")
    if engine.brave_api_key:
        query = input("검색할 질문을 입력하세요: ")
        if query.strip():
            print("\n🔍 검색 및 답변 생성 중...")
            try:
                result = engine.search_and_answer(query)
                
                print(f"\n📝 답변:")
                print("=" * 40)
                print(result['answer'])
                
                print(f"\n📚 출처 ({len(result['sources'])}개):")
                print("-" * 40)
                for source in result['sources'][:3]:
                    print(f"[{source['id']}] {source['title']}")
                    print(f"    {source['url']}")
                
                print(f"\n🔗 관련 쿼리:")
                print("-" * 40)
                for rq in result['related_queries']:
                    print(f"  • {rq}")
                
                print(f"\n⏱️  처리 시간: {result['processing_time']}초")
                print(f"🤖 사용 모델: {result['model_used']}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    else:
        print("⚠️  Brave API 키가 설정되지 않아 웹 검색 데모를 건너뜁니다.")
        print("환경 변수 BRAVE_API_KEY를 설정하거나 .env 파일을 사용하세요.")
    
    # 벡터 검색 데모
    print("\n--- 벡터 검색 데모 ---")
    if engine.vector_module.model:
        sample_corpus = [
            "기후 변화는 지구 온난화로 인한 전 지구적 환경 변화를 의미합니다.",
            "인공지능은 기계가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다.",
            "Python은 데이터 과학과 웹 개발에 널리 사용되는 프로그래밍 언어입니다.",
            "에펠탑은 프랑스 파리에 위치한 철제 격자 구조의 탑입니다.",
            "광합성은 식물이 빛 에너지를 화학 에너지로 변환하는 과정입니다."
        ]
        
        print(f"📊 샘플 코퍼스 ({len(sample_corpus)}개 문서)로 인덱스 구축 중...")
        index, corpus = engine.vector_module.build_index(sample_corpus)
        
        if index:
            vector_query = input("벡터 검색할 질문을 입력하세요: ")
            if vector_query.strip():
                results = engine.vector_module.search(vector_query, index, corpus, k=2)
                print("\n🎯 벡터 검색 결과:")
                print("-" * 40)
                for i, result in enumerate(results, 1):
                    print(f"{i}. 문서: {result.document}")
                    print(f"   유사도: {result.similarity_score:.4f}")
                    print(f"   거리: {result.distance:.4f}\n")
    
    # DeepSeek 직접 테스트
    print("\n--- DeepSeek 모델 직접 테스트 ---")
    test_query = input("DeepSeek에게 직접 질문하세요: ")
    if test_query.strip():
        print("\n🤖 DeepSeek 응답:")
        print("-" * 40)
        response = engine.ollama_client.generate(
            f"다음 질문에 한국어로 간단히 답변해주세요: {test_query}",
            max_tokens=500
        )
        print(response)
    
    print("\n✨ 검색엔진 데모 완료!")

if __name__ == "__main__":
    main() 