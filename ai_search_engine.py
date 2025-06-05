"""
AI 기반 검색엔진 프레임워크
환경 변수를 통한 설정 지원
"""

import os
import json
import requests
from newspaper import Article
from anthropic import Anthropic
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
class Config:
    """환경 변수에서 설정값 로드"""
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    BRAVE_API_KEY: str = os.environ.get("BRAVE_API_KEY", "")
    VECTOR_MODEL_NAME: str = os.environ.get("VECTOR_MODEL_NAME", "bert-base-uncased")
    HOST_IP: str = os.environ.get("HOST_IP", "localhost")
    HOST_PORT: int = int(os.environ.get("HOST_PORT", "8000"))
    MAX_SEARCH_RESULTS: int = int(os.environ.get("MAX_SEARCH_RESULTS", "10"))
    CACHE_DIR: str = os.environ.get("CACHE_DIR", "./cache")

config = Config()

@dataclass
class SearchResult:
    id: int
    title: str
    url: str
    content: str
    description: str = ""
    score: float = 0.0

@dataclass
class VectorSearchResult:
    document: str
    distance: float
    similarity_score: float = 0.0

class VectorSearchModule:
    def __init__(self, model_name: str = None):
        model_name = model_name or config.VECTOR_MODEL_NAME
        
        # "none"으로 설정시 벡터 검색 비활성화
        if model_name.lower() == "none":
            logger.info("벡터 검색 모듈 비활성화됨")
            self.tokenizer = None
            self.model = None
            return
            
        try:
            logger.info(f"벡터 모델 로딩 시작: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=config.CACHE_DIR)
            self.model = BertModel.from_pretrained(model_name, cache_dir=config.CACHE_DIR)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"벡터 검색 모듈 초기화 완료: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"벡터 검색 모듈 초기화 실패: {e}")
            self.tokenizer = None
            self.model = None

    def text_to_vector(self, text: str) -> Optional[np.ndarray]:
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
        if not corpus:
            return None, None
        
        vectors = []
        valid_docs = []
        
        for i, doc in enumerate(corpus):
            if i % 10 == 0:
                logger.info(f"벡터화 진행: {i+1}/{len(corpus)}")
            
            vec = self.text_to_vector(doc)
            if vec is not None:
                vectors.append(vec)
                valid_docs.append(doc)
        
        if not vectors:
            return None, None

        vectors_np = np.vstack(vectors)
        dimension = vectors_np.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors_np)
        
        logger.info(f"FAISS 인덱스 구축 완료: {index.ntotal}개 벡터")
        return index, valid_docs

    def search(self, query: str, index: faiss.Index, corpus: List[str], k: int = 3) -> List[VectorSearchResult]:
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

class AISearchEngine:
    def __init__(self, anthropic_api_key: str = None, brave_api_key: str = None, vector_model: str = None):
        self.anthropic_api_key = anthropic_api_key or config.ANTHROPIC_API_KEY
        self.brave_api_key = brave_api_key or config.BRAVE_API_KEY
        
        # Anthropic 클라이언트 초기화
        if self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key":
            try:
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"Anthropic 클라이언트 초기화 실패: {e}")
                self.anthropic_client = None
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API 키가 설정되지 않았습니다.")
        
        # 벡터 검색 모듈 초기화 (비동기적으로 처리)
        self.vector_module = None
        try:
            self.vector_module = VectorSearchModule(vector_model)
        except Exception as e:
            logger.warning(f"벡터 검색 모듈 초기화 건너뜀: {e}")

    def generate_related_queries(self, query: str) -> List[str]:
        if not self.anthropic_client:
            logger.warning("Anthropic 클라이언트가 없어 관련 쿼리 생성을 건너뜁니다.")
            return [query]
        
        prompt = f"""다음 쿼리와 관련된 5개의 검색 쿼리를 생성해주세요: "{query}"

관련 쿼리는 원래 질문에 대한 답변을 찾는 데 도움이 되어야 하며, JSON 배열 형식으로 반환해주세요.

예시: ["질문1", "질문2", "질문3", "질문4", "질문5"]

응답:"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # JSON 배열 추출
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                related_queries = json.loads(json_str)
                
                if isinstance(related_queries, list):
                    return [query] + related_queries[:4]  # 원본 포함 최대 5개
            
            logger.warning("관련 쿼리 JSON 파싱 실패")
            return [query]
            
        except Exception as e:
            logger.error(f"관련 쿼리 생성 실패: {e}")
            return [query]

    def get_search_results(self, query: str) -> List[Dict]:
        if not self.brave_api_key:
            logger.warning("Brave API 키가 없어 샘플 데이터를 사용합니다.")
            return [
                {
                    "title": f"'{query}'에 대한 검색 결과 1",
                    "url": "https://example.com/1",
                    "description": f"{query}에 대한 상세한 정보를 제공하는 웹사이트입니다."
                },
                {
                    "title": f"'{query}'에 대한 검색 결과 2", 
                    "url": "https://example.com/2",
                    "description": f"{query} 관련 최신 뉴스와 업데이트를 확인할 수 있습니다."
                }
            ]
        
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": config.MAX_SEARCH_RESULTS}
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
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"URL 콘텐츠 추출 실패 {url}: {e}")
            return ""

    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        if not self.anthropic_client:
            # Anthropic 클라이언트가 없을 때 간단한 요약 제공
            if not search_results:
                return "충분한 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
            
            summary = f"'{query}'에 대한 검색 결과 요약:\n\n"
            for result in search_results[:3]:
                summary += f"• {result.title}: {result.description}\n"
            
            return summary
        
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
3. 답변에 사용한 정보의 출처를 [번호] 형태로 인용하세요
4. 명확하고 구조화된 답변을 제공하세요
5. 한국어로 답변해주세요

답변:"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

    def search_and_answer(self, query: str) -> Dict[str, Any]:
        logger.info(f"검색 쿼리: {query}")
        start_time = time.time()
        
        # 1. 관련 쿼리 생성
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
                    
                    all_results.append(SearchResult(
                        id=len(all_results) + 1,
                        title=result.get("title", ""),
                        url=url,
                        content=content,
                        description=result.get("description", "")
                    ))
                
                if len(all_results) >= config.MAX_SEARCH_RESULTS:
                    break
            
            if len(all_results) >= config.MAX_SEARCH_RESULTS:
                break
        
        logger.info(f"수집된 결과: {len(all_results)}개")
        
        # 3. 답변 생성
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
            "config": {
                "host": f"{config.HOST_IP}:{config.HOST_PORT}",
                "vector_model": config.VECTOR_MODEL_NAME,
                "has_anthropic": bool(self.anthropic_client),
                "has_brave": bool(self.brave_api_key),
                "has_vector_search": bool(self.vector_module and self.vector_module.model)
            }
        }
        
        return result

    def demo_vector_search(self) -> Dict[str, Any]:
        """벡터 검색 데모"""
        if not self.vector_module or not self.vector_module.model:
            return {"error": "벡터 검색 모듈이 사용 불가능합니다."}
        
        # 샘플 문서
        sample_corpus = [
            "인공지능은 기계가 인간의 지능을 모방하여 학습하고 추론하는 기술입니다.",
            "Python은 데이터 과학과 웹 개발에 널리 사용되는 프로그래밍 언어입니다.",
            "기후 변화는 지구 온난화로 인한 전 지구적 환경 변화를 의미합니다.",
            "블록체인은 분산 장부 기술로 암호화폐의 기반이 되는 기술입니다.",
            "양자 컴퓨팅은 양자역학적 현상을 이용하여 정보를 처리하는 새로운 컴퓨팅 패러다임입니다."
        ]
        
        logger.info("벡터 인덱스 구축 중...")
        index, corpus = self.vector_module.build_index(sample_corpus)
        
        if not index:
            return {"error": "벡터 인덱스 구축 실패"}
        
        # 테스트 쿼리
        test_query = "AI와 머신러닝에 대해 알려주세요"
        results = self.vector_module.search(test_query, index, corpus, k=3)
        
        return {
            "query": test_query,
            "corpus_size": len(corpus),
            "results": [
                {
                    "document": r.document,
                    "similarity_score": r.similarity_score,
                    "distance": r.distance
                } for r in results
            ]
        }

def print_config():
    """현재 설정 출력"""
    print("🔧 현재 설정:")
    print("=" * 40)
    print(f"HOST: {config.HOST_IP}:{config.HOST_PORT}")
    print(f"VECTOR_MODEL: {config.VECTOR_MODEL_NAME}")
    print(f"CACHE_DIR: {config.CACHE_DIR}")
    print(f"MAX_SEARCH_RESULTS: {config.MAX_SEARCH_RESULTS}")
    print(f"ANTHROPIC_API_KEY: {'설정됨' if config.ANTHROPIC_API_KEY else '없음'}")
    print(f"BRAVE_API_KEY: {'설정됨' if config.BRAVE_API_KEY else '없음'}")

def main():
    print("🤖 AI 기반 검색엔진 (환경 변수 설정 지원)")
    print("=" * 50)
    
    print_config()
    
    # 검색엔진 초기화
    engine = AISearchEngine()
    
    print("\n--- 기본 정보 ---")
    print(f"✅ 초기화 완료")
    print(f"🔍 Anthropic: {'사용 가능' if engine.anthropic_client else '사용 불가'}")
    print(f"🌐 Brave Search: {'사용 가능' if engine.brave_api_key else '사용 불가 (샘플 데이터 사용)'}")
    print(f"🧠 Vector Search: {'사용 가능' if engine.vector_module and engine.vector_module.model else '사용 불가'}")
    
    # 벡터 검색 데모
    if engine.vector_module and engine.vector_module.model:
        print("\n--- 벡터 검색 데모 ---")
        vector_result = engine.demo_vector_search()
        if "error" not in vector_result:
            print(f"📊 쿼리: {vector_result['query']}")
            print(f"📚 코퍼스 크기: {vector_result['corpus_size']}개 문서")
            print("🎯 상위 결과:")
            for i, result in enumerate(vector_result['results'][:2], 1):
                print(f"  {i}. {result['document'][:50]}...")
                print(f"     유사도: {result['similarity_score']:.4f}")
        else:
            print(f"❌ {vector_result['error']}")
    
    # 검색 및 답변 테스트
    print("\n--- 검색 + 답변 생성 테스트 ---")
    test_query = input("검색할 질문을 입력하세요 (엔터: 기본 질문 사용): ").strip()
    
    if not test_query:
        test_query = "인공지능의 최신 동향은 무엇인가요?"
    
    print(f"\n🔍 검색 쿼리: {test_query}")
    print("검색 및 답변 생성 중...")
    
    try:
        result = engine.search_and_answer(test_query)
        
        print(f"\n📝 답변:")
        print("=" * 40)
        print(result['answer'])
        
        print(f"\n📚 출처 ({len(result['sources'])}개):")
        print("-" * 40)
        for source in result['sources'][:3]:
            print(f"[{source['id']}] {source['title']}")
            print(f"    {source['url']}")
        
        if result['related_queries']:
            print(f"\n🔗 관련 쿼리:")
            print("-" * 40)
            for rq in result['related_queries']:
                print(f"  • {rq}")
        
        print(f"\n⏱️  처리 시간: {result['processing_time']}초")
        print(f"🖥️  호스트: {result['config']['host']}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    print("\n✨ 검색엔진 테스트 완료!")

if __name__ == "__main__":
    main()