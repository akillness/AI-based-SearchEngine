"""
AI 기반 검색엔진 프레임워크
=========================

이 모듈은 Anthropic Claude와 Brave Search API를 활용하여 
인텔리전트한 웹 검색 및 답변 생성 시스템을 구현합니다.
BERT와 FAISS를 활용한 벡터 기반 검색도 지원합니다.
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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과를 나타내는 데이터 클래스"""
    id: int
    title: str
    url: str
    content: str
    description: str = ""

@dataclass
class VectorSearchResult:
    """벡터 검색 결과를 나타내는 데이터 클래스"""
    document: str
    distance: float
    similarity_score: float = 0.0

class VectorSearchModule:
    """BERT와 FAISS를 활용한 벡터 검색 모듈"""
    
    def __init__(self, model_name='bert-base-uncased'):
        """벡터 검색 모듈 초기화"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"벡터 검색 모듈 초기화 완료: {model_name}")
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

class AISearchEngine:
    """AI 기반 검색엔진 메인 클래스"""
    
    def __init__(self, anthropic_api_key: str = None, brave_api_key: str = None):
        """검색엔진 초기화"""
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        
        # Anthropic 클라이언트 초기화
        if self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key":
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            logger.info("Anthropic 클라이언트 초기화 완료")
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API 키가 설정되지 않았습니다.")
        
        # 벡터 검색 모듈 초기화
        self.vector_module = VectorSearchModule()

    def generate_related_queries(self, query: str) -> List[str]:
        """LLM을 사용하여 관련 쿼리 생성"""
        if not self.anthropic_client:
            return []
        
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user", 
                        "content": f"다음 쿼리와 관련된 5개의 검색 쿼리를 생성해주세요: '{query}'\n관련 쿼리는 원래 질문에 대한 답변을 찾는 데 도움이 되어야 하며, JSON 배열 형식으로 반환해주세요."
                    }
                ]
            )
            
            response_text = message.content[0].text
            related_queries = json.loads(response_text)
            return related_queries if isinstance(related_queries, list) else []
        except Exception as e:
            logger.error(f"관련 쿼리 생성 실패: {e}")
            return []

    def get_search_results(self, query: str) -> List[Dict]:
        """Brave Search API를 사용하여 검색 결과 가져오기"""
        if not self.brave_api_key:
            logger.warning("Brave API 키가 설정되지 않았습니다.")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": 3}
        headers = {"X-Subscription-Token": self.brave_api_key}
        
        try:
            response = requests.get(url, params=params, headers=headers)
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
        """LLM을 사용하여 답변 생성"""
        if not self.anthropic_client or not search_results:
            return "충분한 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
        
        context_parts = []
        for result in search_results:
            content_snippet = result.content[:500] + "..." if len(result.content) > 500 else result.content
            context_parts.append(f"[{result.id}] {result.title}: {content_snippet}")
        
        context = "\n".join(context_parts)
        
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요. 답변에는 출처를 [번호] 형식으로 포함해주세요.

컨텍스트:
{context}

질문: {query}

답변:"""
                    }
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."

    def search_and_answer(self, query: str) -> str:
        """웹 검색 및 LLM 답변 생성 파이프라인"""
        logger.info(f"검색 쿼리: {query}")
        
        # 관련 쿼리 생성
        related_queries = self.generate_related_queries(query)
        all_queries = [query] + related_queries
        
        # 검색 실행
        all_results = []
        unique_urls = set()
        
        for search_query in all_queries[:3]:  # 최대 3개 쿼리
            logger.info(f"검색 중: {search_query}")
            results = self.get_search_results(search_query)
            
            for result in results:
                url = result.get("url")
                if url and url not in unique_urls:
                    unique_urls.add(url)
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
        
        # 답변 생성
        return self.generate_response(query, all_results)

def main():
    """메인 함수 - 검색엔진 데모"""
    print("🔍 AI 기반 검색엔진 프레임워크")
    print("=" * 40)
    
    # 검색엔진 초기화
    engine = AISearchEngine()
    
    # 웹 검색 데모
    print("\n--- 웹 검색 + LLM 답변 데모 ---")
    if engine.anthropic_client and engine.brave_api_key:
        query = input("검색할 질문을 입력하세요: ")
        if query:
            print("\n검색 중...")
            answer = engine.search_and_answer(query)
            print(f"\n답변:\n{answer}")
    else:
        print("API 키가 설정되지 않아 웹 검색 데모를 건너뜁니다.")
    
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
        
        print(f"샘플 코퍼스 ({len(sample_corpus)}개 문서)로 인덱스 구축 중...")
        index, corpus = engine.vector_module.build_index(sample_corpus)
        
        if index:
            vector_query = input("벡터 검색할 질문을 입력하세요: ")
            if vector_query:
                results = engine.vector_module.search(vector_query, index, corpus, k=2)
                print("\n벡터 검색 결과:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. 문서: {result.document}")
                    print(f"   유사도: {result.similarity_score:.4f}\n")
    
    print("검색엔진 데모 완료!")

if __name__ == "__main__":
    main() 