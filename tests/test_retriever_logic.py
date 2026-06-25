"""검색 정밀도 로직 검증 (모델 불필요).

부스팅 source 감지(_detect_sources)와 rerank score 컷오프(_apply_cutoff)는
순수 함수라 bge-m3/reranker 없이 검증할 수 있다.

실행: python tests/test_retriever_logic.py
"""

import sys

sys.path.insert(0, ".")  # noqa: E402

from src.rag.retriever import (  # noqa: E402
    Retriever,
    _detect_sources,
    _RERANK_SCORE_MIN,
)


def test_detect_sources_scholarship() -> None:
    """장학금 질문은 학과 페이지가 아닌 공식 안내(plus_kr) 그룹을 부스팅한다."""
    assert _detect_sources("교내 장학금 신청 자격이 궁금해요") == ["plus_kr", "job", "manual"]
    assert _detect_sources("국가장학생 선발 기준") == ["plus_kr", "job", "manual"]
    print("OK detect_sources(장학) → plus_kr/job/manual")


def test_detect_sources_department() -> None:
    """기존 학과 부스팅은 그대로 동작한다."""
    assert _detect_sources("컴퓨터융합학부 졸업요건") == ["computer", "manual"]
    assert _detect_sources("인재개발원 채용 공지") == ["job"]
    print("OK detect_sources(학과) 회귀 없음")


def test_detect_sources_none() -> None:
    """부스팅 키워드가 없으면 빈 리스트 (셔틀 등은 부스팅 안 함)."""
    assert _detect_sources("셔틀버스 시간표") == []
    print("OK detect_sources(무매칭) → []")


def test_apply_cutoff_drops_irrelevant() -> None:
    """컷오프 미만(무관) 청크는 컨텍스트·출처에서 탈락한다."""
    cands = [
        {"text": "관련", "rerank_score": _RERANK_SCORE_MIN + 1.0},
        {"text": "무관", "rerank_score": _RERANK_SCORE_MIN - 1.0},
    ]
    kept = Retriever._apply_cutoff(cands)
    assert [c["text"] for c in kept] == ["관련"]
    print("OK apply_cutoff: 무관 청크 탈락")


def test_apply_cutoff_all_irrelevant_returns_empty() -> None:
    """전부 무관이면 빈 리스트 → '확인되지 않은 정보' 정직 응답으로 유도."""
    cands = [{"text": "무관", "rerank_score": _RERANK_SCORE_MIN - 5.0}]
    assert Retriever._apply_cutoff(cands) == []
    print("OK apply_cutoff: 전부 무관 → []")


def test_apply_cutoff_no_score_keeps() -> None:
    """rerank_score가 없는(naive/hybrid) 결과는 컷오프하지 않는다."""
    cands = [{"text": "점수없음"}]
    assert Retriever._apply_cutoff(cands) == cands
    print("OK apply_cutoff: score 없으면 보존")


if __name__ == "__main__":
    test_detect_sources_scholarship()
    test_detect_sources_department()
    test_detect_sources_none()
    test_apply_cutoff_drops_irrelevant()
    test_apply_cutoff_all_irrelevant_returns_empty()
    test_apply_cutoff_no_score_keeps()
    print("\n=== 검색 정밀도 로직 검증 통과 ===")
