"""REST API 모듈.

FastAPI 기반 Q&A API 엔드포인트를 제공한다.
Gradio 웹앱과 함께 또는 독립적으로 실행 가능.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Q&A 요청 스키마."""

    question: str


class AnswerResponse(BaseModel):
    """Q&A 응답 스키마."""

    question: str
    answer: str
    sources: list[str]


def create_api(
    retriever: Any,
    model: Any,
    tokenizer: Any,
) -> FastAPI:
    """FastAPI 앱을 생성한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저

    Returns:
        FastAPI 앱
    """
    from src.model.inference import generate_answer

    app = FastAPI(
        title="충남대학교 학내 정보 Q&A API",
        description="충남대 학사·장학금·취업·컴공과 정보를 RAG 기반으로 답변하는 API",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """서버 상태를 확인한다."""
        return {"status": "ok"}

    @app.post("/api/ask", response_model=AnswerResponse)
    def ask_question(req: QuestionRequest) -> AnswerResponse:
        """질문에 대한 답변을 생성한다.

        Args:
            req: 질문 요청

        Returns:
            답변 응답 (질문, 답변, 출처 URL 목록)
        """
        question = req.question.strip()
        if not question:
            return AnswerResponse(
                question=question,
                answer="질문을 입력해주세요.",
                sources=[],
            )

        context, urls = retriever.build_context(question)

        if not context:
            return AnswerResponse(
                question=question,
                answer="관련 정보를 찾을 수 없습니다.",
                sources=[],
            )

        answer = generate_answer(question, context, urls, model, tokenizer)
        return AnswerResponse(
            question=question,
            answer=answer,
            sources=urls,
        )

    return app
