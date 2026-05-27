"""Gradio 웹 UI 모듈.

충남대학교 학내 정보 Q&A 시스템 인터페이스.
"""

from __future__ import annotations

from typing import Any

import gradio as gr


def create_app(
    retriever: Any,
    model: Any,
    tokenizer: Any,
) -> gr.Blocks:
    """Gradio UI를 생성한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저

    Returns:
        Gradio Blocks 앱
    """
    from src.model.inference import generate_answer

    def answer_question(question: str) -> str:
        """질문에 대한 답변을 생성한다."""
        if not question.strip():
            return "질문을 입력해주세요."

        context, urls = retriever.build_context(question)

        if not context:
            return "관련 정보를 찾을 수 없습니다."

        answer = generate_answer(question, context, urls, model, tokenizer)
        return answer

    with gr.Blocks(title="충남대학교 학내 정보 Q&A") as app:
        gr.Markdown(
            """
            # 충남대학교 학내 정보 Q&A 시스템
            학사, 장학금, 취업, 컴퓨터융합학부 정보를 질문해보세요.
            """
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="질문",
                placeholder="예: 졸업 요건이 어떻게 되나요?",
                lines=2,
            )

        submit_btn = gr.Button("질문하기", variant="primary")

        answer_output = gr.Textbox(
            label="답변",
            lines=10,
            interactive=False,
        )

        gr.Examples(
            examples=[
                "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
                "수강신청은 언제 하나요?",
                "인재개발원에서 하는 취업 프로그램이 뭐가 있나요?",
                "장학금 신청은 어떻게 하나요?",
            ],
            inputs=question_input,
        )

        submit_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
        question_input.submit(fn=answer_question, inputs=question_input, outputs=answer_output)

    return app


def launch(
    retriever: Any,
    model: Any,
    tokenizer: Any,
    share: bool = True,
) -> None:
    """Gradio UI + REST API를 함께 실행한다.

    Gradio 웹앱과 FastAPI REST API(/api/ask)를 동일 서버에서 제공한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저
        share: 공유 링크 생성 여부
    """
    from src.app.api import create_api

    gradio_app = create_app(retriever, model, tokenizer)
    fastapi_app = create_api(retriever, model, tokenizer)

    # Gradio를 FastAPI에 마운트
    gradio_app.launch(share=share, app=fastapi_app)
