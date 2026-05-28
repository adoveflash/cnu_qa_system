"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
대화 이력을 유지하는 챗봇 형태로 동작한다.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

CSS = """
.chatbot-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.chatbot-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #003876;
    margin-bottom: 0.25rem;
}
.chatbot-header p {
    font-size: 0.9rem;
    color: #64748b;
}
footer { display: none !important; }
"""


def create_app(
    retriever: Any,
    model: Any,
    tokenizer: Any,
) -> gr.Blocks:
    """Gradio 챗봇 UI를 생성한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저

    Returns:
        Gradio Blocks 앱
    """
    from src.model.inference import generate_answer

    def respond(message: str, history: list[dict[str, str]]) -> str:
        """사용자 메시지에 대한 챗봇 응답을 생성한다."""
        if not message.strip():
            return "질문을 입력해주세요."

        context, urls = retriever.build_context(message)

        if not context:
            return "관련 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요."

        answer = generate_answer(message, context, urls, model, tokenizer)
        return answer

    with gr.Blocks(css=CSS, title="CNU Q&A 챗봇") as app:
        gr.HTML(
            """
            <div class="chatbot-header">
                <h1>충남대학교 학내 정보 Q&A</h1>
                <p>학사 | 장학금 | 취업 | 컴퓨터융합학부</p>
            </div>
            """
        )

        gr.ChatInterface(
            fn=respond,
            type="messages",
            chatbot=gr.Chatbot(
                height=520,
                type="messages",
                show_copy_button=True,
                placeholder="안녕하세요! 충남대학교 학내 정보에 대해 무엇이든 물어보세요.",
            ),
            textbox=gr.Textbox(
                placeholder="질문을 입력하세요...",
                show_label=False,
                container=False,
            ),
            examples=[
                "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
                "수강신청은 언제, 어떻게 하나요?",
                "교내 장학금 종류와 신청 방법을 알려주세요.",
                "인재개발원에서 하는 취업 프로그램이 뭐가 있나요?",
            ],
            submit_btn="전송",
            retry_btn="다시 생성",
            undo_btn="이전으로",
            clear_btn="대화 초기화",
        )

    return app


def launch(
    retriever: Any,
    model: Any,
    tokenizer: Any,
    share: bool = True,
) -> None:
    """Gradio 챗봇 UI + REST API를 함께 실행한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저
        share: 공유 링크 생성 여부
    """
    from src.app.api import create_api

    gradio_app = create_app(retriever, model, tokenizer)
    fastapi_app = create_api(retriever, model, tokenizer)

    gradio_app.launch(share=share, app=fastapi_app)
