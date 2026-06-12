"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스 (블루 테마).
Gradio 4.x/5.x (tuples) 및 6.x (messages) 자동 호환.

설계 원칙: 메시지/말풍선 레이아웃 CSS는 건드리지 않는다.
(width·display·word-break 등을 덮어쓰면 한글이 세로로 깨지므로
 Gradio 기본 렌더링을 그대로 쓰고, 색은 테마로만 입힌다.)
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream

_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_USE_MESSAGES = _GRADIO_MAJOR >= 5
_NEED_TYPE_PARAM = _GRADIO_MAJOR == 5  # 6.x는 type 파라미터 없음

# 헤더·푸터·타이핑 애니메이션만 스타일링 (메시지 레이아웃은 절대 미터치)
_CSS = """
footer { visibility: hidden; }
.title-area { text-align: center; padding: 24px 0 4px 0; }
.title-area .logo-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 58px; height: 58px; border-radius: 16px;
    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
    font-size: 1.7em; margin-bottom: 10px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.32);
}
.title-area h1 { font-size: 1.9em; font-weight: 800; color: #0B2F5E; margin-bottom: 4px; }
.title-area p { color: #5C6B84; font-size: 0.95em; margin: 0; }
.category-badges {
    display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin: 14px 0 18px 0;
}
.category-badges span {
    background: #fff; color: #1D4ED8; padding: 6px 14px; border-radius: 999px;
    font-size: 0.82em; font-weight: 600; border: 1px solid #D4E1F4;
}
/* 타이핑 점 애니메이션 (메시지 레이아웃과 무관) */
.typing-dots { display: inline-flex; gap: 5px; align-items: center; }
.typing-dots i {
    width: 8px; height: 8px; border-radius: 50%; background: #2563EB; font-style: normal;
    animation: td-bounce 1.1s infinite ease-in-out;
}
.typing-dots i:nth-child(2) { animation-delay: 0.18s; }
.typing-dots i:nth-child(3) { animation-delay: 0.36s; }
@keyframes td-bounce {
    0%, 70%, 100% { transform: translateY(0); opacity: 0.45; }
    35% { transform: translateY(-6px); opacity: 1; }
}
"""

# 생성 대기 표시(타이핑 애니메이션)
_TYPING = '<span class="typing-dots"><i></i><i></i><i></i></span>'

_EXAMPLES = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "이번 학기 수강신청은 언제야?",
    "오늘 기숙사 점심 메뉴 뭐야?",
    "셔틀버스 시간표 알려줘",
    "최근 공지사항 보여줘",
]

_WELCOME = (
    "안녕하세요! 충남대학교 학내 정보를 안내하는 AI 챗봇이에요. 😊\n\n"
    "졸업요건 · 학사일정 · 공지사항 · 식단 · 셔틀버스 등 "
    "궁금한 걸 편하게 물어봐 주세요."
)


def _initial_history() -> list:
    """Gradio 버전에 맞는 초기 환영 메시지 이력을 만든다."""
    if _USE_MESSAGES:
        return [{"role": "assistant", "content": _WELCOME}]
    return [[None, _WELCOME]]


def _extract_text(content: Any) -> str:
    """Gradio content에서 순수 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("value", content.get("text", str(content)))
    return str(content)


def _set_bot(history: list, text: str) -> None:
    """버전에 맞게 마지막 어시스턴트 메시지 내용을 설정한다."""
    if _USE_MESSAGES:
        history[-1]["content"] = text
    else:
        history[-1][1] = text


def create_app(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> gr.Blocks:
    """Gradio 챗봇 UI를 생성한다 (블루 테마)."""

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue", neutral_hue="slate")

    with gr.Blocks(css=_CSS, theme=theme, title="CNU Q&A 챗봇") as app:
        gr.HTML(
            """
            <div class="title-area">
                <div class="logo-icon">🎓</div>
                <h1>CNU Campus AI</h1>
                <p>충남대학교 학내 정보를 안내해 드릴게요</p>
            </div>
            <div class="category-badges">
                <span>🎯 졸업요건</span>
                <span>📢 공지사항</span>
                <span>📅 학사일정</span>
                <span>🍽️ 식단</span>
                <span>🚌 셔틀버스</span>
            </div>
            """
        )

        chatbot_kwargs = {"height": 480, "show_label": False}
        if _NEED_TYPE_PARAM:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(**chatbot_kwargs, value=_initial_history())

        with gr.Row():
            msg = gr.Textbox(
                placeholder="궁금한 점을 물어보세요! (예: 졸업요건, 수강신청, 식단 등)",
                show_label=False,
                scale=9,
                container=False,
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)

        gr.Examples(examples=_EXAMPLES, inputs=msg, label="💡 이런 질문을 해보세요")
        clear_btn = gr.Button("🗑️ 대화 초기화", variant="secondary", size="sm")

        def respond(message: str, history: list):
            if not message.strip():
                yield "", history
                return

            question = _extract_text(message)

            # 입력을 즉시 남기고, 어시스턴트엔 타이핑 애니메이션 표시
            if _USE_MESSAGES:
                history = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": _TYPING},
                ]
            else:
                history = history + [[question, _TYPING]]
            yield "", history

            context, urls = retriever.build_context(question)

            if model is not None and tokenizer is not None:
                # 멀티턴: 이전 대화 이력 전달
                chat_history = None
                if _USE_MESSAGES and len(history) > 2:
                    prior = [
                        {"role": h["role"], "content": h["content"]}
                        for h in history[:-2]
                        if isinstance(h, dict) and h.get("content")
                    ]
                    while prior and prior[0]["role"] != "user":
                        prior.pop(0)
                    chat_history = prior or None

                for partial in generate_answer_stream(
                    question, context, urls, model, tokenizer, history=chat_history
                ):
                    _set_bot(history, partial)
                    yield "", history
            else:
                _set_bot(history, fallback_answer(question, context, urls))
                yield "", history

        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: (_initial_history(), ""), outputs=[chatbot, msg])

    return app


def launch(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
    share: bool = True,
) -> None:
    """Gradio 챗봇 UI를 실행한다."""
    app = create_app(retriever, model, tokenizer)
    print("[ui] Gradio UI 시작...")
    app.launch(share=share)
