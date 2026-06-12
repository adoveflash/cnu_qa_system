"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
Gradio 4.x/5.x (tuples) 및 6.x (messages) 자동 호환.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream

# Gradio 5.x는 type="messages" 필요, 6.x는 기본이 messages (파라미터 제거됨), 4.x는 tuples
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_USE_MESSAGES = _GRADIO_MAJOR >= 5
_NEED_TYPE_PARAM = _GRADIO_MAJOR == 5  # 6.x는 type 파라미터 없음

_CSS = """
/* ===== 블루 테마: 시원한 하늘색 그라데이션 + 카드형 말풍선 ===== */
:root {
    --blue: #2563EB;          /* 메인 블루 */
    --blue-deep: #1D4ED8;     /* 진한 블루 */
    --sky: #3B82F6;           /* 밝은 하늘 */
    --navy: #0B2F5E;          /* 제목용 네이비 */
    --bg: #EAF1FB;            /* 배경 */
    --card: #FFFFFF;          /* 카드/채팅 배경 */
    --soft: #E1ECFB;          /* 칩 배경 */
    --ink: #122038;           /* 본문 텍스트 */
    --muted: #5C6B84;         /* 보조 텍스트 */
    --line: #D4E1F4;          /* 경계선 */
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Apple SD Gothic Neo', sans-serif !important;
    color: var(--ink) !important;
}
body, .gradio-container {
    background: linear-gradient(180deg, #E7F0FC 0%, #F4F8FE 55%, #FFFFFF 100%) !important;
}
/* ===== 헤더 ===== */
.title-area { text-align: center; padding: 26px 0 6px 0; }
.title-area .logo-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 66px; height: 66px; border-radius: 19px;
    background: linear-gradient(135deg, var(--sky), var(--blue-deep));
    color: #fff; font-size: 1.95em; line-height: 1;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.38);
    margin-bottom: 14px;
}
.title-area h1 {
    font-size: 2.05em; font-weight: 800; margin-bottom: 6px;
    color: var(--navy); letter-spacing: -0.5px;
}
.title-area p { color: var(--muted); font-size: 0.96em; margin-top: 0; }
/* ===== 카테고리 칩 ===== */
.category-badges {
    display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin: 18px 0 22px 0;
}
.category-badges span {
    background: var(--card); color: var(--blue-deep);
    padding: 6px 14px; border-radius: 999px; font-size: 0.82em; font-weight: 600;
    border: 1px solid var(--line); box-shadow: 0 1px 3px rgba(11, 47, 94, 0.05);
    transition: all 0.18s ease;
}
.category-badges span:hover {
    background: var(--blue); color: #fff; border-color: var(--blue); transform: translateY(-1px);
}
footer { visibility: hidden }
/* ===== 채팅 영역 ===== */
.chatbot {
    border-radius: 20px !important;
    border: 1px solid var(--line) !important;
    background: var(--card) !important;
    box-shadow: 0 12px 32px rgba(11, 47, 94, 0.08) !important;
}
/* ===== 메시지: Gemini 스타일 (내 메시지=우측 블루 말풍선 / 봇=좌측 박스 없는 평문) ===== */
/* 행 정렬: 사용자 우측, 봇 좌측 */
.message-row.user-row { justify-content: flex-end !important; }
.message-row.bot-row { justify-content: flex-start !important; }
/* 내 메시지: 우측 정렬 + 폭 제한 (침범 방지) */
.chatbot .message.user, .chatbot [data-testid="user"], .message-row.user-row .message {
    display: inline-block !important;
    max-width: 72% !important;
    margin-left: auto !important;
    background: linear-gradient(135deg, var(--sky), var(--blue)) !important;
    border: none !important;
    border-radius: 18px 18px 5px 18px !important;
    color: #fff !important;
    text-align: left !important;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.22) !important;
}
.chatbot .message.user *, .message-row.user-row .message * { color: #fff !important; }
/* 봇 응답: 박스·테두리 없이 평문, 좌측, 넓게 (제미나이처럼) */
.chatbot .message.bot, .chatbot [data-testid="bot"], .message-row.bot-row .message {
    max-width: 90% !important;
    margin-right: auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding-left: 2px !important;
    color: var(--ink) !important;
}
/* ===== 컴팩트: 말풍선 패딩·줄간격·메시지 간격 축소 (세로 길이 줄임) ===== */
.chatbot .message {
    padding: 9px 14px !important;
    line-height: 1.5 !important;
    font-size: 0.95em !important;
}
.chatbot .message p, .chatbot .message li {
    margin: 0 0 0.3em 0 !important;
    line-height: 1.5 !important;
}
.chatbot .message p:last-child, .chatbot .message ul:last-child { margin-bottom: 0 !important; }
.message-row { margin: 2px 0 !important; padding: 2px 0 !important; }
.chatbot .message-wrap { gap: 4px !important; }
/* ===== 전송 버튼 ===== */
button.primary {
    background: linear-gradient(135deg, var(--sky), var(--blue-deep)) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; color: #fff !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.18s ease !important;
}
button.primary:hover { filter: brightness(1.08); transform: translateY(-1px); }
/* 보조 버튼 */
button.secondary {
    background: transparent !important; border: 1px solid var(--line) !important;
    border-radius: 12px !important; color: var(--muted) !important;
}
button.secondary:hover { border-color: var(--blue) !important; color: var(--blue-deep) !important; }
/* ===== 입력창 ===== */
textarea {
    border-radius: 14px !important; border: 1px solid var(--line) !important;
    background: var(--card) !important; color: var(--ink) !important; font-size: 1.0em !important;
}
textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}
/* ===== 타이핑 애니메이션 (점 3개 통통) ===== */
.typing-dots { display: inline-flex; gap: 5px; align-items: center; padding: 5px 2px; }
.typing-dots span {
    width: 9px; height: 9px; border-radius: 50%; background: var(--blue);
    animation: typing-bounce 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.18s; }
.typing-dots span:nth-child(3) { animation-delay: 0.36s; }
@keyframes typing-bounce {
    0%, 70%, 100% { transform: translateY(0) scale(0.85); opacity: 0.45; }
    35% { transform: translateY(-7px) scale(1); opacity: 1; }
}
/* ===== 스트리밍 중 깜빡이는 커서 ===== */
.caret {
    display: inline-block; width: 8px; height: 1.05em; background: var(--blue);
    margin-left: 2px; border-radius: 2px; vertical-align: text-bottom;
    animation: caret-blink 0.9s steps(1) infinite;
}
@keyframes caret-blink { 0%, 50% { opacity: 1; } 50.01%, 100% { opacity: 0; } }
/* ===== 출처 배지 (블루) ===== */
.source-row {
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    margin-top: 12px; padding-top: 11px; border-top: 1px solid var(--line);
}
.source-label {
    font-size: 0.72em; color: var(--muted); font-weight: 700; margin-right: 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.source-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--soft); border: 1px solid var(--line); color: var(--blue-deep);
    padding: 4px 11px; border-radius: 999px; font-size: 0.76em; font-weight: 600;
}
/* ===== 예시 질문 ===== */
.examples-row button, .examples button {
    border-radius: 12px !important; font-size: 0.86em !important;
    background: var(--card) !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}
.examples-row button:hover, .examples button:hover {
    border-color: var(--blue) !important; background: #F0F6FF !important;
}
"""

# 생성 대기 중 표시할 타이핑 애니메이션 / 스트리밍 커서
_TYPING = '<span class="typing-dots"><span></span><span></span><span></span></span>'
_CARET = '<span class="caret"></span>'

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
    """Gradio 챗봇 UI를 생성한다."""

    with gr.Blocks(css=_CSS, title="CNU Q&A 챗봇") as app:
        gr.HTML(
            """
            <div class="title-area">
                <div class="logo-icon">&#10024;</div>
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

        chatbot_kwargs = {"height": 640, "show_label": False, "container": False}
        if _NEED_TYPE_PARAM:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(
            **chatbot_kwargs, value=_initial_history(), elem_classes=["chatbot"]
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="궁금한 점을 물어보세요! (예: 졸업요건, 수강신청, 식단 등)",
                show_label=False,
                scale=9,
                container=False,
                lines=1,
                max_lines=5,
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)

        with gr.Row():
            gr.Examples(
                examples=_EXAMPLES,
                inputs=msg,
                label="💡 이런 질문을 해보세요",
            )

        with gr.Row():
            clear_btn = gr.Button("🗑️ 대화 초기화", variant="secondary", size="sm")

        def respond(message: str, history: list):
            if not message.strip():
                yield "", history
                return

            question = _extract_text(message)

            # 입력을 즉시 채팅창에 남기고, 어시스턴트 말풍선엔 타이핑 애니메이션 표시
            if _USE_MESSAGES:
                history = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": _TYPING},
                ]
            else:
                history = history + [[question, _TYPING]]
            yield "", history

            # RAG 검색 (이 동안 타이핑 점이 계속 통통 튐)
            context, urls = retriever.build_context(question)

            if model is not None and tokenizer is not None:
                # 멀티턴: 이전 대화 이력을 모델에 전달
                chat_history = None
                if _USE_MESSAGES and len(history) > 2:
                    prior = [
                        {"role": h["role"], "content": h["content"]}
                        for h in history[:-2]
                        if isinstance(h, dict) and h.get("content")
                    ]
                    # 환영 메시지 등 user 이전 메시지 제거 → 반드시 user로 시작
                    while prior and prior[0]["role"] != "user":
                        prior.pop(0)
                    chat_history = prior or None

                last = ""
                for partial in generate_answer_stream(
                    question, context, urls, model, tokenizer, history=chat_history
                ):
                    last = partial
                    # 생성 중에는 끝에 깜빡이는 커서를 붙여 '타이핑 중'처럼 보이게
                    _set_bot(history, partial + _CARET)
                    yield "", history
                # 최종: 커서 제거
                _set_bot(history, last)
                yield "", history
            else:
                answer = fallback_answer(question, context, urls)
                _set_bot(history, answer)
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
