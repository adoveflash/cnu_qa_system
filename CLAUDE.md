# CLAUDE.md

충남대학교 학내 정보 Q/A 시스템 — 자연어처리 텀프로젝트

## 프로젝트 개요

- **목적**: 충남대 학사·장학금 정보를 RAG + LoRA 파인튜닝 LLM으로 답변하는 Q/A 시스템
- **언어**: Python 3.10+
- **마감**: 기말고사 날 자정
- **제출물**: Colab 노트북 1개 (`submission.ipynb`)

## 절대 금지

- 평가용 Q&A 100개를 학습 데이터에 포함하지 않는다
- KorQuAD 등 **기존 공개 Q&A 데이터셋을 학습에 사용하지 않는다** (과제 규정 위반)
- 학번/비밀번호/API 키를 코드·문서·커밋에 평문으로 쓰지 않는다
- 충남대 로그인 페이지를 자동 인증으로 우회하지 않는다
- Colab 무료 T4 (15GB VRAM)에서 OOM 나는 설정으로 제출하지 않는다
- base 모델 전체 가중치를 저장소에 커밋하지 않는다 (LoRA 어댑터만)
- 도메인 범위를 사용자 승인 없이 확장하지 않는다

## 고정된 기술 결정

| 항목      | 값                                                       |
| --------- | -------------------------------------------------------- |
| Base 모델 | `Qwen/Qwen2.5-7B-Instruct` (4bit NF4 양자화)             |
| 파인튜닝  | QLoRA, r=16, alpha=32, dropout=0.05, target=q/k/v/o_proj |
| 임베딩    | `BAAI/bge-m3`                                            |
| 벡터 DB   | ChromaDB (로컬)                                          |
| 웹 UI     | Gradio (`share=True`)                                    |
| 시드      | 42 (학습·평가 모든 코드에 고정)                          |

위 항목을 변경하려면 메모리 한계 등 **수치 근거**를 제시한다. 단순 취향 변경은 거부한다.

## 도메인 범위

다음 2개 영역만 다룬다.

- 학사 정보 (수강신청, 졸업 요건, 휴·복학, 전과·복수전공)
- 장학금 (교내·외 종류, 신청 절차, 자격 요건)

**로그인 필요 페이지는 1차 범위에서 제외**한다. 공개 페이지만 다룬다.

## 디렉터리 구조

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── submission.ipynb            # 제출용 — 평가자가 "모두 실행"만으로 결과를 봐야 함
├── src/
│   ├── crawl/                  # 크롤러, HTML 정제
│   ├── data/                   # 청킹, Q&A 자동 생성
│   ├── rag/                    # 임베딩, 벡터 DB, 검색
│   ├── model/                  # base 로드, LoRA 학습, 추론
│   ├── eval/                   # LLM judge, latency 측정
│   └── app/                    # Gradio UI
├── data/
│   ├── corpus/                 # 크롤링 원본 (.jsonl)
│   ├── qa/train.jsonl
│   ├── qa/eval.jsonl           # 100개, 학습 절대 포함 금지
│   └── vector_db/
├── models/lora_adapter/        # 어댑터만, < 200MB
└── docs/
    └── decisions/              # 의사결정 기록 (왜 이 모델? 왜 이 구조?)
```

`src/<모듈>/AGENTS.md`가 있다면 그 디렉터리 작업 시 함께 읽는다.

## 코드 규칙

- 포매터: `ruff format --line-length 100`
- 린터: `ruff check`
- 공개 함수는 타입 힌트 필수
- 함수 docstring에 인자·반환 명시 (한국어 OK)
- 변수명: `user_id`, `is_logged_in` (역할이 드러나게)
- 시크릿은 `.env` 또는 Colab `userdata`에서만 읽는다

## 데이터 파이프라인 규칙

### 크롤링

- `robots.txt`를 매 도메인마다 확인한다
- User-Agent: `CNU-NLP-Project/1.0 (academic-research)`
- 요청 간 `time.sleep(1.5)` 이상
- 출력: `data/corpus/raw/{source}_{YYYYMMDD}.jsonl`, 한 줄에 `{url, title, content, crawled_at}`

### Q&A 자동 생성

- 외부 LLM(GPT-4o-mini 또는 Claude Haiku)으로 청크당 3~5쌍 생성
- 프롬프트는 `src/data/prompts/qa_gen.txt`에 저장 (인라인 작성 금지)
- 평가용 100개는 **반드시 사람(개발자 본인)이 직접 검수**

### 청킹

- 300~500 토큰, 50 토큰 오버랩
- 문단·제목 경계 우선

## 추론 파이프라인 규칙

```
질문 → bge-m3 임베딩 → ChromaDB top-5 검색 →
[시스템 프롬프트 + 컨텍스트 + 질문] → Qwen2.5-7B + LoRA → 답변 + 출처 URL
```

- **모든 답변에 출처 URL을 함께 반환**한다
- 컨텍스트에 없는 내용은 추측하지 않는다 → "확인되지 않은 정보입니다" 응답
- 시스템 프롬프트: "충남대학교 학내 정보 안내 도우미"로 고정

## 평가 노트북 요건

`submission.ipynb`는 다음 순서로 동작한다. 중간에 사용자 입력을 요구하지 않는다.

1. `pip install` 일괄 설치 (런타임 5분 이내)
2. base 모델 4bit + LoRA 어댑터 로드 (어댑터는 HF Hub에서 다운로드)
3. 벡터 DB 인덱스 다운로드 (Google Drive 또는 HF Hub)
4. Gradio UI 실행 (`share=True`)
5. 평가용 100개 일괄 추론 → LLM judge 점수 + latency 출력

## 의사결정 기록

새로운 기술 선택이나 구조 변경 시 `docs/decisions/NNN-제목.md`에 다음을 기록한다.

- 결정 사항
- 검토한 대안
- 선택 이유 (수치 또는 명확한 사유)

이 기록이 다음 세션의 컨텍스트가 된다.

## 작업 시작 전 체크

- [ ] 현재 어느 단계 작업인가? (crawl / data / rag / train / eval / app)
- [ ] 변경이 `submission.ipynb` 실행에 영향을 주는가?
- [ ] 새 의존성이 Colab T4에서 동작하는가?

## 작업 종료 전 체크

- [ ] `ruff check` 통과
- [ ] `submission.ipynb`가 처음부터 끝까지 실행되는가?
- [ ] 평가용 100개를 건드리지 않았는가?
- [ ] 의사결정이 있었다면 `docs/decisions/`에 기록했는가?

---

> 이 문서는 점진적으로 보강한다. 에이전트가 실수할 때마다 해당 규칙을 추가한다.
> 길이를 늘리는 게 아니라, 모호한 규칙을 구체화하는 방향으로.
