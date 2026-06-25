# CLAUDE.md

충남대학교 Campus ChatBot — 자연어처리 텀프로젝트

## 프로젝트 개요

- **목적**: 충남대 재학생을 위한 AI 챗봇 (질문 분류 + RAG 기반 챗봇 + 실시간 정보 반영)
- **언어**: Python 3.10.12
- **마감**: 2026-06-12 (금) 자정 (사이버캠퍼스 제출)
- **제출물**: `Termproject_{이름}.zip` (소스코드, 모델 파일/다운로드 링크, 발표자료, UI 영상)
- **참고 PDF**: `2_term_project_QA.pdf` (초기 과제 설명), `2026_자연어처리_term_project.pdf` (최종 과제 명세)

## 평가 항목 (총 130점)

| Task   | 내용                                       | 배점 | 평가 방식                                                 |
| ------ | ------------------------------------------ | ---- | --------------------------------------------------------- |
| Task 1 | 질문 유형 분류기 (Question Classification) | 40   | F1 Score (정량)                                           |
| Task 2 | 챗봇 모델 및 인터페이스 (Chat Model & UI)  | 60   | 정성 평가 (UI 구동 10 + Chat Interface 10 + 맥락 응답 40) |
| Task 3 | 실시간 정보 반영 (Optional)                | 30   | 정성 평가 (실제 정보 동기화 정확성)                       |

### Task 1: 질문 유형 분류기

- 5개 카테고리로 분류, F1 Score로 평가
- **라벨**: 졸업요건=0, 학교 공지사항=1, 학사일정=2, 식단 안내=3, 통학/셔틀 버스=4
- 입력: `data/test_cls.json` (`[{"question": "..."}]`)
- 출력: `outputs/cls_output.json` (`[{"question": "...", "label": N}]`)
- 실행 파일: `src/classifier.ipynb`

### Task 2: 챗봇 모델 및 인터페이스

- UI: 사용자가 질문 입력 → 응답 확인 (영상으로 평가)
- Chat Model: 테스트셋에 대한 배치 추론 결과 JSON 제출
- 입력: `data/test_chat.json` (`[{"user": "..."}]`)
- 출력: `outputs/chat_output.json` (`[{"user": "...", "model": "..."}]`)
- 실행 파일: `chatbot.sh` (JSON 생성 + UI 실행)

### Task 3: 실시간 정보 반영 (Optional)

- RAG, Crawling 등으로 최신 정보 반영
- tool calling 사용 가능
- 입력: `data/test_realtime.json` (`[{"user": "..."}]`)
- 출력: `outputs/realtime_output.json` (`[{"user": "...", "model": "..."}]`)

## 절대 금지

- 평가용 테스트셋(`test_cls.json`, `test_chat.json`, `test_realtime.json`)을 학습 데이터에 포함하지 않는다
- KorQuAD 등 **기존 공개 Q&A 데이터셋을 학습에 사용하지 않는다** (과제 규정 위반)
- User-Agent를 검색엔진 봇(Googlebot 등)으로 위장하지 않는다
- 수집한 데이터에 본인 또는 타인의 개인정보(학번, 이름, 성적 등)를 남기지 않는다 — 반드시 마스킹
- Colab 무료 T4 (15GB VRAM)에서 OOM 나는 설정으로 제출하지 않는다
- base 모델 전체 가중치를 저장소에 커밋하지 않는다
- 도메인 범위를 사용자 승인 없이 확장하지 않는다

## 고정된 기술 결정

| 항목                             | 값                                                             |
| -------------------------------- | -------------------------------------------------------------- |
| Base 모델                        | `google/gemma-4-12b-it` (4bit NF4 양자화) — Qwen3-8B에서 교체   |
| 파인튜닝                         | 없음 (LoRA 제거, base 모델 직접 사용)                            |
| 임베딩                           | `BAAI/bge-m3`                                                  |
| 벡터 DB                          | ChromaDB (로컬)                                                |
| 웹 UI                            | Gradio (`share=True`)                                          |
| 시드                             | 42 (학습·평가 모든 코드에 고정)                                |

위 항목을 변경하려면 메모리 한계 등 **수치 근거**를 제시한다. 단순 취향 변경은 거부한다.
모델 교체는 `src/model/base.py`에서 한 줄 변경으로 가능하도록 설계한다.

## 도메인 범위 (5개 카테고리)

과제 명세에서 지정한 5개 카테고리:

1. **졸업요건** (label=0) — 졸업학점, 전공/교양 졸업 요건 등
2. **학교 공지사항** (label=1) — 학교/학과 공지사항
3. **학사일정** (label=2) — 수강 신청/정정 기간, 학기 일정 등
4. **식단 안내** (label=3) — 교내 식당의 주간/일일 메뉴
5. **통학/셔틀 버스** (label=4) — 버스 시간표, 정류장 위치, 운행 여부 등

추가로 다루는 영역 (챗봇 응답용):

- 장학금 (교내·외 종류, 신청 절차, 자격 요건)
- 취업·진로 (인재개발원 프로그램, 채용 공지)
- 컴퓨터융합학부 정보 (전공 트랙, 커리큘럼, 학과 공지)

### 수집 대상 도메인 (1차)

robots.txt 정책상 자동 크롤링 가능 또는 사람-속도 수집이 허용되는 공개 페이지:

- `plus.cnu.ac.kr/html/hub/` (4처1국 통합 허브) — `/html/` 경로는 검색엔진 봇에 Allow
- `plus.cnu.ac.kr/html/kr/` (학교 공식 안내)
  - `/html/kr/sub05/sub05_050403.html` — 셔틀버스 시간표, 노선, 정류장
  - `/html/kr/sub05/sub05_05050101.html` — 교내 식당 위치/운영시간
  - `/html/kr/sub05/sub05_050404.html` — 금주의식단 안내
  - `/html/kr/sub01/sub01_01080302.html` — 교통편 안내 (시내버스 노선)
- `computer.cnu.ac.kr` (컴퓨터융합학부)
- `job.cnu.ac.kr` (인재개발원)
- `sugang.cnu.ac.kr/login/data/2026_Sugang.pdf` (수강신청 매뉴얼)

### 수집 대상 도메인 (2차 — 식단/셔틀 보강)

robots.txt 없음(404) = 명시적 차단 없음. 학술 목적 수집 허용:

- `mobileadmin.cnu.ac.kr/food/index.jsp` — **주간 식단 메뉴** (조식/중식/석식, 가격)
  - 제1~4학생회관 + 생활과학대학 식당
  - JS 렌더링 → `playwright` 필요
- `www.cnucoop.co.kr` — **생활협동조합 식당 안내** (robots.txt 없음)
  - `/ezhtml2.php?html=canteen` — 식당 목록, 위치, 운영시간
  - **기숙사 식당(남학생 기숙사 식당, N-14)** 포함
  - 정적 HTML → `requests` + `BeautifulSoup` (EUC-KR 인코딩 주의)
- `dorm.cnu.ac.kr` — **기숙사 생활 안내** (학술 과제 목적, 공개 페이지 한정)
  - `/html/kr/sub03/` — 입사 안내, 생활 규정, 시설 안내 등
  - `/html/kr/sub03/sub03_0304.html` — 기숙사 식단표

### 수집 대상 도메인 (3차, 시간 여유 시)

본인 계정 세션 쿠키를 활용한 인증 후 수집. **모든 학생에게 공통인 정보만 수집**하며 개인정보 페이지는 제외:

- `portal.cnu.ac.kr` (학사정보 메뉴 중 공통 정보)
- `sugang.cnu.ac.kr/main.do` (강의 검색, 시간표)

### 제외

- `www.cnu.ac.kr` — robots.txt 전면 Disallow
- `dorm.cnu.ac.kr` — robots.txt Disallow이나, 본교 재학생의 학술 과제 목적으로 공개 페이지(기숙사 안내, 식단 등)에 한해 최소한으로 수집 허용. 개인정보 페이지 제외, 요청 간 3초 이상 간격 준수.
- 본인 또는 타 학생의 개인정보 페이지 (성적, 등록금 납부내역, 장학금 수령 내역 등)
- `with.cnu.ac.kr` (학생 커뮤니티 — 타 학생 게시물 포함, 개인정보 리스크)

## 디렉터리 구조

과제 명세 기준 제출 구조 (빨간색 = 평가용 고정 경로):

```
Termproject_{이름}/
├── data/
│   ├── test_cls.json           # 평가용 (조교 제공, 고정 경로)
│   ├── test_chat.json          # 평가용 (조교 제공, 고정 경로)
│   ├── test_realtime.json      # 평가용 (조교 제공, optional)
│   ├── train.json              # Task1 분류 시드 (직접 구축)
│   ├── train_augmented.json    # Task1 분류 증강 (Claude Haiku paraphrase)
│   ├── valid.json              # Task1 검증 데이터 (직접 구축)
│   ├── qa/                     # Task2 RAG용 Q&A (train/eval)
│   └── corpus/                 # 크롤링 raw·정제(cleaned)·청크(chunks)·vector_db
├── src/                        # ── 평가 대상 소스 ──
│   ├── classifier.ipynb        # Task 1 — 평가 시 이것만 실행
│   ├── chatbot_ui.py           # Task 2 — 배치추론/UI 진입
│   ├── realtime_model.py       # Task 3 — 실시간 정보 반영
│   ├── app/                    # Streamlit 웹 UI (streamlit_app.py)
│   ├── crawl/                  # 크롤러, HTML/PDF 정제
│   ├── data/                   # 청킹, Q&A 생성, 개인정보 마스킹
│   ├── rag/                    # 임베딩, 벡터 DB, 검색 (Advanced RAG)
│   ├── model/                  # base 로드, 추론, 직접 구현 BERT(bert_scratch)
│   ├── skills/                 # 실시간 스킬(식단·셔틀·학사·공지) + 자동 레지스트리
│   ├── tools/                  # (레거시 shim → skills 위임)
│   └── eval/                   # 평가 스크립트
├── model/                      # 모델 안내·다운로드 (README, download_model.py)
├── models/
│   └── lora_adapter/           # (레거시, 현재 미사용)
├── notebooks/                  # submission·gemma3_submission·스모크 테스트
├── scripts/                    # 개발·유지보수 (build_db, restore.sh, make_submission 등)
├── tests/                      # 단위 테스트
├── outputs/                    # 실행 결과 (자동 생성)
│   ├── cls_output.json         # Task 1 결과
│   ├── chat_output.json        # Task 2 결과
│   └── realtime_output.json    # Task 3 결과
├── chatbot.sh                  # Task 2·3 — 평가 시 이것만 실행 (JSON 생성 + UI)
├── requirements.txt
├── README.md
├── CLAUDE.md
├── DIARY.md                    # 작업 일지 + 회고
└── docs/
    ├── 2_term_project_QA.pdf   # 과제 명세
    └── decisions/              # 의사결정 기록
```

**평가 시 실행 파일**: `src/classifier.ipynb`와 `chatbot.sh`만 실행
(개발·유지보수 스크립트는 `scripts/`, 보조/제출 노트북은 `notebooks/`에 둔다)

`src/<모듈>/AGENTS.md`가 있다면 그 디렉터리 작업 시 함께 읽는다.

## 코드 규칙

- 포매터: `ruff format --line-length 100`
- 린터: `ruff check`
- 공개 함수는 타입 힌트 필수
- 함수 docstring에 인자·반환 명시 (한국어 OK)
- 변수명: `user_id`, `is_logged_in` (역할이 드러나게)
- 시크릿은 `.env` 또는 Colab `userdata`에서만 읽는다
- 모델 로드는 `src/model/base.py`의 `load_model(name: str)`로 추상화한다

## 데이터 파이프라인 규칙

### 크롤링

- 매 도메인마다 `robots.txt`를 확인하고 Disallow 경로는 절대 건드리지 않는다
- User-Agent: `CNU-NLP-StudentProject/1.0 (academic-coursework)`
- 요청 간 `time.sleep(3.0)` 이상 (학교 서버 부담 최소화)
- 출력: `data/corpus/raw/{source}_{YYYYMMDD}.jsonl`, 한 줄에 `{url, title, content, crawled_at}`
- PDF는 `pdfplumber`로 텍스트 추출
- JS 렌더링 페이지는 `playwright`, 정적 HTML은 `requests` + `BeautifulSoup`

### 개인정보 마스킹

- 본인 학번, 이름, 주민번호, 전화번호, 이메일은 크롤링 직후 `[학번]`, `[이름]` 등 마스킹 토큰으로 치환
- 마스킹 규칙은 `src/data/pii_masker.py`에 정의
- 마스킹 후 원본은 폐기

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
[시스템 프롬프트 + 컨텍스트 + 질문] → Gemma 4 12B (4bit) → 답변 + 출처 URL
```

- **모든 답변에 출처 URL을 함께 반환**한다
- 컨텍스트에 없는 내용은 추측하지 않는다 → "확인되지 않은 정보입니다" 응답
- 시스템 프롬프트: "충남대학교 학내 정보 안내 도우미"로 고정

## 평가 실행 요건

### Task 1: `src/classifier.ipynb`

Colab에서 실행. `data/test_cls.json`을 읽어 `outputs/cls_output.json` 생성.

### Task 2: `chatbot.sh`

실행 시 두 가지 동작:

1. `data/test_chat.json`을 읽어 `outputs/chat_output.json` 생성
2. 챗봇 UI 실행 (사용자 대화 가능)

### Task 3 (Optional): `chatbot.sh` 내에서 처리

`data/test_realtime.json`을 읽어 `outputs/realtime_output.json` 생성.

### 환경

- Python 3.10.12, torch 2.5.1, pytorch-lightning 2.4.0 (사용 시)
- Colab 환경에서 실행 가능한 .ipynb 또는 .py
- `requirements.txt` 제출 필수

## 의사결정 기록

새로운 기술 선택이나 구조 변경 시 `docs/decisions/NNN-제목.md`에 다음을 기록한다.

- 결정 사항
- 검토한 대안
- 선택 이유 (수치 또는 명확한 사유)

이 기록이 다음 세션의 컨텍스트가 된다.

### 현재까지의 결정

- `001-base-model-검증.md`: Qwen2.5-7B 4bit 채택 (T4 VRAM 5.33GB 검증) → 이후 Gemma 4 12B로 교체
- `002-크롤링-범위.md`: plus.cnu.ac.kr/html/, computer, job 도메인 한정

## 제출 체크리스트

- [ ] `src/classifier.ipynb` 실행 → `outputs/cls_output.json` 정상 생성
- [ ] `chatbot.sh` 실행 → `outputs/chat_output.json` 정상 생성 + UI 실행
- [ ] (optional) `outputs/realtime_output.json` 정상 생성
- [ ] UI 작동 영상 (2분 내외) 녹화
- [ ] 발표 자료 (5분 내외) 준비 — 구현 방법 + 챗 인터페이스 동작 여부
- [ ] `requirements.txt` 최신화
- [ ] 개인정보가 데이터셋에 남아 있지 않은가?
- [ ] Colab T4에서 OOM 없이 동작하는가?
- [ ] `ruff check` 통과
- [ ] 의사결정이 있었다면 `docs/decisions/`에 기록했는가?

---

> 이 문서는 점진적으로 보강한다. 에이전트가 실수할 때마다 해당 규칙을 추가한다.
> 길이를 늘리는 게 아니라, 모호한 규칙을 구체화하는 방향으로.
