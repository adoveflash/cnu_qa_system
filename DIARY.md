# DIARY.md — 작업 일지

## 2026-05-19 (월) — 프로젝트 초기 설정

### 목적: 프로젝트 구조 수립 및 기술 검증

- 프로젝트 디렉터리 구조 생성 (`src/crawl`, `src/data`, `src/rag`, `src/model`, `src/eval`, `src/app`)
- `CLAUDE.md` 작성 — 프로젝트 규칙, 기술 스택, 금지사항 등 정의
- `.gitignore` 설정
- **모델 스모크 테스트** (`notebooks/00_model_smoke_test.ipynb`)
  - Qwen2.5-7B-Instruct 4bit(NF4) 양자화 로드 검증
  - Colab T4 기준 VRAM 5.33GB 사용 확인 → base 모델 확정
- `README.md` 작성
- 첫 커밋 (`first commit`, `docs: README, DIARY, 모델 스모크 테스트 노트북 추가`)

---

## 2026-05-21 (수) — 크롤링 파이프라인 구축

### 목적: 충남대 학내 웹사이트에서 원본 데이터 수집

- **크롤러 모듈 구현**
  - `src/crawl/static_crawler.py` — requests + BeautifulSoup 기반 정적 HTML 크롤러
  - `src/crawl/js_crawler.py` — Playwright 기반 JS 렌더링 페이지 크롤러
  - `src/crawl/pdf_crawler.py` — pdfplumber 기반 PDF 텍스트 추출
  - `src/crawl/run_crawl.py` — 크롤링 오케스트레이션 스크립트
- **1차 크롤링 실행** (3개 소스)
  - `computer_20260521.jsonl` — 컴퓨터융합학부 (computer.cnu.ac.kr)
  - `job_20260521.jsonl` — 인재개발원 (job.cnu.ac.kr)
  - `sugang_manual_20260521.jsonl` — 수강신청 매뉴얼 PDF

---

## 2026-05-25 (일) — 데이터 전처리 및 RAG 기반 구축

### 목적: 크롤링 데이터 정제, 청킹, 임베딩 파이프라인 완성

- **전처리 모듈 구현**
  - `src/data/cleaner.py` — HTML 태그 제거, 텍스트 정규화
  - `src/data/pii_masker.py` — 개인정보(학번, 이름, 전화번호, 이메일 등) 마스킹
  - `src/data/chunker.py` — 300~500 토큰 단위 청킹 (50 토큰 오버랩)
  - `src/data/run_preprocess.py` — 전처리 파이프라인 실행 스크립트
- **RAG 모듈 구현**
  - `src/rag/embedder.py` — BAAI/bge-m3 임베딩 래퍼
  - `src/rag/retriever.py` — ChromaDB 기반 top-k 검색

---

## 2026-05-26 (월) — 추가 크롤링, QA 생성, 모델/평가 모듈 완성

### 목적: 학습 데이터 생성 및 학습/추론/평가 코드 작성

- **추가 크롤링**
  - `src/crawl/crawl_plus.py` — plus.cnu.ac.kr 전용 크롤러
  - `src/crawl/robots_checker.py` — robots.txt 자동 확인 유틸
  - `plus_kr_20260526.jsonl` — plus.cnu.ac.kr/html/kr/ 크롤링 결과
- **전처리 실행**
  - `data/corpus/cleaned.jsonl` 생성 (841건)
  - `data/corpus/chunks.jsonl` 생성 (1,745 청크)
- **Q&A 자동 생성**
  - `src/data/qa_generator.py` — 외부 LLM 기반 Q&A 쌍 생성기
  - `src/data/prompts/` — QA 생성 프롬프트 템플릿
- **벡터 DB 구축**
  - `src/rag/vector_store.py` — ChromaDB 인덱스 빌드/로드
  - `data/vector_db/` — 임베딩 인덱스 저장
- **모델 모듈 구현**
  - `src/model/base.py` — 모델 로드 추상화 (4bit NF4 양자화)
  - `src/model/train.py` — QLoRA 파인튜닝 (r=16, alpha=32)
  - `src/model/inference.py` — RAG + LoRA 추론 파이프라인
- **평가 모듈 구현**
  - `src/eval/evaluate.py` — LLM judge 점수 + latency 측정

---

## 2026-05-27 (화) — QA 데이터 확정, 앱 UI, 제출 노트북

### 목적: 학습 데이터 확정 및 서비스 레이어 구현

- **Q&A 데이터 확정**
  - `data/qa/train.jsonl` — 학습용 4,745쌍
  - `data/qa/eval.jsonl` — 평가용 100쌍 (학습 데이터 미포함, 직접 검수)
- **앱 모듈 구현**
  - `src/app/api.py` — 추론 API 래퍼
  - `src/app/ui.py` — Gradio 웹 UI (`share=True`)
- **제출 노트북 작성**
  - `submission.ipynb` — Colab "모두 실행" 원클릭 동작 목표

---

## 2026-05-28 (수) — 학습 노트북 작성

### 목적: Colab 환경 학습 실행용 노트북

- **학습 노트북**
  - `train_colab.ipynb` — Colab T4에서 QLoRA 파인튜닝 실행용
- **LoRA 어댑터**
  - `models/lora_adapter/` — 학습된 어댑터 가중치 저장

---

## 2026-06-09~10 (화~수) — 모델 전략 대전환: LoRA 폐기 + Gemma 교체

### 목적: 파인튜닝 의존을 버리고 base 모델 + RAG로 품질 확보

- **전략 전환 ①: LoRA 파인튜닝 폐기**
  - Qwen3-8B + QLoRA → `gemma-4-12b-it` 4bit base 직접 사용 (파인튜닝 없음)
  - 이유: 학습 데이터 품질·검수 부담 대비 효과 낮음, RAG 컨텍스트가 더 결정적
  - 모델 교체는 `src/model/base.py` 한 줄 변경으로 가능하도록 설계 유지
- **전략 전환 ②: SSH GPU 박스 도입**
  - 개발/점검용 RTX 8000 48GB 서버에서 풀 RAG 파이프라인 구동 (torch 2.12, py 3.10.12)
  - 채점은 어디까지나 Colab T4 기준 — 박스는 빠른 반복용, 제출 검증은 Colab으로 별도
  - `restore.sh` 병합 방식 (박스 학과 환경 + Mac 코어)
- **코퍼스 품질 강화**
  - 5개 카테고리 깊은 커버리지로 재정제·마스킹, 벡터DB 재구축 (`build_db.py`)
  - 졸업요건 가드: "130학점 전체 단정" 제거, 학과별 차이(건축 등 예외) 명시
  - 검증된 졸업학점(130) manual 청크 주입
- **답변 품질 튜닝**
  - 출력 스캐폴딩 제거, rep_penalty 1.3→1.1, 식단 되묻기 금지 규칙
  - 인사·잡담엔 RAG 출처 미부착 (엉뚱한 출처 방지)

---

## 2026-06-10~11 (수~목) — torch 충돌 → Gemma 3 폴백 + 제출 모델 확정

### 목적: Colab/채점 환경의 torch 버전 제약 해소

- **전략 전환 ③: Gemma 4 → Gemma 3 폴백 (Tier 체계)**
  - Gemma 4는 `float8_e8m0fnu` 요구 → torch 2.5.1 불가, torch 2.7+/Colab 2.11 필요
  - 채점 환경 안정성 위해 **제출 모델을 Gemma 3로 확정** (2026-06-11)
  - 폴백 Tier: Gemma3 (Tier B, 쉬움) / Qwen2.5-7B (검증됨, Tier C에서 torch 2.5.1)
  - Tier별 검증 노트북으로 SSH·Colab 양쪽 동작 확인
- **이중 실행 경로 확정**
  - 제출 노트북: torch 2.5.1 고정 (런타임 재시작 2번) + Gemma 3, bf16
  - `chatbot.sh`: Colab 기본 torch + Gemma 3, 재시작 없음 (원클릭)
  - sentence-transformers 2.7.0 핀 (torch 2.5.1 호환), torchcodec 제거
- **tool calling 추가 (Task 3)**
  - 보조 노트북에 실시간 tool: 식단·셔틀·학사일정·공지
  - 제1학생회관 메뉴는 코드 상수로 (raw 파일 gitignore 의존 제거)

---

## 2026-06-12 (금) — 최종 제출 + 한글 UI 버그 사투

### 목적: 마감일 제출물 확정

- **최종 제출 zip 확정**: `Termproject_장윤상.zip` (벡터DB 번들 정리 35MB, UTF-8 플래그, ~17MB)
- **전략 전환 ④: UI 한글 세로깨짐 버그 (교훈)**
  - Gradio 말풍선 레이아웃 CSS를 덮어쓰면 한글이 세로로 쪼개짐
  - `display:inline-block`의 `min-content`가 1글자로 쪼개는 게 원인
  - 결론: **기본 렌더링 + 테마만 사용**, 메시지 레이아웃 CSS는 건드리지 않음
  - Gradio 기본 블루테마로 UI 새로 작성
- T4 OOM 방지·130학점 가드·재시도 로직 포함한 제출 최종본

---

## 2026-06-22~23 (월~화) — 최종본 고도화: BERT 직접 구현 + Streamlit + 스킬화

### 목적: 최종 제출 버전을 위한 분류기·UI·RAG 재설계

- **전략 전환 ⑤: Task1 분류기 직접 구현 BERT**
  - HF 사전학습 BERT 파라미터 복사 + 임베딩 검증으로 BERT를 직접 구현
  - 분류기를 이 직접 구현 BERT로 전환
- **전략 전환 ⑥: UI Gradio → Streamlit**
  - `src/app/streamlit_app.py`, `chatbot.sh`가 `streamlit run` 호출
  - bge-m3 임베딩 device 자동감지(GPU)
  - Streamlit history 앞쪽 비-user 메시지 제거 (Gemma 교대 규칙 위반 해결)
  - 결정 기록: `docs/decisions/003-ui-gradio에서-streamlit으로.md`
- **전략 전환 ⑦: Advanced RAG + UI 설정 톱니바퀴**
  - UI 설정에서 RAG 모드 분기 가능
- **전략 전환 ⑧: 툴 → 스킬 모듈 구조화**
  - 자동 레지스트리 기반 스킬 모듈로 리팩터링
- **인프라 픽스**: nvjitlink torch import 심볼 에러 → `chatbot.sh`/`restore.sh`에 `LD_LIBRARY_PATH` 가드 (nvjitlink 휠 앞세움)

---

## 회고 — 키워드 기반 tool-calling의 한계 (발표용)

### 우리가 택한 방식

실시간 정보(식단·셔틀·학사일정·공지)를 **키워드 매칭으로 스킬을 라우팅**하고, 매칭되면
학교 서버를 실시간 조회해 답변에 주입했다. 감지 실패 시 RAG 코퍼스로 fallback.

### 드러난 문제점

1. **자주 안 바뀌는 정보까지 실시간 조회 — 손해**
   - 셔틀버스 시간표·제1학생회관 메뉴는 학기 단위로 고정인데 매 질문마다 서버를 조회.
   - 느리고(요청 간 3초 + 재시도) 불안정하기만 하고 이득이 없음 → 정적 정보는
     기존 RAG 데이터로 생성하는 게 더 빠르고 안정적.
   - 실제로 우리도 6/11에 "제1학생회관 메뉴를 코드 상수로" 박은 커밋이 있음
     (= 무의식중에 이 결론을 부분 인정했으나 원칙으로 정리하지 못하고 땜질).

2. **키워드 매칭은 질문의 의미를 고려하지 않음**
   - "의미 이해"를 LLM이 아니라 사람이 키워드로 대신하는 구조 → 일반화 불가.

3. **서버 조회 실패 시 RAG fallback의 함정**
   - 실시간 조회 실패 → 과거 RAG 데이터 사용 → **수정 전 데이터로 잘못된 정보 전달** 위험.
   - 차라리 "지금 최신 정보 확인 실패"라고 솔직히 답하는 게 나음.

4. **키워드에 없는 표현은 인식 불가 — 하드코딩 지옥**
   - 지정 키워드에 없는 컨텍스트가 들어오면 스킬 자체가 발동 안 됨.
   - 실제 테스트에서 **"1학"(제1학생회관 줄임말)을 추론하지 못해 이상한 답변**을 낸 사례 발생.
   - "1학", "일학", "1학관" 같은 변형·줄임말·오타를 사람이 무한정 하드코딩해야 하는 문제.

### 근본 원인

> 의미 판단을 LLM이 아니라 키워드 규칙(사람)이 대신하게 만든 것.
> 장점(빠름·재현·디버그 쉬움)의 대가로 일반화 불가 + 하드코딩 + 정적/동적 구분 실패를 떠안음.

### 더 나은 설계 (개선 방향)

- **의미 기반 라우팅**: 스킬 설명을 임베딩해두고 질문과 벡터 유사도로 스킬 선택.
  이미 보유한 bge-m3로 추가 비용 거의 없이 가능 — "1학" 사고를 방지할 수 있었음.
- **LLM 의도분류**: 작은 프롬프트로 식단/셔틀/일반을 모델이 판단.
- **정적/동적 분리**: 셔틀·1학 메뉴는 RAG/상수, 진짜 자주 바뀌는 것만 tool.
- **fallback 투명화**: 실패 시 "최신 확인 실패" 명시 또는 정적 데이터에 날짜 표기.

---

## 핵심 전략 결정 타임라인 (요약)

| 시점 | 전환 | 결정 |
|------|------|------|
| ~05-28 | 초기 | Qwen2.5-7B + QLoRA + Gradio |
| 06-09 | 모델 | LoRA 폐기 → Gemma 4 12B base + RAG |
| 06-10 | 인프라 | SSH GPU 박스(개발) / Colab(채점) 분리 |
| 06-11 | 모델 | Gemma 4 → **Gemma 3 확정** (torch 호환), Tier 폴백 체계 |
| 06-12 | 제출 | 1차 최종 zip, 한글 UI 버그 → 기본 렌더링 원칙 |
| 06-22 | 분류기 | **직접 구현 BERT** 전환 |
| 06-22 | UI | Gradio → **Streamlit** |
| 06-22 | RAG | Advanced RAG 모드 분기 |
| 06-23 | 구조 | 툴 → 스킬 모듈(자동 레지스트리) |

---

## 진행 현황 요약

| 단계 | 상태 | 산출물 |
|------|------|--------|
| 프로젝트 설정 | 완료 | CLAUDE.md, 디렉터리 구조, 스모크 테스트 |
| 크롤링 | 완료 | 5개 카테고리 소스, raw JSONL |
| 전처리 | 완료 | 재정제·마스킹 코퍼스 + 청킹 |
| Q&A 생성 | 완료 | train + eval 100쌍(직접 검수) |
| RAG 구축 | 완료 | bge-m3 임베딩 + ChromaDB, Advanced RAG 모드 |
| Task1 분류기 | 완료 | 직접 구현 BERT (klue/bert-base copy + 검증) |
| 챗봇 모델 | 완료 | Gemma 3 base + RAG (LoRA 없음) |
| Task3 실시간 | 완료 | 스킬 모듈(식단·셔틀·학사일정·공지) tool calling |
| 앱 UI | 완료 | Streamlit (`chatbot.sh` → streamlit run) |
| 제출 | 1차 완료(06-12) | Termproject_장윤상.zip / Final 버전 고도화 중 |

---

## Task별 해결 상태 점검 (자가 기록)

### Task 2 (챗봇 + UI, 60점) — ✅ 해결

| 평가 항목 | 상태 |
|----------|------|
| UI 구동 (10) | ✅ Streamlit (`chatbot.sh` → streamlit run) |
| Chat Interface (10) | ✅ 멀티턴, 스트리밍 출력, 출처 배지 |
| 맥락 응답 (40) | ✅ RAG(bge-m3+ChromaDB) + Gemma 3, 시스템 프롬프트로 환각 억제 |
| 배치 추론 산출물 | ✅ `chat_output.json` |

- 약점: 작은 양자화 모델 + greedy → 답변 품질 상한 낮음, T4에서 느림. 치명적이지 않음.

### Task 3 (실시간, 30점 optional) — ⚠️ 구현 완료, 품질 의문

- 동작: ✅ 식단·셔틀·학사·공지 실시간 조회 + `realtime_output.json` 생성
- 설계 품질: ⚠️ 위 "회고 — 키워드 기반 tool-calling의 한계" 4가지가 그대로 약점
  (키워드 라우팅·정적정보 실시간조회·fallback 오답·"1학" 미인식)

### Task 1 (질문 5분류, 40점) — ✅ 해결

- 모델: `klue/bert-base`를 **밑바닥부터 직접 구현한 BERT**에 가중치 copy
  (`src/model/bert_scratch.py`, 노트북에 인라인). 임베딩 오차 < 1e-4로 구현 검증.
- 학습: 백본 copy + 분류 헤드(`Linear 768→5`)만 학습, **AdamW + linear warmup + AMP**,
  5 epoch / batch 32 / lr 2e-5 / max_len 128, seed 42. 매 epoch **검증 Macro-F1
  최고점 모델만 저장** → 그걸로 테스트셋 추론. 실행: `src/classifier.ipynb`.
- 평가: Macro-F1 + Confusion Matrix(Plotly) → `outputs/cls_output.json`.

#### 학습 데이터 구축 (2단계: 시드 직접 + LLM 증강)

1. **시드 `train.json` (316건)** — 5개 카테고리 질문을 라벨당 ~64개 직접 작성
   (거의 균등). 검증셋 `valid.json`(271건, 라벨당 ~54) 별도 분리.
2. **증강 `train_augmented.json` (10,085건)** — `src/data/augment_cls.py`로
   원본 질문을 **Claude Haiku(claude-haiku-4-5)로 paraphrase**, 라벨당 ~2,000개.
   - 다양성 규칙: 말투(존/반/구어/문어)·표현·길이 다양화, **오타·줄임말 일부 포함**,
     카테고리 정확성, 원본 동일 금지. temperature 1.0, batch 20, seed 42.
   - 원본 포함 + 공백·소문자 정규화 후 중복 제거.
   - 라벨 분포: 0~2 ≈2,180 / 식단 1,670 / 셔틀 1,879 (큰 불균형 아님).
   - 폴백 `augment_cls_local.py`: API 없이 키워드×패턴×말투 조합 규칙 생성.
- 설계 의도: 1만 개를 손으로 못 만드니 **품질 시드만 직접 + 양은 LLM**.
  말투·오타·줄임말 다양화로 실제 학생 질문 변형에 강한 분류기(F1↑), 라벨 균등으로
  Macro-F1에 맞춤. **Task 2 RAG Q&A 데이터와는 완전 별개** (질문→라벨 vs 질문→답변).

### 공통 주의 (미확인)

- 위 ✅는 **코드가 있다** 기준. `chatbot.sh` 실제 실행 → `chat_output.json` /
  `realtime_output.json` 무에러 생성은 Streamlit·스킬 리팩터링 이후 **재검증 필요**.
