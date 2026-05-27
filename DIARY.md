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

## 진행 현황 요약

| 단계 | 상태 | 산출물 |
|------|------|--------|
| 프로젝트 설정 | 완료 | CLAUDE.md, 디렉터리 구조, 스모크 테스트 |
| 크롤링 | 완료 | 4개 소스, raw JSONL 4건 |
| 전처리 | 완료 | cleaned 841건, chunks 1,745건 |
| Q&A 생성 | 완료 | train 4,745쌍 + eval 100쌍 |
| RAG 구축 | 완료 | bge-m3 임베딩 + ChromaDB 인덱스 |
| 모델 학습 | 완료 | QLoRA 어댑터 생성 |
| 평가 | 구현 완료 | evaluate.py (실행 결과 확인 필요) |
| 앱 UI | 구현 완료 | Gradio UI |
| 제출 노트북 | 작성 완료 | submission.ipynb (통합 테스트 필요) |
