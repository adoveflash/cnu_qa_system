"""박스의 최신 벡터DB·코퍼스를 HF Hub에 업로드.

Colab 노트북들(submission.ipynb, gemma3_submission.ipynb)이 snapshot_download로
받아쓰는 소스를 갱신한다. 오늘 복구·병합한 벡터DB를 올려야 Colab에서도 반영됨.

사전: HF write 토큰으로 로그인돼 있어야 함 (huggingface-cli login).
사용법: python upload_vectordb.py
"""

from __future__ import annotations

from huggingface_hub import HfApi

HF_REPO = "adoveflash/cnu-qa-system"


def main() -> None:
    api = HfApi()
    print(f"업로드 대상: {HF_REPO}")

    print("[1/2] data/vector_db 업로드...")
    api.upload_folder(
        folder_path="data/vector_db",
        path_in_repo="data/vector_db",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="update vector_db (복구·병합본)",
    )

    print("[2/2] data/corpus/chunks.jsonl 업로드...")
    api.upload_file(
        path_or_fileobj="data/corpus/chunks.jsonl",
        path_in_repo="data/corpus/chunks.jsonl",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="update chunks.jsonl",
    )

    print("✅ 업로드 완료 — 이제 Colab 노트북이 최신 DB를 받습니다.")


if __name__ == "__main__":
    main()
