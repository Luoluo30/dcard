"""
build_gender_retrieval_index.py

用途：
1. 讀取 Dcard CSV
2. 只保留有性別資料的文章
3. 將 article_content 放入 SBERT 模型 paraphrase-multilingual-MiniLM-L12-v2 產生語意向量
4. 使用 jieba 斷詞建立 BM25 索引
5. 將處理後資料、向量、BM25 索引與設定檔存起來，供後續推薦系統使用

安裝套件：
pip install pandas numpy sentence-transformers jieba rank-bm25 tqdm

執行範例：
python build_gender_retrieval_index.py \
  --input_csv "dcard_3000_merged_cleaned(3).csv" \
  --output_dir "retrieval_store"
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List

import jieba
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VALID_GENDERS = {"male", "female"}


def normalize_text(text: object) -> str:
    """基本文字清理：轉字串、去除多餘空白。"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_gender(gender: object) -> str:
    """將 gender 欄位標準化，只保留 male / female。"""
    if pd.isna(gender):
        return ""
    gender = str(gender).strip().lower()

    # 兼容舊格式，例如 --color-gender-male 或 --color-gender-female
    if "female" in gender:
        return "female"
    if "male" in gender:
        return "male"
    return gender if gender in VALID_GENDERS else ""


def jieba_tokenize(text: str) -> List[str]:
    """使用 jieba 斷詞，並移除空白 token。"""
    tokens = jieba.lcut(text)
    return [tok.strip() for tok in tokens if tok.strip()]


def load_and_filter_data(
    input_csv: Path,
    content_col: str = "article_content",
    gender_col: str = "gender",
) -> pd.DataFrame:
    """讀取 CSV，只保留有性別且有文章內容的資料。"""
    df = pd.read_csv(input_csv)

    required_cols = {content_col, gender_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必要欄位：{sorted(missing_cols)}。目前欄位：{list(df.columns)}")

    df = df.copy()
    df[gender_col] = df[gender_col].apply(normalize_gender)
    df[content_col] = df[content_col].apply(normalize_text)

    df = df[df[gender_col].isin(VALID_GENDERS)]
    df = df[df[content_col].str.len() > 0]
    df = df.reset_index(drop=True)

    # 建立穩定 doc_id，後續推薦系統可用 doc_id 回查資料
    df.insert(0, "doc_id", np.arange(len(df), dtype=int))
    return df


def build_sbert_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """使用 SBERT 產生文章內容向量。"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype("float32")


def build_bm25_index(texts: List[str]) -> tuple[BM25Okapi, List[List[str]]]:
    """使用 jieba 斷詞後建立 BM25Okapi 索引。"""
    tokenized_corpus = []
    for text in tqdm(texts, desc="Jieba tokenizing"):
        tokenized_corpus.append(jieba_tokenize(text))

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def save_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    tokenized_corpus: List[List[str]],
    output_dir: Path,
    model_name: str,
    content_col: str,
    gender_col: str,
) -> None:
    """儲存推薦系統需要的資料。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    embeddings_path = output_dir / "sbert_embeddings.npy"
    bm25_path = output_dir / "bm25_index.pkl"
    tokens_path = output_dir / "jieba_tokenized_corpus.pkl"
    config_path = output_dir / "config.json"

    df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    np.save(embeddings_path, embeddings)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(tokens_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    config = {
        "model_name": model_name,
        "content_col": content_col,
        "gender_col": gender_col,
        "num_documents": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "files": {
            "metadata": str(metadata_path.name),
            "sbert_embeddings": str(embeddings_path.name),
            "bm25_index": str(bm25_path.name),
            "jieba_tokenized_corpus": str(tokens_path.name),
        },
        "notes": "metadata.doc_id 對應 embeddings 的 row index，也對應 BM25 corpus 的文件順序。",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SBERT + Jieba BM25 retrieval store from gender-labeled Dcard data.")
    parser.add_argument("--input_csv", type=str, required=True, help="輸入 CSV 檔案路徑")
    parser.add_argument("--output_dir", type=str, default="retrieval_store", help="輸出資料夾")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer 模型名稱")
    parser.add_argument("--content_col", type=str, default="article_content", help="文章內容欄位名稱")
    parser.add_argument("--gender_col", type=str, default="gender", help="性別欄位名稱")
    parser.add_argument("--batch_size", type=int, default=32, help="SBERT encode batch size")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    print(f"讀取資料：{input_csv}")
    df = load_and_filter_data(
        input_csv=input_csv,
        content_col=args.content_col,
        gender_col=args.gender_col,
    )

    if len(df) == 0:
        raise ValueError("篩選後沒有任何有性別且有 content 的資料，請檢查 gender 與 article_content 欄位。")

    print("篩選後資料筆數：", len(df))
    print("性別分布：")
    print(df[args.gender_col].value_counts())

    texts = df[args.content_col].tolist()

    print("\n開始建立 SBERT embeddings...")
    embeddings = build_sbert_embeddings(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    print("SBERT embeddings shape:", embeddings.shape)

    print("\n開始建立 Jieba + BM25 index...")
    bm25, tokenized_corpus = build_bm25_index(texts)

    print("\n儲存輸出檔案...")
    save_outputs(
        df=df,
        embeddings=embeddings,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
        output_dir=output_dir,
        model_name=args.model_name,
        content_col=args.content_col,
        gender_col=args.gender_col,
    )

    print("\n完成！輸出位置：", output_dir.resolve())
    print("包含：metadata.csv、sbert_embeddings.npy、bm25_index.pkl、jieba_tokenized_corpus.pkl、config.json")


if __name__ == "__main__":
    main()
