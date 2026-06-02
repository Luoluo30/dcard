"""
Streamlit WebApp: SBERT + Jieba BM25 + RRF + AG2 Gemini Recommendation

在 src/ 內執行：
streamlit run main.py -- --store_dir ../data/vector

.env 範例：
GOOGLE_GEMINI_API_KEY=你的 Gemini API Key
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.code_utils import content_str
from neo4j_utils import get_neo4j_driver, query_articles_by_topic


AGENT_AVATARS = {"User": "👤", "Assistant": "🤖"}

def paging():
    st.caption("Dcard Article Recommender")

def display_session_msg(container):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for msg in st.session_state["messages"]:
        with container.chat_message(msg["role"]):
            st.markdown(msg["content"])

def render_chat_message(container, role, content, name=None, avatar=None):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"role": role, "content": content})
    with container.chat_message(role):
        st.markdown(content)


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

DEFAULT_STORE_DIR = "../data/vector"
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

placeholderstr = "請輸入想找的文章主題，例如：遠距離、曖昧、分手、伴侶溝通"
user_name = "User"
user_image = AGENT_AVATARS.get("User", "👤")


@dataclass
class SearchResult:
    doc_id: int
    rrf_score: float
    sbert_rank: int | None
    bm25_rank: int | None
    sbert_score: float | None
    bm25_score: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", default=DEFAULT_STORE_DIR)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--gemini_model", default=DEFAULT_GEMINI_MODEL)
    return parser.parse_args()


def save_lang():
    st.session_state["lang_setting"] = st.session_state.get("language_select")


@st.cache_resource(show_spinner="載入檢索資料庫中...")
def load_store(store_dir: str):
    store_path = Path(store_dir)

    metadata_path = store_path / "metadata.csv"
    embeddings_path = store_path / "sbert_embeddings.npy"
    bm25_path = store_path / "bm25_index.pkl"
    tokenized_path = store_path / "jieba_tokenized_corpus.pkl"
    config_path = store_path / "config.json"

    required = [metadata_path, embeddings_path, bm25_path, tokenized_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "找不到必要檔案：\n"
            + "\n".join(missing)
            + "\n請先執行 train.py 建立 vector。"
        )

    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(tokenized_path, "rb") as f:
        tokenized_corpus = pickle.load(f)

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    return metadata, embeddings, bm25, tokenized_corpus, config


@st.cache_resource(show_spinner="載入 SBERT 模型中...")
def load_sbert_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="初始化 AG2 Gemini Agent 中...")
def load_ag2_agent(gemini_model: str, lang_setting: str):
    """依照你提供的格式：LLMConfig({...})，不要用 LLMConfig(config_list=...)。"""
    if not GOOGLE_GEMINI_API_KEY:
        raise EnvironmentError(
            "GOOGLE_GEMINI_API_KEY is not set. 請在 webapp/.env 中加入 GOOGLE_GEMINI_API_KEY。"
        )

    llm_config_gemini = LLMConfig({
        "api_type": "google",
        "model": gemini_model,
        "api_key": GOOGLE_GEMINI_API_KEY,
        "temperature": 0.3,
    })

    assistant = AssistantAgent(
        name="Article_Recommender_Agent",
        system_message=(
            "你是 Article_Recommender_Agent，是一個文章推薦系統助理。"
            "你會根據使用者需求與檢索出的候選文章推薦最適合的文章。"
            "不要捏造候選文章以外的內容。"
            "請優先根據文章內容、性別欄位、RRF 分數、SBERT 排名與 BM25 排名說明推薦理由。"
            f"Please output in {lang_setting}. "
            "End your final response with 'ALL DONE'."
        ),
        llm_config=llm_config_gemini,
        max_consecutive_auto_reply=1,
    )

    def is_done_message(x):
        content = x.get("content") if isinstance(x, dict) else ""
        if content is None:
            content = ""
        return "ALL DONE" in str(content)

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=is_done_message,
    )

    return assistant, user_proxy


@st.cache_resource(show_spinner="初始化 Neo4j 連線中...")
def load_neo4j_driver():
    return get_neo4j_driver()


def tokenize_zh(text: str) -> List[str]:
    return [t.strip() for t in jieba.lcut(str(text).strip()) if t.strip()]


def get_top_rank_map(scores: np.ndarray, top_n: int) -> Dict[int, Tuple[int, float]]:
    sorted_indices = np.argsort(scores)[::-1]
    rank_map: Dict[int, Tuple[int, float]] = {}
    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        rank_map[int(idx)] = (rank, float(scores[idx]))
    return rank_map


def reciprocal_rank_fusion(
    sbert_rank_map: Dict[int, Tuple[int, float]],
    bm25_rank_map: Dict[int, Tuple[int, float]],
    rrf_k: int = 60,
    final_top_k: int = 10,
) -> List[SearchResult]:
    candidate_ids = set(sbert_rank_map.keys()) | set(bm25_rank_map.keys())
    results: List[SearchResult] = []

    for doc_id in candidate_ids:
        rrf_score = 0.0
        sbert_rank = sbert_score = bm25_rank = bm25_score = None

        if doc_id in sbert_rank_map:
            sbert_rank, sbert_score = sbert_rank_map[doc_id]
            rrf_score += 1.0 / (rrf_k + sbert_rank)

        if doc_id in bm25_rank_map:
            bm25_rank, bm25_score = bm25_rank_map[doc_id]
            rrf_score += 1.0 / (rrf_k + bm25_rank)

        results.append(
            SearchResult(
                doc_id=doc_id,
                rrf_score=rrf_score,
                sbert_rank=sbert_rank,
                bm25_rank=bm25_rank,
                sbert_score=sbert_score,
                bm25_score=bm25_score,
            )
        )

    return sorted(results, key=lambda x: x.rrf_score, reverse=True)[:final_top_k]


def retrieve_articles(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    bm25,
    top_n_each: int = 30,
    final_top_k: int = 10,
    rrf_k: int = 60,
) -> List[SearchResult]:
    query_embedding = model.encode([query], normalize_embeddings=True)
    sbert_scores = cosine_similarity(query_embedding, embeddings)[0]
    sbert_rank_map = get_top_rank_map(sbert_scores, top_n_each)

    query_tokens = tokenize_zh(query)
    bm25_scores = np.asarray(bm25.get_scores(query_tokens))
    bm25_rank_map = get_top_rank_map(bm25_scores, top_n_each)

    return reciprocal_rank_fusion(
        sbert_rank_map=sbert_rank_map,
        bm25_rank_map=bm25_rank_map,
        rrf_k=rrf_k,
        final_top_k=final_top_k,
    )


def make_candidate_context(metadata: pd.DataFrame, results: List[SearchResult], content_col: str) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        row = metadata.iloc[r.doc_id]
        title = row.get("article_title", "")
        gender = row.get("gender", "")
        url = row.get("link", row.get("url", ""))
        content = str(row.get(content_col, ""))
        preview = content[:800].replace("\n", " ")

        blocks.append(
            f"候選文章 {i}\n"
            f"doc_id: {r.doc_id}\n"
            f"title: {title}\n"
            f"gender: {gender}\n"
            f"url: {url}\n"
            f"rrf_score: {r.rrf_score:.6f}\n"
            f"sbert_rank: {r.sbert_rank}\n"
            f"bm25_rank: {r.bm25_rank}\n"
            f"sbert_score: {r.sbert_score}\n"
            f"bm25_score: {r.bm25_score}\n"
            f"content_preview: {preview}\n"
        )
    return "\n---\n".join(blocks)


def generate_recommendation(assistant, user_proxy, user_query: str, candidate_context: str, lang_setting: str) -> str:
    prompt = f"""
使用者想找的文章主題或需求：
{user_query}

Response language:
{lang_setting}

以下是由 SBERT + BM25 + RRF 找出的候選文章：
{candidate_context}

請完成以下任務：
1. 推薦最適合的 5 篇文章。
2. 嚴格遵守使用者對於性別的要求（如果有的話）。例如，如果使用者提到「我想看女生的文章」，你就只能推薦女生的文章。
3. 每篇都要包含文章標題、原文網址以及說明為什麼適合使用者。
4. 請不要提到 SBERT、BM25、RRF 或任何後端技術名稱，直接說明推薦理由即可。
5. 不要捏造候選文章以外的內容。
6. 如果文章內容不足以判斷，請明確說「內容摘要不足」。

請使用以下格式輸出：

### 推薦文章 1
- 標題：
- 原文網址：
- 推薦理由：

### 推薦文章 2
- 標題：
- 原文網址：
- 推薦理由：
...依此類推，直到推薦文章 5。
""".strip()

    result = user_proxy.initiate_chat(
        recipient=assistant,
        message=prompt,
        max_turns=2,
    )
    return result.summary.replace("ALL DONE", "").strip()


def result_table(metadata: pd.DataFrame, results: List[SearchResult], content_col: str) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(results, start=1):
        row = metadata.iloc[r.doc_id]
        content = str(row.get(content_col, ""))
        rows.append({
            "rank": rank,
            "doc_id": r.doc_id,
            "title": row.get("article_title", ""),
            "gender": row.get("gender", ""),
            "url": row.get("url", row.get("link", "")),
            "rrf_score": round(r.rrf_score, 6),
            "sbert_rank": r.sbert_rank,
            "bm25_rank": r.bm25_rank,
            "sbert_score": None if r.sbert_score is None else round(r.sbert_score, 4),
            "bm25_score": None if r.bm25_score is None else round(r.bm25_score, 4),
            "content_preview": content[:120],
        })
    return pd.DataFrame(rows)


def _short_text(value: object, max_len: int = 42) -> str:
    text = str(value or "")
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def neo4j_result_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    def as_list(value: object) -> List[object]:
        return value if isinstance(value, list) else []

    def similar_text(items: object) -> str:
        if not isinstance(items, list):
            return ""
        labels = []
        for item in items:
            if not isinstance(item, dict) or item.get("doc_id") is None:
                continue
            labels.append(f"#{item.get('doc_id')} ({item.get('score', '')})")
        return ", ".join(labels)

    def reason_text(row: Dict[str, object]) -> str:
        topics = as_list(row.get("topics"))
        topic_part = "、".join(str(t) for t in topics[:3])
        similar_part = similar_text(row.get("similar_articles"))
        parts = []
        if topic_part:
            parts.append(f"主題連結：{topic_part}")
        if similar_part:
            parts.append(f"相似文章：{similar_part}")
        return "；".join(parts)

    return pd.DataFrame(
        [
            {
                "doc_id": row.get("doc_id"),
                "title": row.get("title", ""),
                "gender": row.get("gender", ""),
                "topics": "、".join(str(t) for t in as_list(row.get("topics"))[:5]),
                "similar_articles": similar_text(row.get("similar_articles")),
                "recommendation_reason": reason_text(row),
                "url": row.get("url", ""),
            }
            for row in rows
        ]
    )


def build_neo4j_graph_html(topic: str, rows: List[Dict[str, object]]) -> str:
    visible_rows = rows[:10]
    height = max(420, min(900, 150 + len(visible_rows) * 86))
    query_x = 85
    topic_x = 270
    article_x = 495
    similar_x = 725
    gender_x = 890
    query_y = height // 2
    article_gap = max(62, min(86, (height - 140) // max(1, len(visible_rows) - 1 or 1)))
    article_start_y = query_y - article_gap * (len(visible_rows) - 1) // 2

    topic_names = []
    for row in visible_rows:
        row_topics = row.get("topics")
        row_topics = row_topics if isinstance(row_topics, list) else []
        for topic_name in row_topics[:2]:
            if topic_name and topic_name not in topic_names:
                topic_names.append(str(topic_name))
    topic_names = topic_names[:8]
    topic_positions = {
        name: int((index + 1) * height / (len(topic_names) + 1))
        for index, name in enumerate(topic_names)
    }

    genders = sorted({str(row.get("gender", "") or "unknown") for row in visible_rows})
    gender_positions = {
        gender: int((index + 1) * height / (len(genders) + 1))
        for index, gender in enumerate(genders)
    }

    edges = []
    article_nodes = []
    similar_nodes = []
    for index, row in enumerate(visible_rows):
        y = article_start_y + index * article_gap
        gender = str(row.get("gender", "") or "unknown")
        gender_y = gender_positions[gender]

        raw_topics = row.get("topics")
        raw_topics = raw_topics if isinstance(raw_topics, list) else []
        row_topics = [str(t) for t in raw_topics[:2] if str(t) in topic_positions]
        if row_topics:
            for topic_name in row_topics:
                topic_y = topic_positions[topic_name]
                edges.append(
                    f'<path class="edge-topic" d="M {query_x + 58} {query_y} C 155 {query_y}, 170 {topic_y}, {topic_x - 76} {topic_y}" />'
                )
                edges.append(
                    f'<path class="edge-topic" d="M {topic_x + 76} {topic_y} C 360 {topic_y}, 360 {y}, {article_x - 135} {y}" />'
                )
        else:
            edges.append(
                f'<path d="M {query_x + 58} {query_y} C 245 {query_y}, 330 {y}, {article_x - 135} {y}" />'
            )

        edges.append(
            f'<path class="edge-gender" d="M {article_x + 135} {y} C 685 {y}, 725 {gender_y}, {gender_x - 72} {gender_y}" />'
        )
        raw_similar = row.get("similar_articles")
        raw_similar = raw_similar if isinstance(raw_similar, list) else []
        similar_items = [
            item for item in raw_similar
            if isinstance(item, dict) and item.get("doc_id") is not None
        ][:1]
        if similar_items:
            similar = similar_items[0]
            similar_y = min(height - 38, max(38, y + 34))
            edges.append(
                f'<path class="edge-similar" d="M {article_x + 135} {y} C 620 {y}, 635 {similar_y}, {similar_x - 82} {similar_y}" />'
            )
            similar_nodes.append(
                f"""
                <g class="node similar" transform="translate({similar_x},{similar_y})">
                  <rect x="-82" y="-22" width="164" height="44" rx="10"></rect>
                  <text class="label-main" y="-3">Similar #{html.escape(str(similar.get("doc_id", "")))}</text>
                  <text class="label-sub" y="14">score {html.escape(str(similar.get("score", "")))}</text>
                </g>
                """
            )

        article_nodes.append(
            f"""
            <g class="node article" transform="translate({article_x},{y})">
              <rect x="-135" y="-27" width="270" height="54" rx="11"></rect>
              <text class="label-main" y="-5">Article #{html.escape(str(row.get("doc_id", "")))}</text>
              <text class="label-sub" y="14">{html.escape(_short_text(row.get("title", ""), 31))}</text>
            </g>
            """
        )

    topic_nodes = [
        f"""
        <g class="node topic" transform="translate({topic_x},{y})">
          <rect x="-76" y="-24" width="152" height="48" rx="11"></rect>
          <text class="label-main" y="-3">Topic</text>
          <text class="label-sub" y="15">{html.escape(_short_text(name, 16))}</text>
        </g>
        """
        for name, y in topic_positions.items()
    ]

    gender_nodes = [
        f"""
        <g class="node gender" transform="translate({gender_x},{y})">
          <rect x="-72" y="-25" width="144" height="50" rx="11"></rect>
          <text class="label-main" y="-4">Gender</text>
          <text class="label-sub" y="14">{html.escape(_short_text(gender, 18))}</text>
        </g>
        """
        for gender, y in gender_positions.items()
    ]

    return f"""
    <div class="neo4j-graph">
      <svg viewBox="0 0 980 {height}" role="img" aria-label="Neo4j article relationship graph">
        <defs>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#0f172a" flood-opacity="0.14"/>
          </filter>
        </defs>
        <g class="edges">{"".join(edges)}</g>
        <g class="node query" transform="translate({query_x},{query_y})">
          <circle r="58"></circle>
          <text class="label-main" y="-5">Query</text>
          <text class="label-sub" y="16">{html.escape(_short_text(topic, 14))}</text>
        </g>
        {"".join(topic_nodes)}
        {"".join(article_nodes)}
        {"".join(gender_nodes)}
        {"".join(similar_nodes)}
      </svg>
    </div>
    <style>
      .neo4j-graph {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        background: #f8fafc;
      }}
      .neo4j-graph svg {{
        min-width: 920px;
        display: block;
      }}
      .neo4j-graph .edges path {{
        fill: none;
        stroke: #94a3b8;
        stroke-width: 2;
      }}
      .neo4j-graph .edges .edge-topic {{ stroke: #38bdf8; }}
      .neo4j-graph .edges .edge-gender {{ stroke: #f59e0b; }}
      .neo4j-graph .edges .edge-similar {{ stroke: #a78bfa; stroke-dasharray: 5 5; }}
      .neo4j-graph .node rect,
      .neo4j-graph .node circle {{
        filter: url(#shadow);
        stroke-width: 2;
      }}
      .neo4j-graph .query circle {{ fill: #dbeafe; stroke: #3b82f6; }}
      .neo4j-graph .topic rect {{ fill: #e0f2fe; stroke: #0284c7; }}
      .neo4j-graph .article rect {{ fill: #dcfce7; stroke: #22c55e; }}
      .neo4j-graph .gender rect {{ fill: #fef3c7; stroke: #f59e0b; }}
      .neo4j-graph .similar rect {{ fill: #ede9fe; stroke: #8b5cf6; }}
      .neo4j-graph text {{
        text-anchor: middle;
        font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif;
        fill: #0f172a;
      }}
      .neo4j-graph .label-main {{
        font-size: 13px;
        font-weight: 700;
      }}
      .neo4j-graph .label-sub {{
        font-size: 11px;
      }}
    </style>
    """


def render_neo4j_results(topic: str, rows: List[Dict[str, object]]) -> None:
    st.subheader("Neo4j 文章檢索結果")
    components.html(build_neo4j_graph_html(topic, rows), height=max(400, min(820, 150 + len(rows) * 72)))

    df_neo4j = neo4j_result_table(rows)
    st.dataframe(
        df_neo4j,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url"),
        },
    )


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Dcard Article Recommender - AG2 Gemini",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "SBERT + BM25 + RRF + AG2 Gemini recommendation webapp",
        },
        page_icon="🧭",
    )

    st.title("Dcard Article Recommender")

    with st.sidebar:
        paging()

        selected_lang = st.selectbox(
            "Language",
            ["Traditional Chinese", "English"],
            index=0,
            on_change=save_lang,
            key="language_select",
        )
        lang_setting = st.session_state.get("lang_setting", selected_lang)
        st.session_state["lang_setting"] = lang_setting


        st.divider()
        st.subheader("Retrieval Settings")
        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ],
            index=0,
        )
        top_n_each = st.slider("SBERT/BM25 各取前 N 筆", 10, 100, 30, 5)
        final_top_k = st.slider("RRF 最後候選文章數", 3, 20, 10, 1)
        rrf_k = st.slider("RRF k 值", 10, 100, 60, 5)
        use_neo4j = st.checkbox("啟用 Neo4j 文章檢索擴展", value=False)
        neo4j_gender = st.selectbox(
            "Neo4j 性別過濾",
            ["不過濾", "female", "male"],
            index=0,
        )
        use_ag2 = st.checkbox("使用 AG2 產生推薦理由", value=True)

    st_c_chat = st.container(border=True)
    display_session_msg(st_c_chat)

    if not GOOGLE_GEMINI_API_KEY:
        st.warning("GOOGLE_GEMINI_API_KEY is not set. 請在 webapp/.env 加入 GOOGLE_GEMINI_API_KEY。")

    try:
        store_dir = args.store_dir
        metadata, embeddings, bm25, tokenized_corpus, config = load_store(store_dir)
        model = load_sbert_model(args.model_name)
        content_col = config.get("content_col", "article_content")
        st.success(f"已載入 {len(metadata):,} 筆有性別資料的文章。")
    except Exception as e:
        st.error(str(e))
        st.stop()

    def chat(prompt: str):
        render_chat_message(st_c_chat, "user", prompt, name="User", avatar=user_image)

        with st.spinner("正在計算 SBERT、BM25 與 RRF..."):
            results = retrieve_articles(
                query=prompt,
                model=model,
                embeddings=embeddings,
                bm25=bm25,
                top_n_each=top_n_each,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
            )
            df_results = result_table(metadata, results, content_col)

        st.subheader("RRF 檢索結果")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        if use_neo4j:
            try:
                with st.spinner("從 Neo4j 查詢相關文章..."):
                    gender_filter = None if neo4j_gender == "不過濾" else neo4j_gender
                    neo4j_driver = load_neo4j_driver()
                    neo4j_results = query_articles_by_topic(
                        neo4j_driver,
                        prompt,
                        limit=final_top_k,
                        gender=gender_filter,
                    )
            except Exception as e:
                neo4j_results = []
                st.warning(f"Neo4j 查詢失敗：{e}")

            if neo4j_results:
                render_neo4j_results(prompt, neo4j_results)
            else:
                st.info("Neo4j 尚未找到符合查詢的文章。")

        if use_ag2:
            try:
                with st.spinner("AG2 Gemini 正在產生推薦理由..."):
                    assistant, user_proxy = load_ag2_agent(gemini_model, lang_setting)
                    candidate_context = make_candidate_context(metadata, results, content_col)
                    response = generate_recommendation(
                        assistant=assistant,
                        user_proxy=user_proxy,
                        user_query=prompt,
                        candidate_context=candidate_context,
                        lang_setting=lang_setting,
                    )
            except Exception as e:
                response = f"AG2 產生推薦理由失敗：{e}\n\n你仍然可以先使用上方 RRF 表格檢查推薦結果。"
        else:
            response = "已完成 RRF 檢索。你可以查看上方表格中的候選文章。"

        render_chat_message(
            st_c_chat,
            "assistant",
            response,
            name="Article_Recommender_Agent",
        )

        with st.expander("查看候選文章完整內容"):
            for r in results:
                row = metadata.iloc[r.doc_id]
                url = row.get("link", row.get("url", ""))
                st.markdown(f"### doc_id {r.doc_id}｜{row.get('article_title', '')}")
                st.write(f"gender: {row.get('gender', '')}")
                if url and url != "無":
                    st.markdown(f"[原文網址]({url})")
                else:
                    st.write("原文網址：無")
                st.write(f"RRF: {r.rrf_score:.6f}｜SBERT rank: {r.sbert_rank}｜BM25 rank: {r.bm25_rank}")
                st.write(str(row.get(content_col, "")))
                st.divider()

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)


if __name__ == "__main__":
    main()
