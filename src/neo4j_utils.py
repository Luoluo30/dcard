from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List

import jieba
import numpy as np
from neo4j import GraphDatabase, Transaction


STOPWORDS = {
    "一個", "一些", "一下", "什麼", "但是", "如果", "自己", "沒有", "可以", "因為", "所以",
    "覺得", "真的", "文章", "推薦", "分享", "請問", "想問", "幫我", "尋找", "大家", "有人",
    "這個", "那個", "我們", "你們", "他們", "女生", "男生", "女性", "男性", "異性", "交友",
    "希望", "或是", "或者", "喜歡", "網址", "圖片", "影片", "照片", "年齡", "身高", "體重",
    "興趣", "單身", "地區", "另外", "上傳",
    "dcard", "https", "www", "com", "amp", "nbsp", "span", "class", "color", "gender",
    "width", "height", "style", "image", "img", "src", "jpg", "png", "webp", "border", "lurl",
}


def get_neo4j_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
):
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "NEO4J 密碼不存在，請在 .env 中設定 NEO4J_PASSWORD 或透過函式參數傳入。"
        )

    return GraphDatabase.driver(uri, auth=(user, password))


def _ensure_constraints(session) -> None:
    session.run(
        "CREATE CONSTRAINT article_doc_id IF NOT EXISTS "
        "FOR (a:Article) REQUIRE a.doc_id IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT gender_name IF NOT EXISTS "
        "FOR (g:Gender) REQUIRE g.name IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT topic_name IF NOT EXISTS "
        "FOR (t:Topic) REQUIRE t.name IS UNIQUE"
    )


def _create_article_nodes(tx: Transaction, rows: List[Dict[str, Any]]):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (a:Article {doc_id: row.doc_id})
        SET a.title = row.title,
            a.content = left(row.content, 1000),
            a.articleContent = left(row.content, 1000),
            a.body = row.content,
            a.gender = row.gender,
            a.url = row.url,
            a.scrapeOrder = row.doc_id
        WITH a, row
        WHERE row.gender IS NOT NULL AND row.gender <> ''
        MERGE (g:Gender {gender: row.gender})
        SET g.name = row.gender
        MERGE (a)-[:HAS_GENDER]->(g)
        """,
        rows=rows,
    )


def _extract_topics(title: str, content: str, max_topics: int = 6) -> List[Dict[str, Any]]:
    text = f"{title} {content}"
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff#]+", " ", text)
    tokens = []
    for token in jieba.lcut(text):
        token = token.strip().lower().lstrip("#")
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        if re.fullmatch(r"[a-z_]+", token) and len(token) < 4:
            continue
        tokens.append(token)

    counts = Counter(tokens)
    return [
        {"name": name, "weight": int(weight)}
        for name, weight in counts.most_common(max_topics)
    ]


def _create_topic_nodes(tx: Transaction, rows: List[Dict[str, Any]]):
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (a:Article {doc_id: row.doc_id})
        OPTIONAL MATCH (a)-[old:MENTIONS]->(:Topic)
        DELETE old
        WITH a, row
        UNWIND row.topics AS topic
        MERGE (t:Topic {name: topic.name})
        MERGE (a)-[r:MENTIONS]->(t)
        SET r.weight = topic.weight
        """,
        rows=rows,
    )


def _create_similar_edges(tx: Transaction, rows: List[Dict[str, Any]]):
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (a:Article {doc_id: row.source_doc_id})
        MATCH (b:Article {doc_id: row.target_doc_id})
        WHERE a <> b
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.score = row.score
        """,
        rows=rows,
    )


def _similarity_rows(
    embeddings: np.ndarray,
    top_k: int = 3,
    min_score: float = 0.62,
) -> List[Dict[str, Any]]:
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    scores = vectors @ vectors.T
    np.fill_diagonal(scores, -1.0)

    rows = []
    for source_doc_id in range(scores.shape[0]):
        candidates = np.argpartition(scores[source_doc_id], -top_k)[-top_k:]
        candidates = candidates[np.argsort(scores[source_doc_id][candidates])[::-1]]
        for target_doc_id in candidates:
            score = float(scores[source_doc_id, target_doc_id])
            if score >= min_score:
                rows.append(
                    {
                        "source_doc_id": int(source_doc_id),
                        "target_doc_id": int(target_doc_id),
                        "score": round(score, 4),
                    }
                )
    return rows


def import_articles_to_neo4j(
    driver,
    df,
    embeddings: np.ndarray | None = None,
    content_col: str = "article_content",
    gender_col: str = "gender",
    url_cols: list[str] | None = None,
    topic_limit: int = 6,
    similar_top_k: int = 3,
    similar_min_score: float = 0.62,
):
    if url_cols is None:
        url_cols = ["url", "link"]

    rows = []
    for _, row in df.iterrows():
        url_value = ""
        for col in url_cols:
            if col in row and str(row[col]).strip():
                url_value = str(row[col]).strip()
                break

        title = str(row.get("article_title", "") or "")
        content = str(row.get(content_col, "") or "")
        rows.append(
            {
                "doc_id": int(row["doc_id"]),
                "title": title,
                "content": content,
                "gender": str(row.get(gender_col, "") or ""),
                "url": url_value,
                "topics": _extract_topics(title, content, topic_limit),
            }
        )

    with driver.session() as session:
        _ensure_constraints(session)
        if hasattr(session, "execute_write"):
            session.execute_write(_create_article_nodes, rows)
            session.execute_write(_create_topic_nodes, rows)
            if embeddings is not None:
                session.execute_write(
                    _create_similar_edges,
                    _similarity_rows(embeddings, similar_top_k, similar_min_score),
                )
        else:
            session.write_transaction(_create_article_nodes, rows)
            session.write_transaction(_create_topic_nodes, rows)
            if embeddings is not None:
                session.write_transaction(
                    _create_similar_edges,
                    _similarity_rows(embeddings, similar_top_k, similar_min_score),
                )


def _topic_terms(topic: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(topic).strip())
    terms = []
    for token in jieba.lcut(normalized):
        token = token.strip().lower()
        if len(token) >= 2 and token not in STOPWORDS:
            terms.append(token)

    if normalized and len(normalized) <= 20:
        terms.append(normalized.lower())

    return list(dict.fromkeys(terms))


def query_articles_by_topic(
    driver,
    topic: str,
    limit: int = 10,
    gender: str | None = None,
) -> List[Dict[str, Any]]:
    if not topic:
        return []

    terms = _topic_terms(topic)
    if not terms:
        return []

    cypher = [
        "MATCH (a:Article)",
        "OPTIONAL MATCH (a)-[:MENTIONS]->(topic:Topic)",
        "WITH a, collect(DISTINCT topic.name) AS topics",
        "WHERE ANY(term IN $terms WHERE"
        " toLower(coalesce(a.title, '')) CONTAINS term"
        " OR toLower(coalesce(a.content, '')) CONTAINS term"
        " OR toLower(coalesce(a.body, '')) CONTAINS term"
        " OR ANY(topicName IN topics WHERE toLower(topicName) CONTAINS term))"
    ]
    if gender:
        cypher.append("AND a.gender = $gender")
    cypher.append(
        """
        OPTIONAL MATCH (a)-[sim:SIMILAR_TO]->(similar:Article)
        WITH a, topics, sim, similar
        ORDER BY sim.score DESC
        RETURN a.doc_id AS doc_id,
               a.title AS title,
               a.gender AS gender,
               a.url AS url,
               topics AS topics,
               collect({
                   doc_id: similar.doc_id,
                   title: similar.title,
                   gender: similar.gender,
                   url: similar.url,
                   score: sim.score
               })[0..3] AS similar_articles
        """
    )
    cypher.append("LIMIT $limit")
    query = "\n".join(cypher)

    with driver.session() as session:
        result = session.run(query, terms=terms, gender=gender, limit=limit)
        return [record.data() for record in result]
