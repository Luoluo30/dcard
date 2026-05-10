import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="SVD and Word2Vec Visualizer",
    page_icon="📊",
    layout="wide",
)


SVD_MATRIX = np.array(
    [
        [8, 7, 1, 0, 0],
        [7, 8, 1, 0, 0],
        [1, 1, 7, 6, 6],
        [0, 0, 6, 7, 7],
        [0, 0, 6, 7, 8],
    ],
    dtype=float,
)

SVD_LABELS = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]

CORPUS_TEXT = """
king queen prince princess royal palace crown throne
man woman boy girl family child parent home
dog puppy cat kitten pet animal home
car bus train bicycle scooter travel road city
apple orange banana fruit juice sweet market
teacher student school lesson classroom book paper
doctor nurse hospital patient clinic health care
king queen palace royal leader kingdom crown
dog cat pet animal friend home
car train bus road travel city station
apple banana orange fruit market fresh sweet
teacher school student lesson classroom study book
doctor hospital nurse patient health clinic care
"""


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)


@st.cache_data
def tokenize_corpus(text: str) -> list[list[str]]:
    sentences = []
    for line in text.strip().splitlines():
        tokens = re.findall(r"[a-z]+", line.lower())
        if tokens:
            sentences.append(tokens)
    return sentences


@st.cache_data
def build_cooccurrence(
    sentences: list[list[str]], window_size: int, min_count: int
) -> tuple[list[str], np.ndarray, Counter]:
    counts = Counter(token for sentence in sentences for token in sentence)
    vocab = sorted([word for word, count in counts.items() if count >= min_count])
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)), dtype=float)

    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in index:
                continue
            left = max(0, i - window_size)
            right = min(len(sentence), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context_word = sentence[j]
                if context_word in index:
                    matrix[index[word], index[context_word]] += 1.0

    return vocab, matrix, counts


def build_word_embeddings(
    cooccurrence: np.ndarray, words: list[str], dimensions: int
) -> pd.DataFrame:
    if len(words) < 3:
        return pd.DataFrame(columns=["word", "x", "y", "size"])

    row_strength = cooccurrence.sum(axis=1)
    weighted = cooccurrence / np.maximum(row_strength[:, None], 1.0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(weighted)
    return pd.DataFrame(
        {
            "word": words,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "size": row_strength,
        }
    )


def render_svd_section() -> None:
    st.header("SVD")
    st.write(
        "SVD 會把矩陣拆成方向與權重。你可以把它想成：找到最重要的幾條軸，"
        "用較少維度重新描述原本的資料。"
    )

    max_rank = min(SVD_MATRIX.shape)
    rank = st.slider("保留前幾個奇異值", 1, max_rank, 2, key="svd_rank")

    u, singular_values, vt = np.linalg.svd(SVD_MATRIX, full_matrices=False)
    truncated = u[:, :rank] @ np.diag(singular_values[:rank]) @ vt[:rank, :]
    explained = singular_values[:rank].sum() / singular_values.sum()

    col1, col2 = st.columns(2)

    with col1:
        original_df = pd.DataFrame(SVD_MATRIX, index=SVD_LABELS, columns=SVD_LABELS)
        fig_original = px.imshow(
            original_df,
            text_auto=".1f",
            color_continuous_scale="Blues",
            title="原始矩陣",
            aspect="auto",
        )
        st.plotly_chart(fig_original, use_container_width=True)

    with col2:
        reconstructed_df = pd.DataFrame(
            truncated, index=SVD_LABELS, columns=SVD_LABELS
        )
        fig_reconstructed = px.imshow(
            reconstructed_df,
            text_auto=".1f",
            color_continuous_scale="Teal",
            title=f"Rank-{rank} 重建矩陣",
            aspect="auto",
        )
        st.plotly_chart(fig_reconstructed, use_container_width=True)

    sigma_df = pd.DataFrame(
        {
            "component": [f"s{i + 1}" for i in range(len(singular_values))],
            "value": singular_values,
        }
    )
    fig_sigma = px.bar(
        sigma_df,
        x="component",
        y="value",
        text="value",
        color="value",
        color_continuous_scale="Sunset",
        title="奇異值大小",
    )
    fig_sigma.update_traces(texttemplate="%{text:.2f}")
    st.plotly_chart(fig_sigma, use_container_width=True)

    error = np.linalg.norm(SVD_MATRIX - truncated)
    st.metric("保留資訊比例", f"{explained:.1%}")
    st.metric("重建誤差", f"{error:.3f}")

    st.caption(
        "當 rank 變小，矩陣會被壓縮得更厲害；當 rank 變大，重建結果會更接近原始資料。"
    )


def render_word2vec_section() -> None:
    st.header("Word2Vec Intuition")
    st.write(
        "Word2Vec 的重點不是背單字，而是從上下文學到語意。"
        "常一起出現的詞，向量通常也會彼此靠近。"
    )

    sentences = tokenize_corpus(CORPUS_TEXT)
    col1, col2, col3 = st.columns(3)
    with col1:
        window_size = st.slider("Context window", 1, 3, 2, key="window_size")
    with col2:
        min_count = st.slider("Minimum count", 1, 3, 1, key="min_count")
    with col3:
        top_k = st.slider("顯示相近詞數量", 3, 8, 5, key="top_k")

    words, cooccurrence, counts = build_cooccurrence(sentences, window_size, min_count)
    if len(words) < 3:
        st.warning("目前詞彙太少，請降低 minimum count。")
        return

    embedding_df = build_word_embeddings(cooccurrence, words, dimensions=2)
    target_word = st.selectbox("觀察哪個詞", words, index=words.index("king") if "king" in words else 0)

    idx = words.index(target_word)
    similarities = []
    for word, vector in zip(words, cooccurrence):
        if word == target_word:
            continue
        similarities.append(
            {
                "word": word,
                "similarity": cosine_similarity(cooccurrence[idx], vector),
                "count": counts[word],
            }
        )
    similarity_df = (
        pd.DataFrame(similarities)
        .sort_values("similarity", ascending=False)
        .head(top_k)
    )

    heatmap_words = words[: min(12, len(words))]
    heatmap_idx = [words.index(word) for word in heatmap_words]
    heatmap_matrix = cooccurrence[np.ix_(heatmap_idx, heatmap_idx)]
    heatmap_df = pd.DataFrame(
        heatmap_matrix, index=heatmap_words, columns=heatmap_words
    )

    plot_df = embedding_df.copy()
    plot_df["highlight"] = np.where(
        plot_df["word"].eq(target_word)
        | plot_df["word"].isin(similarity_df["word"].tolist()),
        "focus",
        "other",
    )

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        fig_embed = px.scatter(
            plot_df,
            x="x",
            y="y",
            text="word",
            size="size",
            color="highlight",
            color_discrete_map={"focus": "#ef553b", "other": "#636efa"},
            title="詞向量的 2D 投影",
        )
        fig_embed.update_traces(textposition="top center")
        st.plotly_chart(fig_embed, use_container_width=True)

    with col_b:
        fig_neighbors = px.bar(
            similarity_df,
            x="similarity",
            y="word",
            orientation="h",
            text="similarity",
            color="similarity",
            color_continuous_scale="Viridis",
            title=f"與 {target_word} 最相近的詞",
        )
        fig_neighbors.update_traces(texttemplate="%{text:.2f}")
        fig_neighbors.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_neighbors, use_container_width=True)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="Magma",
            text=heatmap_df.values,
            texttemplate="%{text:.0f}",
        )
    )
    fig_heatmap.update_layout(title="共現矩陣")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.caption(
        "這裡用共現矩陣近似 Word2Vec 的直覺。真正的 Word2Vec 通常用 skip-gram 或 CBOW "
        "搭配神經網路訓練，但核心精神一樣：從上下文學出密集向量。"
    )


st.title("Visualizing SVD and Word2Vec")
st.write(
    "這個小工具用互動圖把兩個常見概念串起來："
    "SVD 負責找出資料中最重要的方向，Word2Vec 則讓詞在向量空間裡形成語意鄰近。"
)

tab_svd, tab_word2vec = st.tabs(["SVD", "Word2Vec"])

with tab_svd:
    render_svd_section()

with tab_word2vec:
    render_word2vec_section()
