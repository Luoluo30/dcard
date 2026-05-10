from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec


st.set_page_config(
    page_title="Dcard Keyword Expander",
    page_icon="🔎",
    layout="wide",
)


DATA_CANDIDATES = [
    Path(r"C:\Users\張博瀚\Desktop\webapp\dcard(1) (1).xlsx"),
    Path(r"C:\Users\張博瀚\Downloads\dcard(1) (1).xlsx"),
    Path(r"C:\Users\張博瀚\Downloads\dcard(1).xlsx"),
    Path(r"C:\Users\張博瀚\Downloads\dcard.xlsx"),
]

SPLIT_KEYWORDS = [
    "【擇偶條件】",
    "希望你",
    "希望妳",
    "理想對象條件",
    "理想對象",
    "理想類型",
    "期望對象",
    "【希望你】",
    "【希望妳】",
    "期望對象條件",
    "關於你",
    "關於妳",
    "擇偶條件",
    "希望的妳",
    "希望的你",
    "希望的對象",
    "希望認識的妳",
    "理想的對象",
    "理想中的妳",
    "想找的對象",
    "找尋對象",
    "對象條件",
    "我的條件",
    "【理想型】",
    "【基本條件】",
    "《期待理想中的你》",
    "期望你",
    "期待你",
    "尋找你",
    "想找",
]

CUSTOM_WORDS = [
    "情緒穩定",
    "價值觀",
    "分享欲",
    "生活圈",
    "軟體工程師",
    "硬體工程師",
    "科技業",
    "醫學系",
    "公務員",
    "佛系",
    "慢熱",
    "以結婚為前提",
    "長期穩定",
    "健身房",
    "有在健身",
    "飲食控制",
    "大台北",
    "雙北",
    "台北市",
    "羽球",
    "爬山",
    "打球",
    "潛水",
]

STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "有",
    "和",
    "也",
    "都",
    "不",
    "一",
    "會",
    "很",
    "就",
    "但",
    "或",
    "與",
    "等",
    "這",
    "那",
    "所以",
    "如果",
    "因為",
    "可以",
    "希望",
    "自己",
    "覺得",
    "喜歡",
    "比較",
    "不是",
    "沒有",
    "雖然",
    "然後",
    "還是",
    "可能",
    "一些",
    "對方",
    "認識",
    "聊天",
    "私訊",
    "留言",
    "交往",
    "感情",
    "相處",
    "朋友",
    "開始",
    "一起",
    "找到",
    "關於",
    "以上",
    "以下",
    "如下",
    "補充",
    "更新",
    "自己",
    "目前",
    "真的",
    "有點",
    "一點",
    "那種",
    "感覺",
    "生活",
    "工作",
    "興趣",
    "年齡",
    "個性",
}


def resolve_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("找不到 Dcard Excel 資料檔。")


def init_jieba() -> None:
    if st.session_state.get("_jieba_initialized"):
        return
    for word in CUSTOM_WORDS:
        jieba.add_word(word)
    st.session_state["_jieba_initialized"] = True


def split_content(text: str) -> tuple[str, str]:
    for keyword in SPLIT_KEYWORDS:
        idx = text.find(keyword)
        if idx > 50 and (idx == 0 or text[idx - 1] in "\n\r 　"):
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_text(text: str, stopwords: set[str] | None = None) -> list[str]:
    init_jieba()
    active_stopwords = STOPWORDS if stopwords is None else stopwords
    tokens: list[str] = []
    for token in jieba.cut(clean_text(text)):
        token = token.strip()
        if not token:
            continue
        if token in active_stopwords:
            continue
        if len(token) < 2:
            continue
        if re.fullmatch(r"[\d\W_]+", token):
            continue
        tokens.append(token)
    return tokens


@st.cache_data(show_spinner=False)
def load_posts() -> pd.DataFrame:
    data_path = resolve_data_path()
    df = pd.read_excel(data_path, engine="openpyxl")
    df["title"] = df["title"].fillna("").map(clean_text)
    df["content"] = df["content"].fillna("").map(clean_text)
    split_cols = df["content"].map(split_content)
    df["self_intro"] = split_cols.map(lambda x: x[0])
    df["expect"] = split_cols.map(lambda x: x[1])
    df["gender"] = df["gender"].fillna("未提供").map(clean_text)
    df["search_text"] = (
        df["title"] + " " + df["self_intro"] + " " + df["expect"]
    ).map(clean_text)
    df["tokens"] = df["search_text"].map(segment_text)
    return df


@st.cache_resource(show_spinner=True)
def build_word2vec_model() -> tuple[Word2Vec, Counter]:
    df = load_posts()
    sentences = [tokens for tokens in df["tokens"] if len(tokens) >= 2]
    if not sentences:
        raise ValueError("資料斷詞後沒有足夠內容可訓練 Word2Vec。")

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        workers=1,
        epochs=80,
        seed=42,
    )
    frequencies = Counter(token for sentence in sentences for token in sentence)
    return model, frequencies


def expand_keywords(
    query: str, model: Word2Vec, frequencies: Counter, topn: int = 8
) -> tuple[list[str], list[dict[str, float | str]]]:
    base_tokens = list(dict.fromkeys(segment_text(query, stopwords=set())))
    scores: dict[str, float] = {}

    for token in base_tokens:
        if token in model.wv:
            for word, similarity in model.wv.most_similar(token, topn=topn * 2):
                if word == token:
                    continue
                if len(word) < 2:
                    continue
                scores[word] = max(scores.get(word, 0.0), float(similarity))

    vocab_matches = []
    for word, count in frequencies.most_common():
        if any(token in word or word in token for token in base_tokens):
            if word not in scores and word not in base_tokens and len(word) >= 2:
                vocab_matches.append(word)
        if len(vocab_matches) >= topn:
            break

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    suggestions = [word for word, _ in ranked[:topn]]
    for word in vocab_matches:
        if word not in suggestions:
            suggestions.append(word)
        if len(suggestions) >= topn:
            break

    detail_rows = [
        {
            "keyword": word,
            "similarity": round(score, 3),
            "frequency": int(frequencies.get(word, 0)),
        }
        for word, score in ranked[:topn]
    ]
    return suggestions, detail_rows


def search_posts(
    df: pd.DataFrame, query: str, selected_keywords: list[str]
) -> pd.DataFrame:
    query_tokens = segment_text(query, stopwords=set())
    all_terms = list(
        dict.fromkeys([query.strip(), *query_tokens, *selected_keywords])
    )
    all_terms = [term for term in all_terms if term]

    records: list[dict[str, object]] = []
    for row in df.itertuples():
        token_set = set(row.tokens)
        raw_text = row.search_text
        matched_terms = [
            term for term in all_terms if term in token_set or term in raw_text
        ]
        if not matched_terms:
            continue

        score = 0
        for term in matched_terms:
            if term in token_set:
                score += 2
            if term in raw_text:
                score += 1

        records.append(
            {
                "title": row.title,
                "link": row.link,
                "gender": row.gender,
                "content": row.content,
                "self_intro": row.self_intro,
                "expect": row.expect,
                "match_count": len(set(matched_terms)),
                "match_score": score,
                "matched_terms": "、".join(dict.fromkeys(matched_terms)),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "title",
                "link",
                "gender",
                "content",
                "self_intro",
                "expect",
                "match_count",
                "match_score",
                "matched_terms",
            ]
        )

    result_df = pd.DataFrame(records)
    return result_df.sort_values(
        ["match_score", "match_count"], ascending=[False, False]
    ).reset_index(drop=True)


def render_result_card(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown(f"### {row['title'] or '未命名貼文'}")
        st.write(f"性別：{row['gender']}  |  命中關鍵字：{row['matched_terms']}")
        if row["self_intro"]:
            st.caption("自我介紹")
            st.write(row["self_intro"][:300] + ("..." if len(row["self_intro"]) > 300 else ""))
        if row["expect"]:
            st.caption("期望對象")
            st.write(row["expect"][:220] + ("..." if len(row["expect"]) > 220 else ""))
        if row["link"]:
            st.markdown(f"[查看原文]({row['link']})")


def main() -> None:
    st.title("Dcard 條件關鍵字擴充與聯想器")
    st.write(
        "輸入你想找的條件，例如 `運動`、`健身`、`旅行`，系統會先用 Word2Vec "
        "找出語意相近的詞，再讓你用 `multiselect` 決定哪些擴充條件要加入搜尋。"
    )

    try:
        posts = load_posts()
        model, frequencies = build_word2vec_model()
    except Exception as exc:
        st.error(f"資料或模型載入失敗：{exc}")
        return

    with st.sidebar:
        st.subheader("資料設定")
        st.write(f"資料筆數：{len(posts)}")
        st.write(f"詞彙量：{len(model.wv)}")
        gender_options = ["全部", *sorted(posts["gender"].dropna().unique().tolist())]
        gender_filter = st.selectbox("性別篩選", gender_options)
        result_limit = st.slider("顯示幾筆結果", 5, 50, 10, 5)

    filtered_posts = posts.copy()
    if gender_filter != "全部":
        filtered_posts = filtered_posts[filtered_posts["gender"] == gender_filter].copy()

    query = st.text_input(
        "輸入想找的特質或興趣",
        placeholder="例如：運動、健身、旅行、慢熱",
    ).strip()

    if not query:
        hot_terms = [word for word, _ in frequencies.most_common(20)]
        st.info("先輸入一個條件，系統才會幫你擴充相似詞。")
        st.caption("目前語料中的高頻詞")
        st.write("、".join(hot_terms))
        return

    suggestions, detail_rows = expand_keywords(query, model, frequencies)
    default_selected = suggestions[: min(5, len(suggestions))]

    selected_keywords = st.multiselect(
        "Word2Vec 推薦的相似詞",
        options=suggestions,
        default=default_selected,
        help="你可以取消不想加入的詞，也可以只保留最符合需求的幾個。",
    )

    if detail_rows:
        with st.expander("查看相似詞分數"):
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
    else:
        st.warning("這個詞目前在語料中太少見，沒有找到穩定的 Word2Vec 相似詞，會先用原始關鍵字搜尋。")

    result_df = search_posts(filtered_posts, query, selected_keywords)

    term_summary = [query, *selected_keywords]
    st.write(f"目前搜尋條件：{'、'.join(dict.fromkeys(term_summary))}")
    st.metric("符合條件的貼文數", len(result_df))

    if result_df.empty:
        st.warning("找不到符合條件的貼文，可以改試更常見的詞，例如 `運動`、`健身`、`旅遊`。")
        return

    for _, row in result_df.head(result_limit).iterrows():
        render_result_card(row)


if __name__ == "__main__":
    main()
