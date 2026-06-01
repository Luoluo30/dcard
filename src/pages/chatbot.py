"""
pages/chatbot.py  ── Dcard 語氣串接 AI 聊天機器人

修正點：
- 移除雙層 @st.cache_resource 包裝（原因是 [Errno 22]）
- 直接呼叫 main.py 已快取的 load_store / load_sbert_model
- 無 API Key 時顯示明確提示
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

# ── 加入 src/ 至 import 路徑 ────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent  # .../src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from main import (
    load_store,
    load_sbert_model,
    retrieve_articles,
    DEFAULT_STORE_DIR,
    DEFAULT_MODEL_NAME,
    GOOGLE_GEMINI_API_KEY,
)

# ── 頁面設定 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="語氣聊天機器人 ✦ Dcard",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Dcard 語氣串接聊天機器人")
st.caption("從 Dcard 文章萃取語氣，由 AI 以相同口吻與你對話")

# ── 無 API Key 時提前顯示提示並停止 ─────────────────────────────────────
if not GOOGLE_GEMINI_API_KEY:
    st.error("🔑 需要 Gemini API Key 才能使用聊天機器人")
    st.markdown("""
### 取得免費 API Key（1 分鐘完成）

1. 前往 **[Google AI Studio](https://aistudio.google.com/app/apikey)**（免費，需 Google 帳號）
2. 點選「Create API key」
3. 複製金鑰
4. 用記事本開啟 `C:\\Users\\xshunwei\\dcard_project\\.env`
5. 將內容改為：
```
GOOGLE_GEMINI_API_KEY=貼上你的金鑰
```
6. 存檔後重新整理瀏覽器

> 主頁面的 RRF 搜尋表格**不需要 API Key** 仍可正常使用。
    """)
    st.stop()

# ── 載入向量資料庫（直接呼叫 main.py 已快取版本，避免雙層快取 [Errno 22]）─
try:
    metadata, embeddings, bm25, _corpus, config = load_store(DEFAULT_STORE_DIR)
    sbert_model = load_sbert_model(DEFAULT_MODEL_NAME)
    content_col = config.get("content_col", "article_content")
except Exception as exc:
    st.error(f"❌ 載入向量資料庫失敗：{exc}")
    st.stop()

# ── Gemini client（延遲初始化，避免無 key 時 crash）──────────────────────
@st.cache_resource
def _get_gemini_client():
    from google import genai
    return genai.Client(api_key=GOOGLE_GEMINI_API_KEY)

# ── 側邊欄 ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ 設定")
    gemini_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        index=0,
    )
    top_n_each = st.slider("SBERT/BM25 各取前 N 筆", 5, 50, 15, 5)
    final_top_k = st.slider("RRF 候選篇數", 3, 10, 5, 1)
    rrf_k       = st.slider("RRF k 值", 10, 100, 60, 5)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 重置對話", use_container_width=True):
            for k in ["chat_history", "tone_profile",
                      "source_articles", "articles_text", "current_topic"]:
                st.session_state.pop(k, None)
            st.rerun()
    with c2:
        if st.button("🔀 換語氣", use_container_width=True):
            for k in ["tone_profile", "source_articles",
                      "articles_text", "current_topic"]:
                st.session_state.pop(k, None)
            st.rerun()

    if st.session_state.get("tone_profile"):
        st.divider()
        st.subheader("🎨 語氣特徵")
        st.caption(f"主題：**{st.session_state.get('current_topic', '')}**")
        with st.container(border=True):
            st.write(st.session_state["tone_profile"])

        st.subheader("📄 來源文章")
        for art in st.session_state.get("source_articles", []):
            with st.expander(art["title"] or "（無標題）", expanded=False):
                st.caption(f"RRF：{art['rrf_score']:.5f}")
                st.write(art["preview"] + "…")
                url = art.get("url", "")
                if url and url not in ("無", ""):
                    st.markdown(f"[➡ 原文]({url})")

# ── Session State ────────────────────────────────────────────────────────
for k, v in [("chat_history", []), ("tone_profile", None),
             ("source_articles", []), ("articles_text", ""), ("current_topic", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── 工具函式 ─────────────────────────────────────────────────────────────

def _build_articles_text(results, max_chars: int = 700) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        row = metadata.iloc[r.doc_id]
        title   = row.get("article_title", "")
        content = str(row.get(content_col, ""))[:max_chars].replace("\n", " ")
        parts.append(f"【文章 {i}：{title}】\n{content}")
    return "\n\n".join(parts)


def extract_tone_profile(articles_text: str) -> str:
    from google.genai import types  # noqa: PLC0415
    client = _get_gemini_client()
    prompt = (
        "以下是從 Dcard 平台蒐集的文章內容：\n\n"
        f"{articles_text}\n\n"
        "請仔細分析這些文章共同的語氣特色，用繁體中文描述：\n"
        "1. 整體語感（輕鬆閒聊 / 情緒宣洩 / 認真訴說 / 幽默調侃…）\n"
        "2. 用詞傾向（口語化程度、常見用語）\n"
        "3. 情感表達（直白 / 婉轉、情緒濃度）\n"
        "4. 常用語助詞或語氣詞\n"
        "5. 句子節奏（長短句偏好）\n\n"
        "只輸出語氣描述段落，不需標題或編號。"
    )
    resp = client.models.generate_content(model=gemini_model, contents=prompt)
    return resp.text.strip()


def chat_with_tone(history: List[dict], user_msg: str) -> str:
    from google.genai import types  # noqa: PLC0415
    client = _get_gemini_client()

    system_instruction = (
        "你是一個能深度模仿語氣的聊天夥伴。\n\n"
        "【語氣特徵（從 Dcard 文章分析而來）】\n"
        f"{st.session_state['tone_profile']}\n\n"
        "【語氣來源文章節錄】\n"
        f"{st.session_state['articles_text'][:1800]}\n\n"
        "嚴格依照上述語氣特徵與使用者對話，就像你是那些文章的作者。\n"
        "用繁體中文回應，語氣自然，不要提及「語氣」、「文章」、「Dcard」等字眼。"
    )

    contents = [
        types.Content(role=t["role"], parts=[types.Part(text=t["content"])])
        for t in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

    resp = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.85,
        ),
    )
    return resp.text.strip()


# ── 主介面 ───────────────────────────────────────────────────────────────

# ═══ 階段一：選擇語氣主題 ═══
if st.session_state["tone_profile"] is None:
    st.info("請先輸入主題，系統會從相關 Dcard 文章萃取語氣，再由 AI 以該語氣與你對話。")

    with st.form("tone_form", clear_on_submit=True):
        topic = st.text_input(
            "🔍 語氣主題關鍵字",
            placeholder="例如：分手、暗戀、異地戀、職場壓力、考研、租屋…",
        )
        submitted = st.form_submit_button("擷取語氣並開始對話 →", type="primary",
                                          use_container_width=True)

    if submitted and topic.strip():
        with st.spinner(f"正在從 Dcard 文章分析「{topic}」的語氣…"):
            results = retrieve_articles(
                query=topic,
                model=sbert_model,
                embeddings=embeddings,
                bm25=bm25,
                top_n_each=top_n_each,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
            )
            articles_text  = _build_articles_text(results)
            tone_profile   = extract_tone_profile(articles_text)
            source_articles = [
                {
                    "title":   metadata.iloc[r.doc_id].get("article_title", ""),
                    "preview": str(metadata.iloc[r.doc_id].get(content_col, ""))[:250],
                    "url":     metadata.iloc[r.doc_id].get(
                                   "link", metadata.iloc[r.doc_id].get("url", "")),
                    "rrf_score": r.rrf_score,
                }
                for r in results
            ]

        st.session_state.update({
            "current_topic":   topic,
            "tone_profile":    tone_profile,
            "source_articles": source_articles,
            "articles_text":   articles_text,
            "chat_history":    [],
        })
        st.success(f"✅ 已從 {len(results)} 篇文章萃取語氣，可以開始對話！")
        st.rerun()

# ═══ 階段二：對話介面 ═══
else:
    st.markdown(
        f"**語氣主題：** `{st.session_state['current_topic']}`　｜　"
        f"來源文章 {len(st.session_state['source_articles'])} 篇　｜　"
        "可在左側切換主題或重置"
    )
    st.divider()

    # 對話歷史
    chat_box = st.container(height=480, border=True)
    with chat_box:
        if not st.session_state["chat_history"]:
            st.caption("── 對話開始，AI 將以萃取的語氣回應 ──")
        for msg in st.session_state["chat_history"]:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # 輸入框
    user_input = st.chat_input("輸入任何話題，AI 會以 Dcard 文章語氣回應…")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI 思考中…"):
            try:
                reply = chat_with_tone(
                    history=st.session_state["chat_history"][:-1],
                    user_msg=user_input,
                )
            except Exception as exc:
                reply = f"⚠️ 發生錯誤：{exc}"
        st.session_state["chat_history"].append({"role": "model", "content": reply})
        st.rerun()
