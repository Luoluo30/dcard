"""
pages/chatbot.py  ── Dcard 語氣串接 AI 聊天機器人

修正點：
- 移除雙層 @st.cache_resource 包裝（原因是 [Errno 22]）
- 直接呼叫 main.py 已快取的 load_store / load_sbert_model
- 無 API Key 時顯示明確提示
- 支援自動語言偵測（打英文回英文、打中文回中文）+ 手動切換
"""

from __future__ import annotations

import re
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
    lang_mode = st.radio(
        "回應語言",
        ["🔍 自動偵測（依輸入語言）", "🇹🇼 繁體中文", "🇺🇸 English"],
        index=0,
        help="自動偵測：打中文就回中文，打英文就回英文",
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

def _detect_lang(text: str) -> str:
    """
    簡易語言偵測：計算中文字比例。
    > 15% 中文字 → 'zh'（繁中回應）
    否則          → 'en'（英文回應）
    """
    chinese_chars = len(re.findall(r'[一-鿿㐀-䶿]', text))
    ratio = chinese_chars / max(len(text.strip()), 1)
    return "zh" if ratio > 0.15 else "en"


def _resolve_lang(user_msg: str) -> str:
    """依 lang_mode 側邊欄設定決定本輪回應語言。"""
    if "English" in lang_mode:
        return "en"
    if "繁體中文" in lang_mode:
        return "zh"
    return _detect_lang(user_msg)   # 自動偵測


def _translate_to_zh(text: str) -> str:
    """
    若輸入為英文，用 Gemini 翻譯成繁體中文後回傳，
    供 Jieba / BM25 使用；已是中文則直接回傳原文。
    使用 thinking_budget=0 避免思考過程混入回應。
    """
    if _detect_lang(text) == "zh":
        return text
    from google.genai import types as _t
    client = _get_gemini_client()
    prompt = (
        "請將以下關鍵字或句子翻譯成繁體中文，"
        "只輸出翻譯結果，不需要任何說明或標點補充：\n"
        f"{text}"
    )
    resp = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=_t.GenerateContentConfig(
            temperature=0,
            thinking_config=_t.ThinkingConfig(thinking_budget=0),  # 關閉思考，避免輸出雜訊
        ),
    )
    return resp.text.strip()


def _build_articles_text(results, max_chars: int = 1200) -> str:
    """每篇文章取更多原文（1200字），讓語氣分析有足夠素材。"""
    parts = []
    for i, r in enumerate(results, 1):
        row = metadata.iloc[r.doc_id]
        title   = row.get("article_title", "")
        content = str(row.get(content_col, ""))[:max_chars].replace("\n", " ")
        parts.append(f"【文章 {i}：{title}】\n{content}")
    return "\n\n".join(parts)


def extract_tone_profile(articles_text: str, topic: str = "") -> str:
    """
    從原文萃取具體、可操作的語氣特徵（顯示給使用者看）：
    - 標誌性句型（引用原句）
    - 核心情緒與語氣
    - 仿寫示範（few-shot 錨點）
    口頭禪與禁止詞由 system prompt 內部處理，不對外顯示。
    """
    client = _get_gemini_client()
    topic_hint = f"（主題：{topic}）" if topic else ""
    prompt = (
        f"以下是從 Dcard 平台蒐集{topic_hint}的文章原文：\n\n"
        f"{articles_text}\n\n"
        f"請深度分析上述文章中與「{topic or '此主題'}」情緒直接相關的語氣，"
        "必須從原文找具體依據，用繁體中文輸出以下三點：\n\n"
        "1.【標誌性句型】從原文中找出 4-5 個**能直接反映主題情緒**的句子（盡量直接引用原文）。"
        "⚠️ 嚴格過濾：身高體重、外貌描述、自我介紹條件、找對象條件等與情緒無關的內容一律不引用。\n\n"
        "2.【核心情緒與語氣】用 2 句話說明：①這些人說話時情緒有多濃、節奏快還是慢；"
        "②他們習慣用什麼方式表達這個情緒（自嘲？傾訴？幽默帶過？）。"
        "禁止出現「口語化」「自然」「真誠」「展現」「營造」這類空洞詞。\n\n"
        "3.【仿寫示範】假設有人傳訊息說「你最近怎樣」，寫出 3 種這批文章作者會回覆的方式，"
        f"回覆要明顯帶有「{topic or '此主題'}」的情緒底色，像真人在手機傳訊息，1-3 句，不要像 AI 在寫作文。\n\n"
        "只輸出上述三點，格式清晰即可。"
    )
    from google.genai import types as _types
    resp = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=_types.GenerateContentConfig(temperature=0.3),  # 低隨機性，結果較穩定
    )
    return resp.text.strip()


def chat_with_tone(history: List[dict], user_msg: str):
    """回傳文字串流 generator，供 st.write_stream() 使用。"""
    from google.genai import types  # noqa: PLC0415
    client = _get_gemini_client()

    lang = _resolve_lang(user_msg)
    if lang == "en":
        lang_instruction = (
            "IMPORTANT: The user typed in English — respond entirely in English. "
            "Mirror the same raw emotional tone from the articles but in English. "
            "Keep it short, casual, like texting a friend. No AI-speak."
        )
    else:
        lang_instruction = "用繁體中文回覆，像在傳訊息，不要像 AI 在寫文章。"

    raw_excerpt = st.session_state["articles_text"][:4000]
    current_topic = st.session_state.get("current_topic", "").split("（")[0].strip()

    system_instruction = (
        "━━ 你的身份 ━━\n"
        "你是一個正在和朋友聊天的真實人類。"
        f"你不是在「扮演一個{current_topic}的人」，也不是在模擬情境——"
        f"你只是一個說話語氣、節奏、用詞天生帶有「{current_topic}」色彩的朋友。"
        "對方說什麼，你就正常回應什麼，只是你的語氣骨子裡自然帶著這個調性。\n\n"
        "━━ 語氣原始素材（直接從這裡學習句型與措辭）━━\n"
        f"{raw_excerpt}\n\n"
        "━━ 語氣分析（具體特徵，必須遵守）━━\n"
        f"{st.session_state['tone_profile']}\n\n"
        "━━ 絕對禁止 ━━\n"
        "• AI 腔一律禁止：當然、確實、非常好、值得注意、當然可以、沒問題、很棒、絕對、必須承認\n"
        "• 禁止條列（不用 1. 2. 3. 或 •）\n"
        "• 每次回覆 1-4 句，像在手機傳訊息，不要寫作文\n"
        "• 要有情緒，不要像客服\n"
        "• 不提「語氣」「文章」「Dcard」「分析」\n\n"
        f"━━ 語言 ━━\n{lang_instruction}"
    )

    contents = [
        types.Content(role=t["role"], parts=[types.Part(text=t["content"])])
        for t in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

    stream = client.models.generate_content_stream(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=1.0,
        ),
    )

    def _gen():
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    return _gen()


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
            # ── 英文→繁中翻譯，讓 Jieba/BM25 能正確分詞 ──────────────
            topic_zh = _translate_to_zh(topic.strip())
            if topic_zh != topic.strip():
                st.info(f"🔄 已將「{topic}」翻譯為「{topic_zh}」進行搜尋")

            results = retrieve_articles(
                query=topic_zh,
                model=sbert_model,
                embeddings=embeddings,
                bm25=bm25,
                top_n_each=top_n_each,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
            )
            articles_text  = _build_articles_text(results)
            tone_profile   = extract_tone_profile(articles_text, topic=topic_zh)
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
            "current_topic":   f"{topic}（{topic_zh}）" if topic_zh != topic.strip() else topic,
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

        # 直接在 chat_box 裡串流顯示，不用 spinner
        with chat_box:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            with st.chat_message("model", avatar="🤖"):
                try:
                    reply = st.write_stream(
                        chat_with_tone(
                            history=st.session_state["chat_history"][:-1],
                            user_msg=user_input,
                        )
                    )
                except Exception as exc:
                    reply = f"⚠️ 發生錯誤：{exc}"
                    st.markdown(reply)

        st.session_state["chat_history"].append({"role": "model", "content": reply})
        st.rerun()
