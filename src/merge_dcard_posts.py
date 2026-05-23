"""
合併 Dcard 文章資料

功能：
1. 讀取第一個主檔 dcard_3000.csv。
2. 讀取後面六個文章檔 dcard_posts_part_001.csv ~ dcard_posts_part_006.csv。
3. 依照主檔的 link 與文章檔的 web_scraper_start_url 進行對應合併。
4. 將文章標題 title、清理後內容 content、性別 gender 加到主檔後面。
5. 清理 content 開頭重複的標題、暱稱、發文時間、已編輯等資訊。
6. 從 author_or_gender_area 中擷取 male / female。
7. 輸出新的 CSV 檔。
"""

from pathlib import Path
import re
import pandas as pd


# ====== 檔案設定 ======
DATA_DIR = Path(".")
BASE_FILE = DATA_DIR / "dcard_3000.csv"
POST_FILES = [DATA_DIR / "sitemaps_outputs" / f"dcard_posts_part_{i:03d}.csv" for i in range(1, 7)]
OUTPUT_FILE = DATA_DIR / "dcard_3000_merged_cleaned.csv"


# ====== 清理函式 ======
def extract_gender(raw_value):
    """從 CSS 字串中擷取 male 或 female。沒有找到則回傳空值。"""
    if pd.isna(raw_value):
        return pd.NA

    text = str(raw_value)
    match = re.search(r"color-gender-(male|female)", text)
    if match:
        return match.group(1)
    return pd.NA


def clean_post_content(content, title):
    """
    清理文章內容開頭常見的重複資訊：
    - 重複的文章標題
    - 暱稱
    - 追蹤字樣
    - 發文時間，例如：5 月 17 日 08:54、昨天 13:49、25/12/10 10:33
    - (已編輯)

    注意：Dcard 抓下來的內容格式不完全一致，所以此函式採取保守清理。
    """
    if pd.isna(content):
        return pd.NA

    text = str(content).replace("\r\n", "\n").replace("\r", "\n").strip()
    title_text = "" if pd.isna(title) else str(title).strip()

    # 1) 移除開頭重複標題；若重複出現多次，也一起移除。
    if title_text:
        while text.startswith(title_text):
            text = text[len(title_text):].lstrip()

    # 2) 移除「暱稱 + 追蹤 + 日期時間 + 已編輯」。
    date_time = r"(?:昨天|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{2,4}/\d{1,2}/\d{1,2})\s*\d{1,2}:\d{2}"
    metadata_patterns = [
        rf"^.*?追蹤\s*{date_time}\s*(?:\(已編輯\))?",
        rf"^.*?{date_time}\s*(?:\(已編輯\))?",
    ]
    for pattern in metadata_patterns:
        cleaned = re.sub(pattern, "", text, count=1, flags=re.S)
        if cleaned != text:
            text = cleaned.lstrip()
            break

    # 3) 少數資料沒有日期時間，但標題後仍接暱稱或問候語。
    #    若第一行很短，且下一行看起來像正文起點，就移除第一行。
    first_line, separator, rest = text.partition("\n")
    if separator and len(first_line.strip()) <= 40:
        rest_stripped = rest.lstrip()
        body_start_pattern = (
            r"^(?:[【\[（(《［]|年齡|年紀|身高|身高體重|地點|座標|地區|職業|"
            r"關於我|自我介紹|個人資料|哈囉|嗨|小弟|大家好)"
        )
        profile_line_pattern = (
            r"^(?:[【\[（(《［]|年齡|年紀|身高|身高體重|地點|座標|地區|職業|"
            r"關於我|自我介紹|個人資料)"
        )
        if re.match(body_start_pattern, rest_stripped) and not re.match(profile_line_pattern, first_line.strip()):
            text = rest_stripped

    return text.strip()


def main():
    # 讀取主檔
    base_df = pd.read_csv(BASE_FILE)

    # 讀取並合併六個文章檔
    post_dfs = []
    for file_path in POST_FILES:
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案：{file_path}")
        post_dfs.append(pd.read_csv(file_path))

    posts_df = pd.concat(post_dfs, ignore_index=True)

    # 檢查必要欄位
    required_base_cols = {"link"}
    required_post_cols = {"web_scraper_start_url", "title", "content", "author_or_gender_area"}

    missing_base = required_base_cols - set(base_df.columns)
    missing_posts = required_post_cols - set(posts_df.columns)

    if missing_base:
        raise ValueError(f"主檔缺少必要欄位：{missing_base}")
    if missing_posts:
        raise ValueError(f"文章檔缺少必要欄位：{missing_posts}")

    # 建立清理欄位
    posts_df["gender"] = posts_df["author_or_gender_area"].apply(extract_gender)
    posts_df["content_cleaned"] = posts_df.apply(
        lambda row: clean_post_content(row["content"], row["title"]), axis=1
    )

    # 只保留要合併回主檔的欄位
    posts_for_merge = posts_df[["web_scraper_start_url", "title", "content_cleaned", "gender"]].copy()
    posts_for_merge = posts_for_merge.rename(
        columns={
            "web_scraper_start_url": "link",
            "title": "article_title",
            "content_cleaned": "article_content",
        }
    )

    # 若文章檔中網址重複，保留第一筆，避免 merge 後膨脹列數。
    posts_for_merge = posts_for_merge.drop_duplicates(subset=["link"], keep="first")

    # 以主檔為準，將文章標題、內容、性別合併到主檔後面。
    merged_df = base_df.merge(posts_for_merge, on="link", how="left")

    # 輸出 UTF-8 with BOM，Excel 開啟中文較不容易亂碼。
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 印出簡短檢查資訊
    print(f"主檔列數：{len(base_df)}")
    print(f"文章檔合併後不重複網址數：{len(posts_for_merge)}")
    print(f"成功對應列數：{merged_df['article_title'].notna().sum()}")
    print(f"未對應列數：{merged_df['article_title'].isna().sum()}")
    print("性別統計：")
    print(merged_df["gender"].value_counts(dropna=False))
    print(f"已輸出：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
