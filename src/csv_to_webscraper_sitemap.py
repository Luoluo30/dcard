import pandas as pd
import json
import argparse
from pathlib import Path
import math


def build_sitemap(urls, sitemap_id):
    return {
        "_id": sitemap_id,
        "startUrl": urls,
        "selectors": [
            {
                "id": "title",
                "parentSelectors": ["_root"],
                "type": "SelectorText",
                "selector": "h1",
                "multiple": False,
                "regex": "",
                "delay": 0,
            },
            {
                "id": "content",
                "parentSelectors": ["_root"],
                "type": "SelectorText",
                "selector": "article",
                "multiple": False,
                "regex": "",
                "delay": 0,
            },
            {
                "id": "author_or_gender_area",
                "parentSelectors": ["_root"],
                "type": "SelectorText",
                "selector": "article header, header, [data-key='author']",
                "multiple": False,
                "regex": "",
                "delay": 0,
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CSV of Dcard post URLs into multiple Web Scraper sitemap JSON files."
    )
    parser.add_argument("--input", default="dcard (2).csv", help="Input CSV file")
    parser.add_argument("--url-column", default="link", help="Column containing Dcard post URLs")
    parser.add_argument("--output-dir", default="sitemaps", help="Folder for output sitemap JSON files")
    parser.add_argument("--batch-size", type=int, default=500, help="URLs per sitemap file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--id-prefix", default="dcard_posts", help="Prefix for sitemap _id and filenames")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.url_column not in df.columns:
        raise ValueError(
            f"Column '{args.url_column}' not found. Available columns: {list(df.columns)}"
        )

    urls = df[args.url_column].dropna().drop_duplicates().astype(str).tolist()

    # 只保留 Dcard 文章網址，避免混入其他連結
    urls = [u for u in urls if "dcard.tw" in u and "/p/" in u]

    if args.limit:
        urls = urls[: args.limit]

    total_urls = len(urls)

    if total_urls == 0:
        print("No valid URLs found.")
        return

    total_batches = math.ceil(total_urls / args.batch_size)

    for batch_index in range(total_batches):
        start = batch_index * args.batch_size
        end = start + args.batch_size
        batch_urls = urls[start:end]

        part_no = batch_index + 1
        sitemap_id = f"{args.id_prefix}_part_{part_no:03d}"

        sitemap = build_sitemap(batch_urls, sitemap_id)

        output_file = output_dir / f"{sitemap_id}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sitemap, f, ensure_ascii=False, indent=2)

        print(
            f"Saved part {part_no:03d}/{total_batches:03d}: "
            f"{len(batch_urls)} URLs -> {output_file}"
        )

    print(f"Done. Created {total_batches} sitemap files in: {output_dir.resolve()}")
    print(f"Total URLs exported: {total_urls}")


if __name__ == "__main__":
    main()