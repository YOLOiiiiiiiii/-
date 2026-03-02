#!/usr/bin/env python3
"""高中历史课本 OCR + LLM 知识体系抽取脚本。

用法示例:
  python ocr_history_pipeline.py \
    --input-dir ./textbook_pages \
    --output-json ./history_knowledge.json \
    --model gpt-4.1

依赖:
  pip install paddleocr pillow openai

环境变量:
  OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI
from paddleocr import PaddleOCR

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".pdf"}


@dataclass
class OCRPage:
    filename: str
    text: str


def collect_input_files(input_dir: Path) -> List[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"未在目录中找到可处理文件: {input_dir}")
    return files


def run_ocr(files: Iterable[Path], lang: str = "ch") -> List[OCRPage]:
    """使用 PaddleOCR 提取每页文字。"""
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    pages: List[OCRPage] = []

    for f in files:
        result = ocr.ocr(str(f), cls=True)
        lines: List[str] = []

        for page_block in result or []:
            for line in page_block or []:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    text = line[1][0]
                    if text and text.strip():
                        lines.append(text.strip())

        pages.append(OCRPage(filename=f.name, text="\n".join(lines)))

    return pages


def chunk_text(pages: List[OCRPage], max_chars: int = 12000) -> List[str]:
    """将 OCR 文本按长度分块，避免单次上下文过长。"""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for page in pages:
        block = f"\n\n### 文件: {page.filename}\n{page.text}"
        if current_len + len(block) > max_chars and current:
            chunks.append("".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += len(block)

    if current:
        chunks.append("".join(current))

    return chunks


def summarize_chunk(client: OpenAI, model: str, chunk: str) -> str:
    prompt = (
        "你是高中历史知识整理专家。请从以下OCR文本中抽取核心历史知识点，"
        "并尽量去除噪声(页码、目录、错字)。输出为紧凑中文要点列表。\n\n"
        f"{chunk}"
    )

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.2,
    )
    return resp.output_text.strip()


def build_knowledge_json(client: OpenAI, model: str, merged_notes: str) -> dict:
    schema_hint = {
        "version": "1.0",
        "subject": "高中历史",
        "framework": [
            {
                "id": "A",
                "name": "一级主题",
                "children": [
                    {
                        "id": "A1",
                        "name": "二级知识点",
                        "children": [
                            {
                                "id": "A1-1",
                                "name": "三级知识点",
                                "keywords": ["关键词1", "关键词2"],
                                "related": ["关联概念"],
                                "summary": "一句话说明"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    prompt = (
        "请基于给定历史文本要点，生成‘三层结构’的高中历史知识体系JSON。"
        "要求：\n"
        "1) 覆盖中国史+世界史核心模块。\n"
        "2) 仅输出合法JSON，不要markdown代码块。\n"
        "3) 尽量补全关联知识点，体现时序、制度、经济、思想、外交等维度。\n"
        "4) 严格参考如下结构：\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        "以下是文本要点:\n"
        f"{merged_notes}"
    )

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.1,
    )

    return json.loads(resp.output_text)


def save_raw_ocr(pages: List[OCRPage], path: Path) -> None:
    payload = [{"filename": p.filename, "text": p.text} for p in pages]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="从历史课本图片/PDF自动生成三层知识点JSON")
    parser.add_argument("--input-dir", required=True, help="课本扫描图或PDF目录")
    parser.add_argument("--output-json", required=True, help="输出知识体系 JSON 路径")
    parser.add_argument("--output-ocr", default="ocr_raw.json", help="OCR原文输出路径")
    parser.add_argument("--model", default="gpt-4.1", help="用于整理知识体系的模型")
    parser.add_argument("--lang", default="ch", help="PaddleOCR语言，如ch/en")
    parser.add_argument("--max-chars", type=int, default=12000, help="文本分块上限")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = collect_input_files(input_dir)

    pages = run_ocr(files, lang=args.lang)
    save_raw_ocr(pages, Path(args.output_ocr))

    chunks = chunk_text(pages, max_chars=args.max_chars)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    notes = [summarize_chunk(client, args.model, chunk) for chunk in chunks]
    merged_notes = "\n\n".join(notes)

    knowledge = build_knowledge_json(client, args.model, merged_notes)
    Path(args.output_json).write_text(
        json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"完成: 识别文件 {len(files)} 个，输出 {args.output_json}")


if __name__ == "__main__":
    main()
