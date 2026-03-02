# 高中历史课本 OCR + 知识体系自动整理

这个仓库提供了一个可直接运行的脚本：

- 先用 **PaddleOCR** 提取历史课本扫描图/PDF中的文本；
- 再调用 **OpenAI 大模型** 对文本做去噪、归纳；
- 最终输出一个约三层结构的高中历史知识点体系 `JSON`。

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 准备数据

把课本扫描件放到一个目录，例如：

```text
textbook_pages/
  001.jpg
  002.jpg
  ...
```

支持格式：`png/jpg/jpeg/bmp/tiff/webp/pdf`。

## 3. 配置模型密钥

```bash
export OPENAI_API_KEY="你的key"
```

## 4. 运行

```bash
python ocr_history_pipeline.py \
  --input-dir ./textbook_pages \
  --output-json ./history_knowledge.json \
  --output-ocr ./ocr_raw.json \
  --model gpt-4.1
```

## 5. 输出说明

- `ocr_raw.json`：每个文件对应的 OCR 原文；
- `history_knowledge.json`：三层历史知识体系（一级主题 -> 二级知识点 -> 三级知识点），并附关键词与关联概念。

## 6. 建议

- 如果扫描质量一般，建议先做图像预处理（去噪、拉伸对比度）。
- 若文本超大，脚本会自动分块，先局部归纳再全局生成结构化知识体系。
- 你可以把 `history_knowledge.json` 再接入知识图谱或向量数据库用于检索问答。
