# -*- coding: utf-8 -*-
"""
向量化脚本 — 将反馈记录或文献文本块向量化并写入 Faiss 索引
用法:
  python vectorize_and_index.py                    # 默认 feedback 模式
  python vectorize_and_index.py --mode feedback    # 反馈记录向量化
  python vectorize_and_index.py --mode literature  # 文献文本块向量化

依赖: faiss-cpu, numpy
使用 DashScope text-embedding-v3 API 生成向量（与 ai_chat.py 保持一致）

输出:
  feedback 模式:
    knowledge_base/faiss.index         — Faiss 向量索引
    knowledge_base/knowledge_meta.json — 元数据
  literature 模式:
    literature/lit_faiss.index         — 文献向量索引
    literature/lit_meta.json           — 文献元数据
"""
import sys
import os
import json
import argparse
import numpy as np
import urllib.request
import urllib.error

# 强制 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 反馈模式路径
FEEDBACK_LOG = os.path.join(SCRIPT_DIR, 'knowledge_base', 'feedback_log.jsonl')
FEEDBACK_INDEX = os.path.join(SCRIPT_DIR, 'knowledge_base', 'faiss.index')
FEEDBACK_META = os.path.join(SCRIPT_DIR, 'knowledge_base', 'knowledge_meta.json')

# 文献模式路径
LIT_CHUNKS = os.path.join(SCRIPT_DIR, 'literature', 'chunks.jsonl')
LIT_INDEX = os.path.join(SCRIPT_DIR, 'literature', 'lit_faiss.index')
LIT_META = os.path.join(SCRIPT_DIR, 'literature', 'lit_meta.json')

# 共用 API 配置（与 ai_chat.py 保持一致的在用 key）
API_KEY = "sk-ws-H.EMPYXEY.U028.MEQCIB9QuZf3IOySwm7Pl7Z0MDz-F9DPicQSMZL-gOSmTiSLAiBZOXjjFtVoOPV9SDbSgycHi3XneoXyooU_X3fUFEo-eQ"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024


def get_embeddings(texts):
    """调用 DashScope embedding API，返回向量列表（含重试）"""
    if not texts:
        return []
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "encoding_format": "float"
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                EMBEDDING_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return [item["embedding"] for item in sorted(result.get("data", []), key=lambda x: x.get("index", 0))]
        except urllib.error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                pass
            sample = (texts[0][:60] if texts else '')
            if attempt < 2:
                import time
                time.sleep(2)
                continue
            raise RuntimeError(f"embedding HTTP {e.code} | 首条样本={sample!r} | 响应={body}") from e
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(2)
                continue
            raise


def vectorize_feedback():
    """向量化审核通过的反馈记录"""
    import faiss

    if not os.path.exists(FEEDBACK_LOG):
        print(json.dumps({"success": True, "message": "反馈日志不存在", "count": 0}), flush=True)
        return

    # 读取已审核记录
    approved = []
    with open(FEEDBACK_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") == "approved":
                    approved.append(r)
            except json.JSONDecodeError:
                continue

    if not approved:
        print(json.dumps({"success": True, "message": "无已审核记录", "count": 0}), flush=True)
        return

    # 加载已有元数据（增量追加）
    existing_meta = []
    existing_ids = set()
    if os.path.exists(FEEDBACK_META):
        try:
            with open(FEEDBACK_META, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
            existing_ids = {m["id"] for m in existing_meta}
        except Exception:
            existing_meta = []

    new_records = [r for r in approved if r["id"] not in existing_ids]
    if not new_records:
        print(json.dumps({"success": True, "message": "无新增反馈", "count": len(existing_meta)}), flush=True)
        return

    # 向量化
    texts = [f"问题：{r.get('question','')}\n回答：{r.get('answer','')}" for r in new_records]
    all_embeddings = []
    for i in range(0, len(texts), 20):
        all_embeddings.extend(get_embeddings(texts[i:i + 20]))

    if not all_embeddings:
        print(json.dumps({"success": False, "error": "向量化失败"}), flush=True)
        return

    vectors = np.array(all_embeddings, dtype='float32')

    # 加载或创建索引
    if os.path.exists(FEEDBACK_INDEX):
        index = faiss.read_index(FEEDBACK_INDEX)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    faiss.normalize_L2(vectors)
    index.add(vectors)

    # 更新元数据
    for r in new_records:
        existing_meta.append({
            "id": r["id"],
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "feedback_type": r.get("feedback_type", ""),
            "timestamp": r.get("timestamp", "")
        })

    os.makedirs(os.path.dirname(FEEDBACK_INDEX), exist_ok=True)
    faiss.write_index(index, FEEDBACK_INDEX)
    with open(FEEDBACK_META, 'w', encoding='utf-8') as f:
        json.dump(existing_meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "success": True, "mode": "feedback",
        "new_count": len(new_records), "total_count": len(existing_meta)
    }, ensure_ascii=False), flush=True)


def vectorize_literature():
    """向量化文献文本块"""
    import faiss

    if not os.path.exists(LIT_CHUNKS):
        print(json.dumps({"success": False, "error": "chunks.jsonl 不存在，请先运行 document_parser.py"}), flush=True)
        return

    # 读取所有文本块
    chunks = []
    with open(LIT_CHUNKS, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not chunks:
        print(json.dumps({"success": True, "message": "无文本块", "count": 0}), flush=True)
        return

    # 检查是否需要重建（对比 chunks 数量）
    existing_meta = []
    if os.path.exists(LIT_META):
        try:
            with open(LIT_META, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = []

    # 如果 chunks 数量与已有 meta 一致，跳过
    if len(chunks) == len(existing_meta):
        print(json.dumps({
            "success": True, "message": "索引已是最新",
            "count": len(chunks), "skipped": True
        }, ensure_ascii=False), flush=True)
        return

    # 向量化（全量重建，因为 chunks 可能已变化）
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(get_embeddings(batch))

    if not all_embeddings:
        print(json.dumps({"success": False, "error": "向量化失败"}), flush=True)
        return

    vectors = np.array(all_embeddings, dtype='float32')
    faiss.normalize_L2(vectors)

    # 创建新索引（全量重建）
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    # 构建元数据
    meta = []
    for i, c in enumerate(chunks):
        meta.append({
            "id": f"lit_{i}",
            "source_file": c.get("source_file", ""),
            "page": c.get("page", 0),
            "chunk_id": c.get("chunk_id", 0),
            "text": c.get("text", ""),
            "tokens_est": c.get("tokens_est", 0)
        })

    os.makedirs(os.path.dirname(LIT_INDEX), exist_ok=True)
    faiss.write_index(index, LIT_INDEX)
    with open(LIT_META, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "success": True, "mode": "literature",
        "total_chunks": len(chunks), "index_file": LIT_INDEX
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='向量化脚本')
    parser.add_argument('--mode', choices=['feedback', 'literature'], default='feedback',
                        help='向量化模式：feedback（反馈记录）或 literature（文献文本块）')
    args = parser.parse_args()

    try:
        import faiss
        if args.mode == 'feedback':
            vectorize_feedback()
        else:
            vectorize_literature()
    except ImportError as e:
        print(json.dumps({"success": False, "error": f"缺少依赖: {e}. 请运行: pip install faiss-cpu"}), flush=True)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
        sys.exit(1)
