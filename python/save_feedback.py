# -*- coding: utf-8 -*-
"""
反馈记录存储脚本
用法: python save_feedback.py < tmpFile.json
或:   echo '{"question":"...","answer":"...",...}' | python save_feedback.py

从 stdin 读取 JSON，追加一行到 knowledge_base/feedback_log.jsonl
"""
import sys
import os
import json
import uuid
from datetime import datetime

# 强制 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 日志文件路径（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'knowledge_base')
LOG_FILE = os.path.join(LOG_DIR, 'feedback_log.jsonl')


def main():
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"success": False, "error": "stdin 为空"}), flush=True)
            sys.exit(1)

        data = json.loads(raw)

        # 构建记录
        record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "context_snapshot": data.get("context_snapshot", {}),
            "feedback_type": data.get("feedback_type", "discarded"),  # adopted / corrected / discarded
            "correction": data.get("correction", ""),
            "status": "pending_review"
        }

        # 确保目录存在
        os.makedirs(LOG_DIR, exist_ok=True)

        # 追加写入
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(json.dumps({"success": True, "id": record["id"]}), flush=True)

    except json.JSONDecodeError:
        print(json.dumps({"success": False, "error": "JSON 解析失败"}), flush=True)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
