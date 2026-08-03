# -*- coding: utf-8 -*-
"""
知识审核脚本 — 清洗 feedback_log.jsonl 中的反馈记录
用法: python knowledge_reviewer.py

处理规则（status == "pending_review" 的记录）：
  - feedback_type == "adopted" 且无 correction → status = "approved"
  - feedback_type == "corrected" → 用纠正版本替换答案，status = "approved"
  - feedback_type == "discarded" 且无 correction → 删除该记录
"""
import sys
import os
import json

# 强制 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'knowledge_base', 'feedback_log.jsonl')


def review():
    if not os.path.exists(LOG_FILE):
        print(json.dumps({"success": True, "message": "日志文件不存在，无需审核", "reviewed": 0, "approved": 0, "discarded": 0}), flush=True)
        return

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    kept = []
    reviewed_count = 0
    approved_count = 0
    discarded_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 只处理 pending_review 的记录
        if record.get("status") != "pending_review":
            kept.append(record)
            continue

        reviewed_count += 1
        feedback_type = record.get("feedback_type", "discarded")
        correction = record.get("correction", "").strip()

        if feedback_type == "adopted" and not correction:
            # 采纳且无纠正 → 直接通过
            record["status"] = "approved"
            kept.append(record)
            approved_count += 1

        elif feedback_type == "corrected" and correction:
            # 有纠正内容 → 用纠正版本替换答案
            record["answer"] = correction
            record["status"] = "approved"
            kept.append(record)
            approved_count += 1

        elif feedback_type == "discarded" and not correction:
            # 未采纳且无反馈 → 丢弃
            discarded_count += 1
        else:
            # 其他情况（如 adopted+correction, corrected+空纠正）→ 保留待人工确认
            kept.append(record)

    # 写回文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    result = {
        "success": True,
        "reviewed": reviewed_count,
        "approved": approved_count,
        "discarded": discarded_count,
        "remaining": len(kept)
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    review()
