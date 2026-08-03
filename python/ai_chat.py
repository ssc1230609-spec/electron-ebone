# -*- coding: utf-8 -*-
"""
AI 助手后端 - 流式调用 Qwen-VL-Max 大模型
用法: python ai_chat.py <json_arg>
json_arg: {"question": "...", "context": {...}}

输出: 逐行 JSON，格式如下
  {"type": "start", "elapsed": 0}
  {"type": "chunk", "content": "..."}
  ...
  {"type": "end", "elapsed": 3.2, "context": "..."}
"""
import sys
import os
import json
import time
import codecs
import urllib.request
import urllib.error
import numpy as np
import nibabel

# 强制 stdout/stderr 使用 UTF-8 编码（Windows 默认 cp1252 不支持中文）
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-ws-H.EDXIYMM.486G.MEQCIGZk6NCHWXyoVZuANMM4Sr1QJYdqT4U04PhVGkdQeABlAiBGTOGMOfyGHCrehUfsg_A5eZ9mKPu-wuF-tfjBU5hcFA"  # 替换为你的API Key
API_URL = "https://ws-nzb728qbrjk9u0fa.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-vl-plus"  # 视觉模型，支持 CT 切片图片；模型ID大小写敏感须全小写

# ============================================================
# RAG 向量检索配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 反馈知识库
FEEDBACK_INDEX = os.path.join(SCRIPT_DIR, 'knowledge_base', 'faiss.index')
FEEDBACK_META = os.path.join(SCRIPT_DIR, 'knowledge_base', 'knowledge_meta.json')

# 文献知识库
LIT_INDEX = os.path.join(SCRIPT_DIR, 'literature', 'lit_faiss.index')
LIT_META = os.path.join(SCRIPT_DIR, 'literature', 'lit_meta.json')

EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-v3"

# 懒加载索引（进程生命周期内只加载一次）
_fb_index = None
_fb_meta = None
_lit_index = None
_lit_meta = None
_indexes_loaded = False


def _load_indexes():
    """懒加载反馈索引 + 文献索引"""
    global _fb_index, _fb_meta, _lit_index, _lit_meta, _indexes_loaded
    if _indexes_loaded:
        return
    _indexes_loaded = True
    try:
        import faiss
        # 反馈索引
        if os.path.exists(FEEDBACK_INDEX) and os.path.exists(FEEDBACK_META):
            _fb_index = faiss.read_index(FEEDBACK_INDEX)
            with open(FEEDBACK_META, 'r', encoding='utf-8') as f:
                _fb_meta = json.load(f)
        # 文献索引
        if os.path.exists(LIT_INDEX) and os.path.exists(LIT_META):
            _lit_index = faiss.read_index(LIT_INDEX)
            with open(LIT_META, 'r', encoding='utf-8') as f:
                _lit_meta = json.load(f)
    except Exception:
        pass  # 索引加载失败静默降级


def _get_embedding(text):
    """调用 DashScope embedding API 获取单条文本的向量"""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [text],
        "encoding_format": "float"
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        EMBEDDING_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["data"][0]["embedding"]


def search_knowledge(question, top_k=5, threshold=0.55):
    """在反馈索引 + 文献索引中检索最相似的知识条目，合并排序返回 top_k"""
    _load_indexes()

    try:
        import numpy as np
        import faiss

        q_vec = np.array([_get_embedding(question)], dtype='float32')
        faiss.normalize_L2(q_vec)

        all_results = []

        # 检索反馈知识库
        if _fb_index is not None and _fb_meta is not None and _fb_index.ntotal > 0:
            k = min(top_k, _fb_index.ntotal)
            scores, indices = _fb_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(_fb_meta):
                    continue
                if score >= threshold:
                    meta = _fb_meta[idx]
                    all_results.append({
                        "source": "feedback",
                        "text": f"Q: {meta.get('question','')}\nA: {meta.get('answer','')}",
                        "score": round(float(score), 3)
                    })

        # 检索文献知识库
        if _lit_index is not None and _lit_meta is not None and _lit_index.ntotal > 0:
            k = min(top_k, _lit_index.ntotal)
            scores, indices = _lit_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(_lit_meta):
                    continue
                if score >= threshold:
                    meta = _lit_meta[idx]
                    src = meta.get("source_file", "未知文献")
                    page = meta.get("page", "")
                    page_str = f" p.{page}" if page else ""
                    all_results.append({
                        "source": "literature",
                        "text": f"[来源: {src}{page_str}] {meta.get('text','')}",
                        "source_file": src,
                        "page": page,
                        "score": round(float(score), 3)
                    })

        # 按相似度排序，取 top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    except Exception:
        return []

SYSTEM_PROMPT = """你是反式肩关节置换术术前规划系统的AI助手，专注肩关节置换领域。

【重要】用户请求分类处理：
1. 影像分析请求：当用户说"分析"、"报告"、"诊断"CT影像时，才输出完整报告。
2. 专业咨询：回答肩关节置换相关问题，提供专业建议。

参数定义：
- length: 入钉长度（mm）
- angle1: 上下倾角（°），正值向上，负值向下
- angle2: 左右倾角（°），正值向左，负值向右

上下文数据：
- 【植入位置信息】包含原三点取圆的圆心坐标（植入点）、圆半径、假体目录路径

【参数调整建议范围】（必须严格遵守）
当操作者询问角度调整、长度调整相关问题时：
- 长度调整：建议控制在当前长度的 ±2-3mm 范围内
- 角度调整（angle1/angle2）：建议控制在 ±1-2° 范围内
- 超出此范围时必须提醒风险，建议分步小幅调整
- 这是术前规划系统，大幅调整可能导致假体位置不良

说话风格：
- 不要说"建议您考虑"、"需要注意的是"、"根据提供的数据"这类话
- 不要用"首先/其次/最后"这种排比结构
- 不要总结、不要重复问题、不要说"希望对您有帮助"
- 用口语化的中文，专业术语正常用就行
- 回答尽量短

专业知识：
- RSA适应症：CSA、复杂肱骨近端骨折、肿瘤、翻修
- 前倾角正常10-20°，下倾角95-105°，偏距影响三角肌张力
- 盂球大小影响活动度和稳定性
- 常见假体：Delta XTRT, Comprehensive, Equinoxe
- 禁止使用*、#、-等Markdown符号，纯文本输出

CT影像分析格式（仅在用户明确要求分析影像时使用）：

影像所见：
骨质情况：（描述骨密度、有无骨质疏松、骨赘、骨缺损）
关节盂形态：（Walch分型、有无磨损、后倾角、骨缺损程度）
肱骨头形态：（有无坏死、骨折、变形、骨赘）
关节间隙：（正常/变窄/消失）
肩袖情况：（如影像可见，描述冈上肌/冈下肌/肩胛下肌形态）
其他发现：（肩峰形态、锁骨、软组织等其他异常）

诊断意见：
（简明列出主要诊断）

手术规划参考：
（关键影像学发现对反肩置换手术的影响）
"""

def emit(obj):
    """输出一行 JSON 到 stdout"""
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def nifti_to_base64_slices(nifti_path, num_slices=3):
    """读取 NIfTI 文件，提取轴状位、冠状位、矢状位各若干切片，返回 base64 PNG 列表"""
    import base64
    import io
    from PIL import Image

    img = nibabel.load(nifti_path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]

    # 归一化到 0-255
    mask = data > 0
    if np.any(mask):
        vmin, vmax = np.percentile(data[mask], [1, 99])
    else:
        vmin, vmax = 0, 1
    data = np.clip((data - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

    sx, sy, sz = data.shape
    images_b64 = []

    # 轴状位（axial）：沿 Z 轴取中间若干层
    for i in np.linspace(sz // 4, 3 * sz // 4, num_slices, dtype=int):
        slice_img = data[:, :, i].T[::-1]
        img_pil = Image.fromarray(slice_img, mode='L')
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        images_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    # 冠状位（coronal）：沿 Y 轴取中间若干层
    for i in np.linspace(sy // 4, 3 * sy // 4, num_slices, dtype=int):
        slice_img = data[:, i, :].T[::-1]
        img_pil = Image.fromarray(slice_img, mode='L')
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        images_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    # 矢状位（sagittal）：沿 X 轴取中间若干层
    for i in np.linspace(sx // 4, 3 * sx // 4, num_slices, dtype=int):
        slice_img = data[i, :, :].T[::-1]
        img_pil = Image.fromarray(slice_img, mode='L')
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        images_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    return images_b64

# ============================================================
# 流式调用 Qwen-VL-Max API
# ============================================================
def call_qwen_stream(question, context=None):
    ctx_parts = []

    if context:
        if context.get("params"):
            p = context["params"]
            ctx_parts.append(f"【当前规划参数】\n入钉长度: {p.get('length', '未设定')}\n上下倾角: {p.get('angle1', '未设定')}\n左右倾角: {p.get('angle2', '未设定')}")
        if context.get("implant"):
            imp = context["implant"]
            c = imp.get("center", [0,0,0])
            pts = imp.get("points")
            pt_str = ""
            if pts:
                pt_str = f"\n取点坐标: P1=({pts[0][0]:.1f},{pts[0][1]:.1f},{pts[0][2]:.1f}) P2=({pts[1][0]:.1f},{pts[1][1]:.1f},{pts[1][2]:.1f}) P3=({pts[2][0]:.1f},{pts[2][1]:.1f},{pts[2][2]:.1f})"
            ctx_parts.append(f"【植入位置信息】\n圆心(植入点): ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})\n圆半径: {imp.get('radius', 0):.1f}mm{pt_str}\n假体目录: {imp.get('stl_dir', '未知')}")
        if context.get("bone_info"):
            ctx_parts.append(f"【骨骼信息】\n{context['bone_info']}")
        if context.get("model_info"):
            ctx_parts.append(f"【当前导入模型】\n{context['model_info']}")
        if context.get("ct_info"):
            info = context["ct_info"]
            desc = f"【CT影像信息】尺寸: {info.get('shape')}, 间距: {info.get('spacing')}"
            if info.get("has_segmentation"):
                desc += "，含分割标注"
            ctx_parts.append(desc)

    # 从 NIfTI 文件提取三视图切片作为图片发送
    nifti_path = context.get("nifti_path") if context else None
    nifti_images = []
    if nifti_path and os.path.exists(nifti_path):
        try:
            nifti_images = nifti_to_base64_slices(nifti_path, num_slices=3)
        except Exception as e:
            emit({"type": "chunk", "content": f"[切片提取失败: {e}] 将仅用文本分析。\n"})

    if nifti_images:
        content_parts = []
        if ctx_parts:
            content_parts.append({"type": "text", "text": "\n\n".join(ctx_parts) + "\n\n"})
        for b64 in nifti_images:
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content_parts.append({"type": "text", "text": f"\n\n【问题】{question}"})
        user_content = content_parts
    elif ctx_parts:
        user_content = "\n\n".join(ctx_parts) + "\n\n【问题】" + question
    else:
        user_content = question

    context_display = user_content if isinstance(user_content, str) else f"[含NIfTI文件] {question}"

    # RAG：向量检索（反馈知识 + 文献知识）
    rag_prompt = ""
    lit_refs = []  # 结构化文献引用，用于回答末尾确定性展示
    try:
        similar = search_knowledge(question, top_k=5, threshold=0.55)
        if similar:
            fb_parts = []
            lit_parts = []
            seen_refs = {}  # source_file -> 引用对象，同一篇文献合并（页码合并、保留最高分）
            for item in similar:
                if item["source"] == "literature":
                    lit_parts.append(item["text"])
                    src = item.get("source_file", "")
                    page = item.get("page", "")
                    prev = seen_refs.get(src)
                    if prev is None:
                        seen_refs[src] = {
                            "source_file": src,
                            "pages": [page] if page else [],
                            "score": item.get("score", 0)
                        }
                    else:
                        if page and page not in prev["pages"]:
                            prev["pages"].append(page)
                        if item.get("score", 0) > prev.get("score", 0):
                            prev["score"] = item.get("score", 0)
                else:
                    fb_parts.append(item["text"])

            # 文献引用按相似度排序
            lit_refs = sorted(seen_refs.values(), key=lambda x: x.get("score", 0), reverse=True)

            rag_sections = []
            if lit_parts:
                rag_sections.append("【文献参考】\n" + "\n\n".join(lit_parts))
            if fb_parts:
                rag_sections.append("【历史经验参考】\n" + "\n\n".join(fb_parts))
            rag_prompt = "\n\n" + "\n\n".join(rag_sections)
    except Exception as e:
        print(f"[RAG] 检索异常: {e}", file=sys.stderr)

    # 将引用来源附加到 context_display（显示在前端"思考详情"中）
    if lit_refs:
        ref_strs = []
        for r in lit_refs:
            name = r["source_file"].replace("\\", "/").split("/")[-1]
            pages = sorted(set(r.get("pages", [])))
            page_str = (" p." + ",".join(str(p) for p in pages)) if pages else ""
            ref_strs.append(f"{name}{page_str}")
        context_display += "\n参考来源: " + " | ".join(ref_strs)

    system_content = SYSTEM_PROMPT + rag_prompt

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
        "stream": True
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )

    start_time = time.time()
    emit({"type": "start", "elapsed": 0})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text_buf = ""
            decoder = codecs.getincrementaldecoder("utf-8")()
            for chunk in iter(lambda: resp.read(1024), b""):
                text_buf += decoder.decode(chunk, False)

                while "\n" in text_buf:
                    line, text_buf = text_buf.split("\n", 1)
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        obj = json.loads(line)
                        delta = (obj.get("choices") or [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            emit({"type": "chunk", "content": content})
                    except json.JSONDecodeError:
                        continue

            # flush decoder，处理最后可能残留的多字节字符
            text_buf += decoder.decode(b"", True)
            if text_buf.strip():
                line = text_buf.strip()
                if line.startswith("data: "):
                    line = line[6:]
                if line and line != "data: [DONE]":
                    try:
                        obj = json.loads(line)
                        delta = (obj.get("choices") or [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            emit({"type": "chunk", "content": content})
                    except json.JSONDecodeError:
                        pass

        elapsed = round(time.time() - start_time, 1)
        emit({"type": "end", "elapsed": elapsed, "context": context_display, "refs": lit_refs})

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        elapsed = round(time.time() - start_time, 1)
        emit({"type": "chunk", "content": f"[API错误] HTTP {e.code}: {error_body}"})
        emit({"type": "end", "elapsed": elapsed, "context": context_display, "refs": lit_refs})
    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        emit({"type": "chunk", "content": f"[请求失败] {str(e)}"})
        emit({"type": "end", "elapsed": elapsed, "context": context_display, "refs": lit_refs})

# ============================================================
# 主入口
# ============================================================
def main():
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "缺少参数"})
        sys.exit(1)

    try:
        # 支持 --file <path> 方式传参，避免命令行过长（含图片时 base64 超出 Windows 限制）
        if sys.argv[1] == '--file' and len(sys.argv) >= 3:
            with open(sys.argv[2], 'r', encoding='utf-8') as f:
                arg = json.load(f)
        else:
            arg = json.loads(sys.argv[1])

        question = arg.get("question", "")
        context = arg.get("context", None)

        if not question:
            emit({"type": "error", "message": "问题不能为空"})
            sys.exit(1)

        call_qwen_stream(question, context)

    except json.JSONDecodeError:
        emit({"type": "error", "message": "参数JSON解析失败"})
        sys.exit(1)
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
