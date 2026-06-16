"""
NIfTI 直接读取脚本
使用 nibabel 直接读取 .nii.gz 文件，将体积数据归一化为 uint8 后
写入 raw 文件，元数据通过 JSON stdout 返回给 Electron 主进程。

用法: python nifti_reader.py <nifti_file_path> <output_dir> [--raw]

参数:
  --raw  保留原始值（用于分割结果），不做归一化

输出: JSON 行格式（进度），最终结果写入 output_dir/meta.json + volume.raw
"""

import sys
import os
import json
import numpy as np
import nibabel as nib


def emit(progress, message):
    """输出进度 JSON 行"""
    print(json.dumps({"progress": progress, "message": message}), flush=True)


def read_nifti(file_path, output_dir, raw_mode=False):
    """读取 NIfTI 文件，写入 raw 文件"""
    emit(10, "正在读取NIfTI文件...")
    img = nib.load(file_path)
    data = img.get_fdata()

    shape = list(data.shape)
    zooms = img.header.get_zooms()
    spacing = [float(abs(z)) for z in zooms[:3]]

    emit(30, f"尺寸: {shape[0]}x{shape[1]}x{shape[2]}, 间距: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}")

    emit(50, "正在处理体积数据...")

    if raw_mode:
        # 保留原始标签值（用于分割结果 0,1,2,...）
        data = np.nan_to_num(data, nan=0.0)
        data = data.astype(np.uint8)
    else:
        # Z-score 归一化 → min-max 到 0-255
        data = data.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        mean = float(np.mean(data))
        std = float(np.std(data))
        if std > 0:
            data = (data - mean) / std
        else:
            data = data - mean

        dmin = float(data.min())
        dmax = float(data.max())

        if dmax - dmin > 0:
            data = (data - dmin) / (dmax - dmin) * 255.0
        else:
            data = np.zeros_like(data)

        data = data.astype(np.uint8)

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    emit(70, "正在写入体积数据...")
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, 'volume.raw')
    data.tofile(raw_path)

    meta_path = os.path.join(output_dir, 'meta.json')
    meta = {"shape": shape, "spacing": spacing}
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    emit(100, json.dumps({"shape": shape, "spacing": spacing, "rawDir": output_dir}))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        emit(0, "错误: 用法: nifti_reader.py <nifti_file_path> <output_dir> [--raw]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    raw_mode = '--raw' in sys.argv

    try:
        read_nifti(file_path, output_dir, raw_mode)
    except Exception as e:
        emit(0, f"错误: {str(e)}")
        sys.exit(1)
