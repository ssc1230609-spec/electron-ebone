"""
nnU-Net v1 推理脚本
调用 nnunet.inference.predict.predict_from_folder 进行分割推理。

用法: python run_inference.py <input_folder> <output_folder>

输入文件夹必须包含 nnU-Net 格式的文件: CASENAME_0000.nii.gz
输出文件夹将包含: CASENAME.nii.gz (分割结果)

进度通过 stdout JSON 行报告给 Electron 主进程。
"""

import sys
import os
import json
import shutil
import time

# 修复 Windows 控制台中文编码问题
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def emit(progress, message):
    """输出进度 JSON 行"""
    print(json.dumps({"progress": progress, "message": message}), flush=True)


def prepare_nnunet_input(input_folder):
    """
    准备 nnU-Net 输入：创建临时干净文件夹，只放入目标 NIfTI 文件。
    避免原文件夹中其他 .nii.gz 文件干扰 nnunet 推理。

    返回 (temp_input_folder, case_name)
    """
    files = os.listdir(input_folder)

    # 1. 优先找已符合 _0000.nii.gz 格式的文件
    for f in files:
        if f.endswith('_0000.nii.gz'):
            case_name = f.replace('_0000.nii.gz', '')
            temp_dir = os.path.join(os.path.dirname(input_folder), '_nnunet_input')
            os.makedirs(temp_dir, exist_ok=True)
            shutil.copy2(os.path.join(input_folder, f), os.path.join(temp_dir, f))
            return temp_dir, case_name

    # 2. 找普通 .nii.gz 文件，复制并重命名为 _0000 格式
    for f in files:
        if f.endswith('.nii.gz'):
            case_name = f.replace('.nii.gz', '')
            temp_dir = os.path.join(os.path.dirname(input_folder), '_nnunet_input')
            os.makedirs(temp_dir, exist_ok=True)
            src = os.path.join(input_folder, f)
            dst = os.path.join(temp_dir, case_name + '_0000.nii.gz')
            shutil.copy2(src, dst)
            return temp_dir, case_name

    # 3. 处理 .nii 文件
    for f in files:
        if f.endswith('.nii'):
            import gzip
            case_name = f.replace('.nii', '')
            temp_dir = os.path.join(os.path.dirname(input_folder), '_nnunet_input')
            os.makedirs(temp_dir, exist_ok=True)
            src = os.path.join(input_folder, f)
            dst = os.path.join(temp_dir, case_name + '_0000.nii.gz')
            with open(src, 'rb') as f_in:
                with gzip.open(dst, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return temp_dir, case_name

    return None, None


def main(input_folder, output_folder):
    import traceback
    try:
        _main_inner(input_folder, output_folder)
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        emit(0, f"推理异常: {tb}")
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)


def _main_inner(input_folder, output_folder):
    emit(2, "准备推理环境...")

    # 1. 准备干净的输入文件夹（只含目标文件，避免干扰）
    temp_input_folder, case_name = prepare_nnunet_input(input_folder)
    if case_name is None:
        emit(0, "错误: 输入文件夹中未找到 .nii.gz 文件")
        sys.exit(1)

    emit(5, f"输入用例: {case_name}")

    # 2. 导入 nnunet（这一步较慢）
    emit(8, "加载 nnU-Net 框架...")
    import torch
    from nnunet.inference.predict import predict_from_folder
    import nnunet.paths

    # 覆盖 nnunet 路径，指向项目目录下的模型（paths.py 硬编码了旧路径）
    project_root = os.path.dirname(os.path.abspath(__file__))
    nnunet.paths.network_training_output_dir = os.path.join(project_root, "nnunet", "nnUNet_trained_models", "nnUNet")
    nnunet.paths.base = os.path.join(project_root, "nnunet", "nnUNet_raw")
    nnunet.paths.preprocessing_output_dir = os.path.join(project_root, "nnunet", "nnUNet_preprocessed")

    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        emit(0, f"CUDA 检测失败: {e}")
        cuda_ok = False

    device = "GPU" if cuda_ok else "CPU"
    emit(10, f"使用设备: {device}")

    # 3. 配置模型路径（与 system/DeepN.py 一致）
    task_name = "Task001_Shoulder"
    model = "3d_fullres"
    trainer_class_name = nnunet.paths.default_trainer  # nnUNetTrainerV2
    plans_identifier = nnunet.paths.default_plans_identifier  # nnUNetPlansv2.1
    checkpoint_name = "model_final_checkpoint"

    model_folder = os.path.join(
        nnunet.paths.network_training_output_dir,
        model,
        task_name,
        trainer_class_name + "__" + plans_identifier
    )

    if not os.path.isdir(model_folder):
        emit(0, f"错误: 模型目录不存在: {model_folder}")
        sys.exit(1)

    emit(15, f"模型: {trainer_class_name}")
    emit(18, f"输入尺寸: 3D full resolution")

    # 4. 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 5. 开始推理
    emit(20, "开始 nnU-Net 推理（可能需要几分钟）...")

    try:
        predict_from_folder(
            model=model_folder,
            input_folder=temp_input_folder,
            output_folder=output_folder,
            folds=None,            # 自动检测 folds
            save_npz=False,
            num_threads_preprocessing=1,
            num_threads_nifti_save=1,
            lowres_segmentations=None,
            part_id=0,
            num_parts=1,
            tta=True,              # 测试时数据增强
            mixed_precision=True,
            overwrite_existing=True,
            mode="normal",
            step_size=0.5,
            checkpoint_name=checkpoint_name
        )
    except Exception as e:
        emit(0, f"推理失败: {str(e)}")
        sys.exit(1)

    emit(95, "推理完成，检查输出...")

    # 6. 确认输出文件存在
    result_file = os.path.join(output_folder, case_name + ".nii.gz")
    if os.path.exists(result_file):
        emit(100, json.dumps({
            "output": result_file,
            "case_name": case_name
        }))
    else:
        # nnunet 有时输出文件名不同，搜索一下
        output_files = [f for f in os.listdir(output_folder) if f.endswith('.nii.gz')]
        if output_files:
            result_file = os.path.join(output_folder, output_files[0])
            emit(100, json.dumps({
                "output": result_file,
                "case_name": case_name
            }))
        else:
            emit(0, "错误: 推理完成但未找到输出文件")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        emit(0, "错误: 用法: run_inference.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        emit(0, f"错误: 输入文件夹不存在: {input_folder}")
        sys.exit(1)

    main(input_folder, output_folder)
