"""
NIfTI 转 STL 脚本
将分割结果（多标签 NIfTI）提取非零标签并转为 STL 网格。
参照 ACL Backend_VTK_1.py 的 NiftiProcessor 流程。

用法: python nifti_to_stl.py <nifti_path> <stl_path> <label_id>

参数:
  nifti_path  - 分割结果 NIfTI 文件路径
  stl_path    - 输出 STL 文件路径（合并后的主文件）
  label_id    - 要提取的标签 ID (0=所有非零标签并分别输出, 1=Scapula, 2=Humerus)

输出文件（label_id=0 时）:
  stl_path          - 合并的完整网格
  stl_path同目录/label_1.stl - 肩胛骨
  stl_path同目录/label_2.stl - 肱骨

进度通过 stdout JSON 行报告给 Electron 主进程。
"""

import sys
import os
import json

import vtk


def emit(progress, message):
    """输出进度 JSON 行"""
    print(json.dumps({"progress": progress, "message": message}), flush=True)


def smooth_polydata(polydata):
    """
    对 vtkPolyData 进行平滑处理（参照 ACL NiftiProcessor.__stl_smoother）
    4 步平滑流程，返回平滑后的 polydata
    """
    # 第1步: 平滑
    smooth1 = vtk.vtkSmoothPolyDataFilter()
    smooth1.SetInputData(polydata)
    smooth1.SetNumberOfIterations(8)
    smooth1.SetRelaxationFactor(0.45)
    smooth1.FeatureEdgeSmoothingOff()
    smooth1.BoundarySmoothingOn()
    smooth1.Update()

    # 第2步: 减面
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smooth1.GetOutputPort())
    decimate.SetTargetReduction(0.27)
    decimate.PreserveTopologyOn()
    decimate.Update()

    # 第3步: Windowed Sinc 平滑
    smooth2 = vtk.vtkWindowedSincPolyDataFilter()
    smooth2.SetInputConnection(decimate.GetOutputPort())
    smooth2.SetNumberOfIterations(5)
    smooth2.SetFeatureEdgeSmoothing(True)
    smooth2.SetFeatureAngle(30.0)
    smooth2.Update()

    # 第4步: 最终平滑
    smooth3 = vtk.vtkSmoothPolyDataFilter()
    smooth3.SetInputConnection(smooth2.GetOutputPort())
    smooth3.SetNumberOfIterations(4)
    smooth3.SetRelaxationFactor(0.27)
    smooth3.FeatureEdgeSmoothingOff()
    smooth3.BoundarySmoothingOn()
    smooth3.Update()

    return smooth3.GetOutput()


def write_stl(polydata, stl_path):
    """将 polydata 写入 STL 文件"""
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_path)
    writer.SetInputData(polydata)
    writer.Write()


def nifti_to_stl(nifti_path, stl_path, label_id):
    """
    从 NIfTI 分割结果中提取指定标签并转为 STL。
    label_id=0 时：分别提取标签1和标签2，各自输出独立STL，再合并输出主STL。
    """
    # 1. 读取 NIfTI
    emit(10, "读取 NIfTI 文件...")
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()

    if reader.GetOutput().GetNumberOfPoints() == 0:
        emit(0, "错误: NIfTI 文件读取失败或为空")
        sys.exit(1)

    # 确保输出目录存在
    out_dir = os.path.dirname(stl_path)
    os.makedirs(out_dir, exist_ok=True)

    # 确定要提取的标签列表
    if label_id == 0:
        label_ids = [1, 2]
        label_names = {1: 'Humerus', 2: 'Scapula'}
    else:
        label_ids = [label_id]
        label_names = {label_id: f'label_{label_id}'}

    # 2. 对每个标签做阈值提取 + Marching Cubes + 平滑，分别保存
    append = vtk.vtkAppendPolyData()
    total_points = 0
    smoothed_polys = {}

    for i, lid in enumerate(label_ids):
        progress = 20 + int(25 * i / len(label_ids))
        emit(progress, f"提取标签 {lid}...")

        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(reader.GetOutputPort())
        threshold.ThresholdBetween(lid, lid)
        threshold.ReplaceInOn()
        threshold.SetInValue(1)
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0)
        threshold.Update()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(threshold.GetOutputPort())
        mc.SetValue(0, 1)
        mc.Update()

        n_pts = mc.GetOutput().GetNumberOfPoints()
        if n_pts > 0:
            # 平滑该标签的网格
            emit(progress + 5, f"平滑标签 {lid} ({n_pts} 顶点)...")
            smoothed = smooth_polydata(mc.GetOutput())
            smoothed_polys[lid] = smoothed
            append.AddInputData(smoothed)
            total_points += smoothed.GetNumberOfPoints()

            # 单独保存该标签的 STL
            label_stl_path = os.path.join(out_dir, f'{label_names[lid]}.stl')
            write_stl(smoothed, label_stl_path)
            emit(progress + 10, f"已保存: {label_names[lid]}.stl")

    if total_points == 0:
        emit(0, "错误: 未生成任何网格")
        sys.exit(1)

    # 3. 合并并保存主 STL（备用，不被前端读取）
    emit(80, "合并网格...")
    append.Update()
    merged = append.GetOutput()
    write_stl(merged, stl_path)

    emit(100, json.dumps({
        "output": out_dir,
        "mesh_dir": out_dir,
        "labels": list(smoothed_polys.keys())
    }))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        emit(0, "错误: 用法: nifti_to_stl.py <nifti_path> <stl_path> <label_id>")
        sys.exit(1)

    nifti_path = sys.argv[1]
    stl_path = sys.argv[2]
    label_id = int(sys.argv[3])

    if not os.path.exists(nifti_path):
        emit(0, f"错误: NIfTI 文件不存在: {nifti_path}")
        sys.exit(1)

    try:
        nifti_to_stl(nifti_path, stl_path, label_id)
    except Exception as e:
        emit(0, f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
