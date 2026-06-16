import time
import sys
import json

import open3d as o3d
import numpy as np
import copy
import math
from tqdm import tqdm
import pyvista as pv
import os
import nibabel

# ThreadSon 仅用 global_jindu 做进度追踪，用 stub 替代避免旧系统依赖链
class _ThreadSonStub:
    global_jindu = 0
ThreadSon = _ThreadSonStub()


def msk2segmentationresult(img, mask, result_path, index):  # 单独提取肩胛骨与肱骨
    # transfer mask to segmentation result,mask image is .nii.gz,1 means Humeral  ，2 means Scapula, 3 Humeral and Scapula

    nifti_img = nibabel.load(img)
    nifti_msk = nibabel.load(mask)
    # nifti_result = nibabel.load('E:/DATA/result/0003.nii.gz')
    img_affine = nifti_img.affine
    nifti_arrayimg = nifti_img.get_fdata()
    nifti_arraymsk = nifti_msk.get_fdata()
    if index == 1:
        index_name = 'Humeral'
        nifti_arraymsk[nifti_arraymsk != 1] = 0
    if index == 2:
        index_name = 'Scapula'
        nifti_arraymsk[nifti_arraymsk != 2] = 0
        nifti_arraymsk[nifti_arraymsk == 2] = 1
    if index == 3:
        index_name = 'All'
        nifti_arraymsk[nifti_arraymsk > 1] = 1
    result = nifti_arrayimg * nifti_arraymsk
    result = nibabel.Nifti1Image(result, img_affine, nifti_img.header)
    nibabel.save(result, os.path.join(result_path, f'{index_name}.nii.gz'))


class scapula():
    def __init__(self, file_name):
        self.mesh_pv = pv.read(file_name)
        n_cell = self.mesh_pv.n_cells
        triangles = []
        for i in range(n_cell):
            triangles.append(self.mesh_pv.get_cell(i).point_ids)
        triangles = np.array(triangles)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(self.mesh_pv.points))
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        self.guide_mesh = 0

        self.pcd = o3d.geometry.PointCloud()
        V_mesh = np.array(self.mesh.vertices)
        self.pcd.points = o3d.utility.Vector3dVector(V_mesh)

        self.change = []

    def select_points2(self, picked_id_pcd):
        a = self.pcd.points
        print(len(a))
        self.p1 = a[picked_id_pcd[0]]
        self.p2 = a[picked_id_pcd[1]]
        self.p3 = a[picked_id_pcd[2]]
        self.id = picked_id_pcd

    def select_points1(self):
        def pick_points(pcd):
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            return vis.get_picked_points()

        value = self.pcd.points
        picked_id_pcd = pick_points(self.pcd)
        self.p1 = value[picked_id_pcd[0]]
        self.p2 = value[picked_id_pcd[1]]
        self.p3 = value[picked_id_pcd[2]]
        self.id = picked_id_pcd

    def computer_circle(self):
        def find_center(p1, p2, p3):
            x1 = p1[0]
            y1 = p1[1]
            z1 = p1[2]
            x2 = p2[0]
            y2 = p2[1]
            z2 = p2[2]
            x3 = p3[0]
            y3 = p3[1]
            z3 = p3[2]
            a1 = (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2)
            b1 = -(x1 * z2 - x2 * z1 - x1 * z3 + x3 * z1 + x2 * z3 - x3 * z2)
            c1 = (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2)
            d1 = -(x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1)
            a2 = 2 * (x2 - x1)
            b2 = 2 * (y2 - y1)
            c2 = 2 * (z2 - z1)
            d2 = x1 * x1 + y1 * y1 + z1 * z1 - x2 * x2 - y2 * y2 - z2 * z2
            a3 = 2 * (x3 - x1)
            b3 = 2 * (y3 - y1)
            c3 = 2 * (z3 - z1)
            d3 = x1 * x1 + y1 * y1 + z1 * z1 - x3 * x3 - y3 * y3 - z3 * z3
            x = -(b1 * c2 * d3 - b1 * c3 * d2 - b2 * c1 * d3 + b2 * c3 * d1 + b3 * c1 * d2 - b3 * c2 * d1) / (
                    a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
            y = (a1 * c2 * d3 - a1 * c3 * d2 - a2 * c1 * d3 + a2 * c3 * d1 + a3 * c1 * d2 - a3 * c2 * d1) / (
                    a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
            z = -(a1 * b2 * d3 - a1 * b3 * d2 - a2 * b1 * d3 + a2 * b3 * d1 + a3 * b1 * d2 - a3 * b2 * d1) / (
                    a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
            return x, y, z

        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        x, y, z = find_center(p1, p2, p3)
        r_circle = np.sqrt((p1[0] - x) ** 2 + (p1[1] - y) ** 2 + (p1[2] - z) ** 2)

        self.center = [x, y, z]
        self.r = r_circle

    def move_center_to_O(self):
        def change_mesh(mesh_first, x, y, z):
            a = [-x, -y, -z]
            mesh_second = copy.deepcopy(mesh_first).translate(tuple(a))
            mesh_second.compute_vertex_normals()
            return mesh_second

        x = self.center[0]
        y = self.center[1]
        z = self.center[2]
        self.mesh = change_mesh(self.mesh, x, y, z)

        self.change.append(['translate', (x, y, z)])

    def find_vector(self, filename, d):

        def find_normal_vector(p1, p2, p3):
            x1 = p1[0]
            y1 = p1[1]
            z1 = p1[2]
            x2 = p2[0]
            y2 = p2[1]
            z2 = p2[2]
            x3 = p3[0]
            y3 = p3[1]
            z3 = p3[2]
            a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
            b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
            c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            return [a, b, c]

        def find_dis(point, mesh):
            mesh2 = copy.deepcopy(mesh)
            mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh2)
            query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
            return scene.compute_signed_distance(query_point)

        def amount_point(normal_vector, mesh_second):
            length = 0.1
            j = 0
            for i in range(100):
                vector_point = normal_vector * (length * i)
                if find_dis(vector_point, mesh_second) < 0:
                    j = j + 1
            return j

        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_angle(p1, p2, p3):
            l1 = dis(p1, p2)
            l2 = dis(p2, p3)
            l3 = dis(p1, p3)
            if l1 * l2 == 0:
                print('出现错误', p1, p2, p3)
            return math.acos((l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)) / np.pi

        def rotate_mesh(normal_vector):
            point_coordinate = [0, 0, 0]
            # 向量OB，也就是法向量
            vector_ob = [normal_vector[0], normal_vector[1], normal_vector[2]]

            # 法向量与z轴的夹角
            theta = find_angle(vector_ob, [0, 0, 0], [0, 0, 1])

            # 第一次旋转
            vector_ob2 = [0, np.sin(np.pi * theta), np.cos(np.pi * theta)]
            alpha = find_angle(vector_ob, [0, 0, np.cos(np.pi * theta)], vector_ob2)
            if vector_ob[0] < 0:
                alpha = - alpha

            R = self.mesh.get_rotation_matrix_from_xyz((0, 0, np.pi * alpha))
            mesh_third = copy.deepcopy(self.mesh)
            mesh_third.rotate(R, center=point_coordinate)

            self.change.append(['rotate', [0, 0, - np.pi * alpha]])
            # self.change.append(['rotate', self.mesh.get_rotation_matrix_from_xyz((0, 0, - np.pi * alpha))])

            # 第二次旋转
            R = self.mesh.get_rotation_matrix_from_xyz((np.pi * theta, 0, 0))
            mesh_fourth = copy.deepcopy(mesh_third)
            mesh_fourth.rotate(R, center=point_coordinate)

            self.change.append(['rotate', [- np.pi * theta, 0, 0]])
            # self.change.append(['rotate', self.mesh.get_rotation_matrix_from_xyz((- np.pi * theta, 0, 0))])
            return mesh_fourth

        def rotate_mesh2(normal_vector, mesh):
            point_coordinate = (0, 0, 0)
            # 向量OB，也就是法向量
            vector_ob = [normal_vector[0], normal_vector[1], normal_vector[2]]

            # 法向量与z轴的夹角
            mesh_second = copy.deepcopy(mesh)
            theta = find_angle(vector_ob, [0, 0, 0], [0, 1, 0])
            R = mesh_second.get_rotation_matrix_from_xyz((0, 0, theta * np.pi))
            mesh_third = copy.deepcopy(mesh)
            mesh_third.rotate(R, center=point_coordinate)

            self.change.append(['rotate', [0, 0, - theta * np.pi]])
            # self.change.append(['rotate', mesh_second.get_rotation_matrix_from_xyz((0, 0, - theta * np.pi))])
            return mesh_third

        def change_cylinder(mesh_cylinder1, up):
            point_coordinate = [0, 0, 0]
            a = [0, 0, 0] - up / 2
            mesh_cylinder2 = copy.deepcopy(mesh_cylinder1).translate(tuple(a))
            mesh_cylinder2.compute_vertex_normals()
            theta1 = find_angle(mesh_cylinder2.get_center(), [0, 0, 0], [0, 0, 1])
            R = mesh_cylinder2.get_rotation_matrix_from_xyz((0, np.pi * theta1, 0))
            mesh_cylinder = copy.deepcopy(mesh_cylinder2)
            mesh_cylinder.rotate(R, center=point_coordinate)
            return mesh_cylinder

        def change_cylinder2(mesh_cylinder1):
            point_coordinate = [0, 0, 0]
            a = [0, 0, 0]
            mesh_cylinder2 = copy.deepcopy(mesh_cylinder1).translate(tuple(a))
            mesh_cylinder2.compute_vertex_normals()
            theta1 = find_angle(mesh_cylinder2.get_center(), [0, 0, 0], [0, 0, 1])
            R = mesh_cylinder2.get_rotation_matrix_from_xyz((np.pi * theta1, 0, 0))
            mesh_cylinder = copy.deepcopy(mesh_cylinder2)
            mesh_cylinder.rotate(R, center=point_coordinate)
            return mesh_cylinder

        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        normal_vector_zero = find_normal_vector(p1, p2, p3)
        normal_vector_module = (normal_vector_zero[0] ** 2 + normal_vector_zero[1] ** 2 + normal_vector_zero[
            2] ** 2) ** 0.5
        normal_vector = (np.asarray(normal_vector_zero)) / normal_vector_module
        normal_vector_back = normal_vector * (-1)
        numeber = amount_point(normal_vector, self.mesh)
        numeber_back = amount_point(normal_vector_back, self.mesh)
        if numeber_back > numeber:
            normal_vector = normal_vector_back
        self.mesh = rotate_mesh(normal_vector)

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        self.mesh_frame.compute_vertex_normals()

        p1 = np.array(self.mesh.vertices)[self.id[0]]
        vector2 = np.array(p1) / ((p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2) ** 0.5)
        self.mesh = rotate_mesh2(vector2, self.mesh)
        self.cylinder10 = o3d.geometry.TriangleMesh.create_cylinder(radius=d / 2,
                                                                    height=50)
        self.cylinder10 = change_cylinder(self.cylinder10, np.asarray(self.cylinder10.vertices)[0] -
                                          np.asarray(self.cylinder10.vertices)[1])
        self.cylinder101pv = pv.read(filename)
        n_cell = self.cylinder101pv.n_cells
        triangles = []
        for i in range(n_cell):
            triangles.append(self.cylinder101pv.get_cell(i).point_ids)
        triangles = np.array(triangles)

        self.cylinder101 = o3d.geometry.TriangleMesh()
        self.cylinder101.vertices = o3d.utility.Vector3dVector(np.array(self.cylinder101pv.points))
        self.cylinder101.triangles = o3d.utility.Vector3iVector(triangles)
        self.cylinder101.compute_vertex_normals()
        self.cylinder101 = change_cylinder2(self.cylinder101)

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        self.mesh_frame.compute_vertex_normals()

    def find_nail(self, theta1, theta2, num_point=400):
        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_dis2(point):
            query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
            return scene.compute_signed_distance(query_point)

        mesh = self.mesh
        point_coordinate = (0, 0, 0)
        mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh2)

        # 1.设定步长，角度1是1°，角度2是18°；已在函数传递中完成

        # 2.初始化记录器
        location = [0, [], []]  # 长度，点的位置，圆柱的位置

        # 3.开始穷举
        p = []
        know = []
        for i in range(int(5 / theta1)):
            for j in range(int(20 / theta2)):
                p.append([i, j])
        tq = tqdm(p, desc="请等待")
        k1 = 40 / len(tq)
        for z in tq:
            ThreadSon.global_jindu = ThreadSon.global_jindu + k1
            print(ThreadSon.global_jindu)
            i = z[0]
            j = z[1]

            # 3.1.得出当前需要计算的圆柱位置，并将位于初始位置的圆柱旋转到那里
            theta_y = 10 + theta1 * i
            theta_z = theta2 * j - 10
            R = self.cylinder10.get_rotation_matrix_from_xyz((0, theta_z * np.pi / 180, 0))
            mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
            mesh_cylinderchange1.rotate(R, center=point_coordinate)
            R = self.cylinder10.get_rotation_matrix_from_xyz((theta_y * np.pi / 180, 0, 0))
            mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
            mesh_cylinderchange.rotate(R, center=point_coordinate)

            # 3.2.对当前圆柱位置进行判定，计算算法为：对于圆柱的每一个点，沿着x轴正负方向各走200个单位长度，如果有一侧全部在模型外侧，则这个点在模型外侧。找到在模型外侧且离圆心最近的钉子上的点。
            dis_origin = 100
            pcd2 = mesh_cylinderchange.sample_points_uniformly(number_of_points=num_point)
            point = np.asarray(pcd2.points)
            point_dis_coordinate = np.array([dis(point[k], point_coordinate) for k in range(num_point)])

            for k in range(num_point):
                if (point_dis_coordinate[k] >= dis_origin) or (point_dis_coordinate[k] <= 5):
                    continue

                judge1 = -1
                judge2 = -1
                position_x = np.arange(0, 40, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z), axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2 >= 0).all():
                    judge1 = 1

                position_x = np.arange(-40, 0, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z), axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2 >= 0).all():
                    judge2 = 1

                if (judge1 > 0 or judge2 > 0) and (dis_origin > point_dis_coordinate[k]):
                    dis_origin = point_dis_coordinate[k]
                    know = point[k]

            if (dis_origin != 100) and (dis_origin > location[0]):
                location[0] = dis_origin
                location[1] = know
                location[2] = [i, j]

        location[2][0] = theta1 * location[2][0] + 10
        location[2][1] = theta2 * location[2][1] - 10
        self.location = location

        # 与搜索循环一致：先绕Y轴(angle2)，再绕X轴(angle1)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, location[2][1] * np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder = copy.deepcopy(mesh_cylinderchange)

        R = self.cylinder101.get_rotation_matrix_from_xyz((0, location[2][1] * np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder101)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder101.get_rotation_matrix_from_xyz((location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder101 = copy.deepcopy(mesh_cylinderchange)

    def find_handle(self, file_name):
        self.cylinder2 = o3d.io.read_triangle_mesh(file_name)
        self.cylinder2.compute_vertex_normals()
        point_coordinate = (0, 0, 0)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, self.location[2][1] * np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder2)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((self.location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder2 = copy.deepcopy(mesh_cylinderchange)

    def find_guide(self):
        mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh1)
        a = np.array([])
        r_circle = self.r
        # r_circle *= 1.75
        r_circle /= 2 / 3
        p1 = np.array(self.mesh.vertices)
        p1 = p1[self.id[0]]

        p = []
        for i in range(180):
            for j in range(180):
                for k in range(5):  # 10
                    p.append([i, j, k])
        tq = tqdm(p, desc="请等待")
        k2 = 10 / len(tq)
        for z1 in tq:
            print(ThreadSon.global_jindu)
            ThreadSon.global_jindu = ThreadSon.global_jindu + k2
            i = z1[0]
            j = z1[1]
            k = z1[2]
            x = (-r_circle / 2) + r_circle / 180 * i
            y = (p1[1]) - r_circle / 180 * j
            z = (-3) + 0.8 * k
            # x=(-r_circle / 2) + r_circle / 180 * i; y= - r_circle / 180 * j; z = 0.8 * k
            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)
            ans = scene.compute_closest_points(query_point)
            points = ans['points'].numpy()
            triangle = ans['primitive_ids'][0].item()
            a = np.append(a, triangle)
            a = a.astype(int)

        mesh2 = copy.deepcopy(self.mesh)
        mesh2.triangles = o3d.utility.Vector3iVector(
            np.asarray(mesh2.triangles)[a])
        mesh2.triangle_normals = o3d.utility.Vector3dVector(
            np.asarray(mesh2.triangle_normals)[a])
        mesh2.paint_uniform_color([0.1, 0.1, 0.7])

        # o3d.visualization.draw_geometries([mesh2, self.cylinder2])

        mesh2.compute_vertex_normals()
        pcd1 = mesh2.sample_points_uniformly(number_of_points=10000)
        print(ThreadSon.global_jindu)
        xyz = np.asarray(pcd1.points)
        xyz2 = []
        for i in range(10000):
            if (xyz[i][0]) ** 2 + (xyz[i][1]) ** 2 > 2.4 ** 2:
                xyz2.append(xyz[i])
        xyz2 = np.array(xyz2)
        xyz = copy.deepcopy(xyz2)
        p = []
        z1 = []
        for i in range(xyz.shape[0]):
            for j in range(10):
                z1.append([i, j])
        tqd = tqdm(z1, desc="请等待")
        k3 = 10 / len(tqd)
        for z in tqd:
            print(ThreadSon.global_jindu)
            ThreadSon.global_jindu = ThreadSon.global_jindu + k3
            i = z[0]
            j = z[1]
            q = [xyz[i, 0], xyz[i, 1], xyz[i, 2] - j * 0.5]
            p.append(q)
        ThreadSon.global_jindu = 91
        p = np.array(p)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(p)
        self.guide_pcd = pcd2
        ThreadSon.global_jindu = 92
        mesh4 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd2, alpha=2)
        ThreadSon.global_jindu = 94
        mesh4.compute_vertex_normals()
        ThreadSon.global_jindu = 95
        mesh4.paint_uniform_color([0, 0.8, 0.8])

        self.guide_mesh = mesh4
        self.guide_mesh.paint_uniform_color([0.1, 0.1, 0.7])

        print(ThreadSon.global_jindu)

    def show(self, l):
        pl = pv.Plotter()
        for i in range(len(l)):
            o3d.io.write_triangle_mesh('%d.stl' % i, l[i])
            p = pv.read('%d.stl' % i)
            _ = pl.add_mesh(p)

        pl.camera_position = 'xz'
        pl.show()

    def save(self, path):
        o3d.io.write_triangle_mesh(os.path.join(path, 'nail1.stl'), self.cylinder)
        o3d.io.write_triangle_mesh(os.path.join(path, 'guide.stl'), self.guide_mesh)
        o3d.io.write_triangle_mesh(os.path.join(path, 'handle.stl'), self.cylinder2)
        o3d.io.write_triangle_mesh(os.path.join(path, 'jizuo.stl'), self.jizuo)

    def save1(self, path):
        o3d.io.write_triangle_mesh(os.path.join(path, 'guide.stl'), self.guide_mesh)
        o3d.io.write_triangle_mesh(os.path.join(path, 'handle.stl'), self.cylinder2)
        o3d.io.write_triangle_mesh(os.path.join(path, 'jizuo.stl'), self.jizuo)

    def go_back(self):
        sphere_x = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate((1, 0, 0))
        sphere_y = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate((0, 1, 0))
        sphere_z = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate((0, 0, 1))

        n = len(self.change)
        for i in range(n):
            p = self.change[n - 1 - i]
            if p[0] == 'translate':
                self.mesh = copy.deepcopy(self.mesh).translate(p[1])
                self.cylinder = copy.deepcopy(self.cylinder).translate(p[1])
                self.cylinder2 = copy.deepcopy(self.cylinder2).translate(p[1])
                self.guide_mesh = copy.deepcopy(self.guide_mesh).translate(p[1])
                self.jizuo = copy.deepcopy(self.jizuo).translate(p[1])
                self.cylinder101 = copy.deepcopy(self.cylinder101).translate(p[1])

                sphere_x = copy.deepcopy(sphere_x).translate(p[1])
                sphere_y = copy.deepcopy(sphere_y).translate(p[1])
                sphere_z = copy.deepcopy(sphere_z).translate(p[1])
            else:
                rotate = self.mesh.get_rotation_matrix_from_xyz(p[1])
                self.mesh = self.mesh.rotate(rotate, center=(0, 0, 0))
                self.cylinder = self.cylinder.rotate(rotate, center=(0, 0, 0))
                self.cylinder2 = self.cylinder2.rotate(rotate, center=(0, 0, 0))
                self.guide_mesh = self.guide_mesh.rotate(rotate, center=(0, 0, 0))
                self.jizuo = self.jizuo.rotate(rotate, center=(0, 0, 0))
                self.cylinder101 = self.cylinder101.rotate(rotate, center=(0, 0, 0))

                sphere_x = sphere_x.rotate(rotate, center=(0, 0, 0))
                sphere_y = sphere_y.rotate(rotate, center=(0, 0, 0))
                sphere_z = sphere_z.rotate(rotate, center=(0, 0, 0))

        self.x = sphere_x.get_center()
        self.y = sphere_y.get_center()
        self.z = sphere_z.get_center()

    def go_to(self, change):

        n = len(change)
        for i in range(n):
            p = change[i]
            p[1] = [-p[1][0], -p[1][1], -p[1][2]]
            if p[0] == 'translate':
                self.mesh = copy.deepcopy(self.mesh).translate(p[1])
                self.cylinder = copy.deepcopy(self.cylinder).translate(p[1])
                self.cylinder101 = copy.deepcopy(self.cylinder101).translate(p[1])
            else:
                rotate = self.mesh.get_rotation_matrix_from_xyz(p[1])
                self.mesh = self.mesh.rotate(rotate, center=(0, 0, 0))
                self.cylinder = self.cylinder.rotate(rotate, center=(0, 0, 0))
                self.cylinder101 = self.cylinder101.rotate(rotate, center=(0, 0, 0))

    def find_jizuo(self, filename):

        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_angle(p1, p2, p3):
            l1 = dis(p1, p2);
            l2 = dis(p2, p3);
            l3 = dis(p1, p3)
            cos = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)
            return math.acos(cos) / np.pi

        def change_jizuo(mesh_jizuo1):
            a = - mesh_jizuo1.get_center() + [0, 0, 0] + [0, 0, 2]
            mesh_jizuo = copy.deepcopy(mesh_jizuo1).translate(tuple(a))
            mesh_jizuo.compute_vertex_normals()
            return mesh_jizuo

        self.jizuo = o3d.io.read_triangle_mesh(filename)
        self.jizuo.compute_vertex_normals()
        # self.jizuo = copy.deepcopy(change_jizuo(self.jizuo)) # 注意 这里被注释后是正常的移动
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, self.location[2][1] * np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.jizuo)
        mesh_cylinderchange1.rotate(R, center=[0, 0, 0])
        R = self.cylinder10.get_rotation_matrix_from_xyz((self.location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=[0, 0, 0])

        mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh1)
        center = self.jizuo.get_center()
        query_point = o3d.core.Tensor([center], dtype=o3d.core.Dtype.Float32)
        unsigned_distance = scene.compute_distance(query_point).numpy() + 2
        unsigned_distance = float(unsigned_distance)

        theta1 = self.location[2][0] * np.pi / 180
        theta2 = self.location[2][1] * np.pi / 180
        y = - np.sin(theta1) * unsigned_distance
        x = np.cos(theta1) * np.cos(theta2) * unsigned_distance
        z = np.cos(theta1) * np.sin(theta2) * unsigned_distance

        self.jizuo = copy.deepcopy(mesh_cylinderchange).translate(tuple([z, y, x]))


def find_distance(meshp, nailp, center):
    def dis(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    def find_dis2(point):
        query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
        return scene.compute_signed_distance(query_point)

    mesh0 = pv.wrap(meshp);
    nail0 = pv.wrap(nailp)  # 这里应该是可以直接和vtk联动的，不需要采用保存-读入的方式
    n_cell = mesh0.n_cells
    triangles = []
    for i in range(n_cell):
        triangles.append(mesh0.get_cell(i).point_ids)
    triangles = np.array(triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh0.points))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    n_cell = nail0.n_cells
    triangles = []
    for i in range(n_cell):
        triangles.append(nail0.get_cell(i).point_ids)
    triangles = np.array(triangles)
    nail = o3d.geometry.TriangleMesh()
    nail.vertices = o3d.utility.Vector3dVector(np.array(nail0.points))
    nail.triangles = o3d.utility.Vector3iVector(triangles)
    nail.compute_vertex_normals()

    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh2)

    dis_origin = 100
    pcd2 = nail.sample_points_uniformly(number_of_points=200)
    point = np.asarray(pcd2.points)
    point_dis_coordinate = np.array([dis(point[k], center) for k in range(200)])

    for k in range(200):
        if (point_dis_coordinate[k] >= dis_origin) or (point_dis_coordinate[k] <= 5):
            continue

        judge1 = -1
        judge2 = -1
        position_x = np.arange(0, 40, 0.1) + point[k][0]
        position_x = position_x.reshape(-1, 1)
        position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
        position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
        position = np.concatenate((position_x, position_y, position_z), axis=1)
        dis2 = find_dis2(position)
        dis2 = dis2.reshape(-1)
        if (dis2 >= 0).all():
            judge1 = 1

        position_x = np.arange(-40, 0, 0.1) + point[k][0]
        position_x = position_x.reshape(-1, 1)
        position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
        position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
        position = np.concatenate((position_x, position_y, position_z), axis=1)
        dis2 = find_dis2(position)
        dis2 = dis2.reshape(-1)
        if (dis2 >= 0).all():
            judge2 = 1

        if (judge1 > 0 or judge2 > 0) and (dis_origin > point_dis_coordinate[k]):
            dis_origin = point_dis_coordinate[k]
            know = point[k]

    return dis_origin


def find_angle(nail0, nail1, x, y, z):
    '''
    这里需要输入一些参数
    nail0是圆心坐标 写在了self.center中
    nail1是钉末端中心点坐标 可以通过读取模型(圆柱体钉子)-获取模型第二个点来得到
        np.asarray(mesh.vertices)[1] # 基于open3d
        mesh.points[1] # 基于pyvista
    xyz分别是基于更新版自动找钉给的坐标点 写在了self.x/self.y/self.z中
    '''
    p1 = np.array(x)
    p2 = np.array(y)
    p3 = np.array(z)
    o = np.array(nail0)
    d = np.array(nail1)

    # 计算垂直于xz面高度
    n_xz = np.cross(o - p1, o - p3)
    h_xz = np.linalg.norm(np.dot(o - d, n_xz)) / np.linalg.norm(n_xz)

    # 计算垂直于xy面高度
    n_xy = np.cross(o - p1, o - p2)
    h_xy = np.linalg.norm(np.dot(o - d, n_xy)) / np.linalg.norm(n_xy)

    # 计算垂直于yz面高度
    n_yz = np.cross(o - p2, o - p3)
    h_yz = np.linalg.norm(np.dot(o - d, n_yz)) / np.linalg.norm(n_yz)

    # 计算上下倾角（和z轴夹角）
    theta1 = np.arctan(h_yz / h_xy) / np.pi * 180

    # 计算前后倾角（和z轴夹角）
    theta2 = np.arctan(h_xy / h_xz) / np.pi * 180

    return 90 - theta2, -theta1


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMPLATE_DIR = os.path.join(SCRIPT_DIR, 'moban')

    # 全局错误处理
    import traceback
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(json.dumps({"progress": 0, "message": f"致命错误: {error_msg}"}), flush=True)
        sys.stderr.write(f"FATAL ERROR: {error_msg}\n")
        sys.stderr.flush()
    sys.excepthook = global_exception_handler

    # ---- 模式: --adjust ----
    # 用法: python MasterWuVtkStlMaker.py --adjust meshPath nailPath centerJson
    if len(sys.argv) > 1 and sys.argv[1] == '--adjust':
        meshPath = sys.argv[2]
        nailPath = sys.argv[3]
        center = json.loads(sys.argv[4])

        s = scapula(meshPath)
        s.pcd = o3d.geometry.PointCloud()
        s.pcd.points = o3d.utility.Vector3dVector(np.asarray(s.mesh.vertices))

        nail_mesh = o3d.io.read_triangle_mesh(nailPath)
        nail_pv = pv.read(nailPath)

        # 计算入钉长度
        length = find_distance(s.mesh, nail_mesh, center)
        # 获取钉子方向
        nail_pts = np.asarray(nail_mesh.vertices)
        if len(nail_pts) >= 2:
            nail_center = nail_pts[0].tolist()
            nail_end = nail_pts[1].tolist()
        else:
            nail_center = center
            nail_end = [center[0], center[1], center[2] + 1]

        # 计算角度（需要坐标系参考点，从 cfg 读取或使用默认值）
        cfg_path = os.path.join(os.path.dirname(meshPath), 'mesh.cfg')
        x_ref = [1, 0, 0]
        y_ref = [0, 1, 0]
        z_ref = [0, 0, 1]
        if os.path.exists(cfg_path):
            import configparser
            conf = configparser.ConfigParser()
            conf.read(cfg_path)
            if 'parameter' in conf:
                cfg = conf['parameter']
                try:
                    x_ref = [float(cfg.get('x1', 1)), float(cfg.get('x2', 0)), float(cfg.get('x3', 0))]
                    y_ref = [float(cfg.get('y1', 0)), float(cfg.get('y2', 1)), float(cfg.get('y3', 0))]
                    z_ref = [float(cfg.get('z1', 0)), float(cfg.get('z2', 0)), float(cfg.get('z3', 1))]
                except:
                    pass

        angle1, angle2 = find_angle(center, nail_end, x_ref, y_ref, z_ref)
        print(json.dumps({"length": length, "angle1": angle1, "angle2": angle2}))
        sys.exit(0)

    # ---- 模式: --regenerate ----
    # 用法: python MasterWuVtkStlMaker.py --regenerate dirPath paramsJson [meshDir]
    if len(sys.argv) > 1 and sys.argv[1] == '--regenerate':
        dirPath = sys.argv[2]
        params = json.loads(sys.argv[3])
        # meshDir: mesh.stl 所在目录（可选，默认为 dirPath）
        meshDir = sys.argv[4] if len(sys.argv) > 4 else dirPath

        def emit(progress, message):
            print(json.dumps({"progress": progress, "message": message}), flush=True)

        try:
            emit(5, "加载骨骼模型...")
            mesh_path = os.path.join(meshDir, 'mesh.stl')
            if not os.path.exists(mesh_path):
                emit(0, f"错误: 骨骼模型不存在: {mesh_path}")
                sys.exit(1)
            s = scapula(mesh_path)

            # 读取 cfg 获取参数
            import configparser
            cfg_path = os.path.join(dirPath, 'mesh.cfg')
            if not os.path.exists(cfg_path):
                emit(0, f"错误: 配置文件不存在: {cfg_path}")
                sys.exit(1)
            conf = configparser.ConfigParser()
            conf.read(cfg_path)
            cfg = conf['parameter']

            # 设置点和 id（验证顶点 ID 有效性）
            s.pcd = o3d.geometry.PointCloud()
            s.pcd.points = o3d.utility.Vector3dVector(np.asarray(s.mesh.vertices))
            n_verts = np.asarray(s.pcd.points).shape[0]
            s.id = [int(cfg['id1']), int(cfg['id2']), int(cfg['id3'])]

            if any(i >= n_verts or i < 0 for i in s.id):
                emit(10, f"[regenerate] ID out of range, fallback to center")
                s.center = [float(cfg['center1']), float(cfg['center2']), float(cfg['center3'])]
                mesh_points = np.asarray(s.pcd.points)
                center_pt = np.array(s.center)
                dists = np.linalg.norm(mesh_points - center_pt, axis=1)
                sorted_ids = np.argsort(dists)
                s.id = [sorted_ids[0], sorted_ids[1], sorted_ids[2]]
                s.p1 = mesh_points[s.id[0]]
                s.p2 = mesh_points[s.id[1]]
                s.p3 = mesh_points[s.id[2]]
            else:
                s.p1 = np.asarray(s.pcd.points)[s.id[0]]
                s.p2 = np.asarray(s.pcd.points)[s.id[1]]
                s.p3 = np.asarray(s.pcd.points)[s.id[2]]

            s.center = [float(cfg['center1']), float(cfg['center2']), float(cfg['center3'])]
            s.r = float(cfg['r'])

            # 新参数（来自 AI 微调或手动调整）
            # 优先使用原始旋转参数（raw_angle），回退到 location2/3（旧版 cfg 兼容）
            default_raw1 = float(cfg.get('raw_angle1', cfg.get('location2', 0)))
            default_raw2 = float(cfg.get('raw_angle2', cfg.get('location3', 0)))
            new_length = params.get('length', float(cfg['location1']))
            new_raw1 = params.get('raw_angle1', params.get('angle1', default_raw1))
            new_raw2 = params.get('raw_angle2', params.get('angle2', default_raw2))
            s.location = [new_length, [], [new_raw1, new_raw2]]

            # ---- 与默认模式完全一致的流程 ----

            # 1. 坐标变换：移到原点
            emit(15, "坐标变换...")
            s.move_center_to_O()

            # 2. 计算方向：建立坐标系，生成 cylinder10 模板
            emit(20, "计算方向...")
            nail_path = os.path.join(TEMPLATE_DIR, 'nail.stl')
            s.find_vector(nail_path if os.path.exists(nail_path) else mesh_path, 5)

            # 3. 从模板生成钉子（与 find_nail 搜索循环一致：先绕Y轴 angle2，再绕X轴 angle1）
            emit(30, "生成钉子...")
            point_coordinate = (0, 0, 0)
            angle1 = s.location[2][0]
            angle2 = s.location[2][1]

            R = s.mesh.get_rotation_matrix_from_xyz((0, angle2 * np.pi / 180, 0))
            mesh_tmp = copy.deepcopy(s.cylinder10)
            mesh_tmp.rotate(R, center=point_coordinate)
            R = s.mesh.get_rotation_matrix_from_xyz((angle1 * np.pi / 180, 0, 0))
            mesh_tmp.rotate(R, center=point_coordinate)
            s.cylinder = copy.deepcopy(mesh_tmp)

            # 4. 缩放钉子到目标长度
            nail_verts = np.asarray(s.cylinder.vertices)
            nail_axis = nail_verts[1] - nail_verts[0]
            nail_axis_len = np.linalg.norm(nail_axis)
            if nail_axis_len > 0:
                nail_axis_dir = nail_axis / nail_axis_len
                scale_factor = s.location[0] / nail_axis_len
                for i in range(len(nail_verts)):
                    v = nail_verts[i]
                    axial = np.dot(v - nail_verts[0], nail_axis_dir)
                    radial = (v - nail_verts[0]) - axial * nail_axis_dir
                    nail_verts[i] = nail_verts[0] + axial * scale_factor * nail_axis_dir + radial
                s.cylinder.vertices = o3d.utility.Vector3dVector(nail_verts)
                s.cylinder.compute_vertex_normals()

            # 5. 生成手柄、导板、基座（与默认模式一致）
            emit(50, "生成手柄...")
            handle_path = os.path.join(TEMPLATE_DIR, 'handle.stl')
            if os.path.exists(handle_path):
                s.find_handle(handle_path)
            else:
                s.cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=30)

            emit(60, "生成导板...")
            s.find_guide()

            emit(85, "生成基座...")
            jizuo_path = os.path.join(TEMPLATE_DIR, 'jizuo.stl')
            if os.path.exists(jizuo_path):
                s.find_jizuo(jizuo_path)
            else:
                s.jizuo = o3d.geometry.TriangleMesh.create_sphere(radius=5)

            emit(92, "坐标还原...")
            s.go_back()

            emit(96, "保存文件...")
            s.save(dirPath)

            # 计算显示角度（find_angle 输出，仅供参考显示）
            nail_center = np.asarray(s.cylinder.vertices)[0].tolist()
            nail_end = np.asarray(s.cylinder.vertices)[1].tolist()
            disp_angle1, disp_angle2 = find_angle(nail_center, nail_end, s.x, s.y, s.z)

            # 更新 cfg（使用新参数值，保留原始旋转参数供下次 regenerate 使用）
            conf.set("parameter", "location1", str(new_length))
            conf.set("parameter", "location2", str(disp_angle1))
            conf.set("parameter", "location3", str(disp_angle2))
            conf.set("parameter", "raw_angle1", str(new_raw1))
            conf.set("parameter", "raw_angle2", str(new_raw2))
            with open(cfg_path, 'w') as f:
                conf.write(f)

            emit(100, "重新生成完成")
            sys.exit(0)
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            emit(0, f"错误: {str(e)}\n{error_msg}")
            sys.exit(1)

    # ---- 默认模式: 从3点生成 ----
    # 参数: jsonFpath modelPath [saveDir]
    # jsonFpath: [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]] 世界坐标
    # modelPath: 骨骼 STL 文件路径
    # saveDir: 输出目录（可选，默认为模型所在目录）
    jsonFpath = sys.argv[1]
    modelPath = sys.argv[2]
    saveDir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(modelPath)

    points = json.loads(jsonFpath)
    os.makedirs(saveDir, exist_ok=True)

    def emit(progress, message):
        print(json.dumps({"progress": progress, "message": message}), flush=True)

    # 1. 加载骨骼模型
    emit(5, "加载骨骼模型...")
    s = scapula(modelPath)
    emit(10, "骨骼模型加载完成")

    # 2. 直接设置3个世界坐标点
    s.p1 = np.array(points[0])
    s.p2 = np.array(points[1])
    s.p3 = np.array(points[2])
    mesh_points = np.asarray(s.pcd.points)
    s.id = []
    for pt in [s.p1, s.p2, s.p3]:
        dists = np.linalg.norm(mesh_points - pt, axis=1)
        s.id.append(np.argmin(dists))

    # 3. 计算圆心
    emit(15, "计算圆心...")
    s.computer_circle()

    # 4. 移到原点
    emit(20, "坐标变换...")
    s.move_center_to_O()

    # 5. 找方向
    emit(25, "计算钉子方向...")
    nail_path = os.path.join(TEMPLATE_DIR, 'nail.stl')
    if not os.path.exists(nail_path):
        mesh_pv = pv.read(modelPath)
        bounds = mesh_pv.bounds
        diameter = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.05
    else:
        diameter = 5
    s.find_vector(nail_path if os.path.exists(nail_path) else modelPath, diameter)

    # 6. 找钉子
    emit(30, "穷举计算最优钉子位置...")
    s.find_nail(1, 18)
    emit(70, "钉子位置计算完成")

    # 6.5 缩放钉子到目标长度
    # 钉子模板固定50mm，location[0]是插入深度参数，需要缩放钉子使其物理长度匹配
    nail_verts = np.asarray(s.cylinder.vertices)
    nail_axis = nail_verts[1] - nail_verts[0]
    nail_axis_len = np.linalg.norm(nail_axis)
    if nail_axis_len > 0:
        nail_axis_dir = nail_axis / nail_axis_len
        target_len = s.location[0]
        scale_factor = target_len / nail_axis_len
        for i in range(len(nail_verts)):
            v = nail_verts[i]
            axial = np.dot(v - nail_verts[0], nail_axis_dir)
            radial = (v - nail_verts[0]) - axial * nail_axis_dir
            nail_verts[i] = nail_verts[0] + axial * scale_factor * nail_axis_dir + radial
        s.cylinder.vertices = o3d.utility.Vector3dVector(nail_verts)
        s.cylinder.compute_vertex_normals()

    # 7. 找手柄
    emit(72, "生成手柄...")
    handle_path = os.path.join(TEMPLATE_DIR, 'handle.stl')
    if os.path.exists(handle_path):
        s.find_handle(handle_path)
    else:
        s.cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=30)

    # 8. 找导板
    emit(75, "生成导板...")
    s.find_guide()
    emit(92, "导板生成完成")

    # 8.5 找基座
    emit(93, "生成基座...")
    jizuo_path = os.path.join(TEMPLATE_DIR, 'jizuo.stl')
    if os.path.exists(jizuo_path):
        s.find_jizuo(jizuo_path)
    else:
        s.jizuo = o3d.geometry.TriangleMesh.create_sphere(radius=5)

    # 9. 还原到世界坐标
    emit(95, "坐标还原...")
    s.go_back()

    # 10. 计算角度
    emit(96, "计算角度参数...")
    nail_center = np.asarray(s.cylinder.vertices)[0].tolist()
    nail_end = np.asarray(s.cylinder.vertices)[1].tolist()
    angle1, angle2 = find_angle(nail_center, nail_end, s.x, s.y, s.z)
    # 原始旋转参数（find_nail 搜索得到的实际旋转角），用于 regenerate 精确重建
    raw_angle1 = s.location[2][0]  # theta_y: 绕X轴旋转角
    raw_angle2 = s.location[2][1]  # theta_z: 绕Y轴旋转角
    length = s.location[0]

    # 11. 保存 STL
    emit(98, "保存 STL 文件...")
    s.save(saveDir)
    # 复制骨骼 mesh 到假体目录，供 regenerate 模式使用
    import shutil
    mesh_dst = os.path.join(saveDir, 'mesh.stl')
    if not os.path.exists(mesh_dst):
        shutil.copy2(modelPath, mesh_dst)

    # 12. 保存参数配置文件（regenerate 模式需要读取这些参数）
    cfg_path = os.path.join(saveDir, 'mesh.cfg')
    with open(cfg_path, 'w') as f:
        f.write(f'[parameter]\n')
        f.write(f'id1={s.id[0]}\n')
        f.write(f'id2={s.id[1]}\n')
        f.write(f'id3={s.id[2]}\n')
        f.write(f'p1x={s.p1[0]}\n')
        f.write(f'p1y={s.p1[1]}\n')
        f.write(f'p1z={s.p1[2]}\n')
        f.write(f'p2x={s.p2[0]}\n')
        f.write(f'p2y={s.p2[1]}\n')
        f.write(f'p2z={s.p2[2]}\n')
        f.write(f'p3x={s.p3[0]}\n')
        f.write(f'p3y={s.p3[1]}\n')
        f.write(f'p3z={s.p3[2]}\n')
        f.write(f'center1={s.center[0]}\n')
        f.write(f'center2={s.center[1]}\n')
        f.write(f'center3={s.center[2]}\n')
        f.write(f'r={s.r}\n')
        f.write(f'location1={length}\n')
        f.write(f'location2={angle1}\n')
        f.write(f'location3={angle2}\n')
        f.write(f'raw_angle1={raw_angle1}\n')
        f.write(f'raw_angle2={raw_angle2}\n')
        f.write(f'change-translate1={s.change[0][1][0]}\n')
        f.write(f'change-translate2={s.change[0][1][1]}\n')
        f.write(f'change-translate3={s.change[0][1][2]}\n')
        f.write(f'change-1rotate1={s.change[1][1][0]}\n')
        f.write(f'change-1rotate2={s.change[1][1][1]}\n')
        f.write(f'change-1rotate3={s.change[1][1][2]}\n')
        f.write(f'change-2rotate1={s.change[2][1][0]}\n')
        f.write(f'change-2rotate2={s.change[2][1][1]}\n')
        f.write(f'change-2rotate3={s.change[2][1][2]}\n')
        f.write(f'change-3rotate1={s.change[3][1][0]}\n')
        f.write(f'change-3rotate2={s.change[3][1][1]}\n')
        f.write(f'change-3rotate3={s.change[3][1][2]}\n')

    # 13. 输出参数（stdout 最后3行）
    emit(100, "生成完成")
    print(length)
    print(angle1)
    print(angle2)
