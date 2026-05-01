import open3d as o3d
import numpy as np
import copy
import math
from tqdm import tqdm
import pyvista as pv
import os
import nibabel
import sys
import json

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
            triangles.append(self.mesh_pv.cell_point_ids(i))
        triangles = np.array(triangles)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(self.mesh_pv.points))
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        self.guide_mesh = 0

        self.pcd = o3d.geometry.PointCloud()
        self.V_mesh = np.array(self.mesh.vertices)
        self.pcd.points = o3d.utility.Vector3dVector(self.V_mesh)

        self.change = []

    def select_points2(self, picked_id_pcd):
        self.tp1 = np.array(picked_id_pcd[0])# 第一个点坐标
        self.tp2 = np.array(picked_id_pcd[1])# 第二个点坐标
        self.tp3 = np.array(picked_id_pcd[2])# 第三个点坐标
        self.d1 = np.linalg.norm(self.V_mesh - self.tp1,axis=1)
        self.cpi1 = np.argmin(self.d1)# 第一个点索引
        self.d2 = np.linalg.norm(self.V_mesh - self.tp2,axis=1)
        self.cpi2 = np.argmin(self.d2)# 第二个点索引
        self.d3 = np.linalg.norm(self.V_mesh - self.tp3,axis=1)
        self.cpi3 = np.argmin(self.d3)# 第三个点索引
        ids = np.array([self.cpi1, self.cpi2, self.cpi3])
        a = self.pcd.points
        self.p1 = a[ids[0]]
        self.p2 = a[ids[1]]
        self.p3 = a[ids[2]]
        self.id = ids

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
            triangles.append(self.cylinder101pv.cell_point_ids(i))
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

        R = self.cylinder10.get_rotation_matrix_from_xyz((location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, location[2][1] * np.pi / 180, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder = copy.deepcopy(mesh_cylinderchange)

        R = self.cylinder101.get_rotation_matrix_from_xyz((location[2][0] * np.pi / 180, 0, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder101)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, location[2][1] * np.pi / 180, 0))
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
           
            i = z[0]
            j = z[1]
            q = [xyz[i, 0], xyz[i, 1], xyz[i, 2] - j * 0.5]
            p.append(q)
     
        p = np.array(p)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(p)
        self.guide_pcd = pcd2
     
        mesh4 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd2, alpha=2)
      
        mesh4.compute_vertex_normals()
  
        mesh4.paint_uniform_color([0, 0.8, 0.8])

        self.guide_mesh = mesh4
        self.guide_mesh.paint_uniform_color([0.1, 0.1, 0.7])

       

    def show(self, l):
        pl = pv.Plotter()
        for i in range(len(l)):
            o3d.io.write_triangle_mesh('%d.stl' % i, l[i])
            p = pv.read('%d.stl' % i)
            _ = pl.add_mesh(p)

        pl.camera_position = 'xz'
        pl.show()

    def save(self, path):
        o3d.io.write_triangle_mesh(path + '/nail1.stl', self.cylinder)
        # o3d.io.write_triangle_mesh('nail.stl', self.cylinder101)
        o3d.io.write_triangle_mesh(path + '/guide.stl', self.guide_mesh)
        o3d.io.write_triangle_mesh(path + '/handle.stl', self.cylinder2)
        # o3d.io.write_triangle_mesh('mesh.stl', self.mesh)
        o3d.io.write_triangle_mesh(path + '/jizuo.stl', self.jizuo)

    def save1(self, path):
        # o3d.io.write_triangle_mesh(path + '/nail1.stl', self.cylinder)
        # o3d.io.write_triangle_mesh('nail.stl', self.cylinder101)
        o3d.io.write_triangle_mesh(path + '/guide.stl', self.guide_mesh)
        o3d.io.write_triangle_mesh(path + '/handle.stl', self.cylinder2)
        # o3d.io.write_triangle_mesh('mesh.stl', self.mesh)
        o3d.io.write_triangle_mesh(path + '/jizuo.stl', self.jizuo)

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
            l3 = dis(p1, p3);
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
        triangles.append(mesh0.cell_point_ids(i))
    triangles = np.array(triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh0.points))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    n_cell = nail0.n_cells
    triangles = []
    for i in range(n_cell):
        triangles.append(nail0.cell_point_ids(i))
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
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
PathDicom = "./zidongnail"
NAIL_PATH = PathDicom + '/6.5mm/nail.stl'
HANDLE_PATH = PathDicom + '/10.stl'
JIZUO_PATH = PathDicom + '/6.5mm/jizuo.stl'
PathSave = PathDicom+'/savestl'
getNail = scapula(PathDicom + '/mesh(4).stl')
json_path = sys.argv[1]
fpath = json.loads(json_path)
getNail.select_points2(fpath)

getNail.computer_circle()

getNail.move_center_to_O()

getNail.find_vector(NAIL_PATH, 6.5)


getNail.find_nail(5 / 5, 20 / 20, 200)
a1 = str(getNail.location[0])
print(a1)
a2 = str(getNail.location[2][0])
print(a2)
a3 = str(getNail.location[2][1])
print(a3)


getNail.find_handle(HANDLE_PATH)

getNail.find_guide()

getNail.find_jizuo(JIZUO_PATH)
getNail.go_back()
getNail.save(PathSave)