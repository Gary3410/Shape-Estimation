import pybullet_data as pd
import pybullet as p

import os
import numpy as np
import cv2
import torch
import argparse
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import shutil
import trimesh
import xml.etree.ElementTree as ET
from my_utils_one import get_point
import _pickle as cPickle
def change_scale_points(data):
    #centre
    xyz_min = np.min(data[:,0:3],axis=0)
    xyz_max = np.max(data[:,0:3],axis=0)
    xyz_move = xyz_min+(xyz_max-xyz_min)/2
    data[:,0:3] = data[:,0:3]-xyz_move
    #scale
    scale = np.max(data[:,0:3])
    data[:,0:3] = data[:,0:3]/scale
    return data
def change_urdf_scale(urdf_path, mesh):
    ex = random.uniform(0.5, 1.5)
    ez = random.uniform(0.5, 1.5)
    ey = random.uniform(0.5, 1.5)
    updateTree = ET.parse(urdf_path)
    root = updateTree.getroot()
    mesh_scale = [float(ex), float(ey), float(ez), float(0)]
    new_scale = str(ex) + ' ' + str(ey) + ' ' + str(ez)
    v_mesh_item = root.find('link').find('visual').find('geometry').find('mesh')
    v_mesh_item.set("scale", new_scale)
    c_mesh_item = root.find('link').find('collision').find('geometry').find('mesh')
    c_mesh_item.set("scale", new_scale)
    updateTree.write(urdf_path)

    ptsn = mesh.vertices
    ptsn[:, 0] = ptsn[:, 0] * ex
    ptsn[:, 1] = ptsn[:, 1] * ey
    ptsn[:, 2] = ptsn[:, 2] * ez


    max_point = np.max(ptsn, axis=0) * 1000
    min_point = np.min(ptsn, axis=0) * 1000

    scale = np.asarray(max_point - min_point).reshape([1, -1])
    return scale

def get_bbox(segImg, obj_one_id):
    y, x = np.where(segImg == obj_one_id)
    if y.shape[0] == 0 or x.shape[0] == 0:
        return np.asarray([0, 0, 0, 0])
    else:
        xmin = x.min()
        ymin = y.min()
        xmax = x.max()
        ymax = y.max()
        return np.asarray([xmin, ymin, xmax, ymax])

def move_2_C(pc):
    x_c=(max(pc[:,0])+min(pc[:,0]))/2
    y_c=(max(pc[:,1])+min(pc[:,1]))/2
    z_c=(max(pc[:,2])+min(pc[:,2]))/2
    pc_t=pc
    pc[:,0]=pc_t[:,0]-x_c
    pc[:,1]=pc_t[:,1]-y_c
    pc[:,2]=pc_t[:,2]-z_c
    return pc
def get_3D_corner_def(pc):
    pc=move_2_C(pc)
    x_r=max(pc[:,0])-min(pc[:,0])
    y_r=max(pc[:,1])-min(pc[:,1])
    z_r=max(pc[:,2])-min(pc[:,2])

    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or2=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or7=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or8=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])

    return OR, x_r,y_r,z_r,min(pc[:,1]),max(pc[:,1])

def save_img(img, base_path_0, time, view_id):
    color_img_print = np.ascontiguousarray(img)
    rgb_path = os.path.join(base_path_0, "rgb_img")
    img_path_2 = os.path.join(rgb_path, str(time) +"_" +str(view_id)+ '.png')
    cv2.imwrite(img_path_2, color_img_print)

def change_scale(mesh, collislion_mesh, object_name, obj_texture, obj_collislion):

    base_path = "/home/potato/workplace/obj_model/urdf_model_copy"
    img_file = os.path.join(base_path, object_name, "texture_map.jpg")
    if not os.path.exists(img_file):
        img_file = os.path.join(base_path, object_name, "texture_map.png")

    ptsn = mesh.vertices
    collision_ptsn = collislion_mesh.vertices
    collision_faces = collislion_mesh.faces
    faces = mesh.faces
    ex = random.uniform(0.8, 1.2)
    ez = random.uniform(0.8, 1.2)
    ey = random.uniform(0.8, 1.2)


    ptsn[:, 0] = ptsn[:, 0] * ex
    ptsn[:, 1] = ptsn[:, 1] * ey
    ptsn[:, 2] = ptsn[:, 2] * ez


    max_point = np.max(ptsn, axis=0) * 1000
    min_point = np.min(ptsn, axis=0) * 1000

    scale = np.asarray(max_point - min_point).reshape([1, -1])

    collision_ptsn[:, 0] = collision_ptsn[:, 0] * ex
    collision_ptsn[:, 1] = collision_ptsn[:, 1] * ey
    collision_ptsn[:, 2] = collision_ptsn[:, 2] * ez
    s = 0
    if "cup" in object_name or "bowl" in object_name:
        OR, lx, ly, lz, miny, maxy = get_3D_corner_def(ptsn)
        OR_1, lx_1, ly_1, lz_1, miny_1, maxy_1 = get_3D_corner_def(collision_ptsn)
        maxz = max(ptsn[:, 2])
        minz = min(ptsn[:, 2])
        maxx = max(ptsn[:, 0])
        minx = min(ptsn[:, 0])

        maxz_c = max(collision_ptsn[:, 2])
        s_c = s = 0.4
        s_pr = s
        s = s * ((maxz - ptsn[:, 2]) / lz)
        s_c = s_c * ((maxz_c - collision_ptsn[:, 2]) / lz_1)
        ptsn[:, 0] = ptsn[:, 0] * (1 + s)
        ptsn[:, 1] = ptsn[:, 1] * (1 + s)
        collision_ptsn[:, 0] = collision_ptsn[:, 0] * (1 + s_c)
        collision_ptsn[:, 1] = collision_ptsn[:, 1] * (1 + s_c)

    tex_img = Image.open(img_file)
    material = trimesh.visual.texture.SimpleMaterial(image=tex_img)
    color_visuals = trimesh.visual.TextureVisuals(image=tex_img, material=material)
    new_mesh = trimesh.Trimesh(vertices=ptsn, faces=faces, visual=color_visuals, process=True)
    new_mesh_file = os.path.join(base_path, object_name, obj_texture + ".obj")
    new_mesh.export(new_mesh_file)

    new_collision = trimesh.Trimesh(vertices=collision_ptsn, faces=collision_faces, process=True)
    new_collision_file = os.path.join(base_path, object_name, obj_collislion + ".obj")
    new_collision.export(new_collision_file)
    #scale = [float(ex), float(ey), float(ez), float(s_pr)]
    return new_mesh, new_collision, scale

def create_box_bullet(scale, pos):
    l = scale[0]
    w = scale[1]
    h = scale[2]

    x = pos[0]
    y = pos[1]
    z = pos[2]
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[l / 2, w / 2, h / 2],
        rgbaColor=[128, 128, 128, 1])

    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[l / 2, w / 2, h / 2]
    )

    wall_id = p.createMultiBody(
        baseMass=10000,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[x, y, z]
    )
    return wall_id

def create_wall():
    xmin = -0.85
    ymin = -0.2
    xmax = -0.25
    ymax = 0.2
    TopHeight = 0.5

    wall_id_1 = create_box_bullet([xmax - xmin, 0.1, TopHeight], [(xmax - xmin) / 2 + xmin, ymin - 0.05, TopHeight / 2])
    wall_id_2 = create_box_bullet([xmax - xmin, 0.1, TopHeight], [(xmax - xmin) / 2 + xmin, ymax + 0.05, TopHeight / 2])
    wall_id_3 = create_box_bullet([0.1, ymax - ymin, TopHeight], [xmin - 0.05, (ymax - ymin) / 2 + ymin, TopHeight / 2])
    wall_id_4 = create_box_bullet([0.1, ymax - ymin, TopHeight], [xmax + 0.05, (ymax - ymin) / 2 + ymin, TopHeight / 2])

    return wall_id_1, wall_id_2, wall_id_3, wall_id_4
def save_points(pts, area_id, office_id, view_id, number_one, object_mesh_list):
    base_path_1 = "/home/potato/workplace/dataset/single_view_dataset"
    num_list = np.ones(len(object_mesh_list))
    area_path = os.path.join(base_path_1, "Area_" + str(area_id + 1))
    office_path = os.path.join(area_path, "office_" + str(office_id + 1) + "_" + str(view_id))
    if os.path.exists(office_path):
        shutil.rmtree(office_path)
    os.mkdir(office_path)
    office_txt_path = os.path.join(office_path, "office_" + str(office_id + 1) + "_" + str(view_id) + ".txt")
    anno_path = os.path.join(office_path, "Annotations")
    if os.path.exists(anno_path):
        shutil.rmtree(anno_path)
    os.mkdir(anno_path)
    pts_all = []
    for i in range(1, number_one + 1):
        pts_one = pts[pts[:, 7] == i][:, :7].copy()
        if pts_one.shape[0] <= 100:
            continue
        cls = int(pts_one[0, 6].item())
        pts_one = sample_data(pts_one, 5000)[0]
        pts_one = pts_one.reshape([-1, 7])
        pts_all.append(pts_one)
        label = object_mesh_list[cls - 1]
        save_label(pts_one[:, :6], label, num_list[cls - 1], anno_path)
        num_list[cls - 1] = num_list[cls - 1] + 1
    pts_all = np.concatenate(pts_all, axis=0)
    np.savetxt(office_txt_path, pts_all[:, :6], fmt='%.6e', delimiter=" ")

def save_label(pts, label, num, anno_path):
    label_name = label
    pts_path = os.path.join(anno_path, label_name + "_" + str(int(num))+".txt")
    np.savetxt(pts_path, pts, fmt='%.6e', delimiter=" ")

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        #points = torch.from_numpy(data).cuda().unsqueeze(0)
        #points = resample(points.float(), 5000)
        points = torch.from_numpy(data[:, :3]).cuda().unsqueeze(0)
        device = points.device
        B, N, C = points.shape

        centroids = torch.zeros(B, num_sample, dtype=torch.long).to(device)

        distance = torch.ones(B, N).to(device) * 1e10

        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(num_sample):

            centroids[:, i] = farthest

            centroid = points[batch_indices, farthest, :].view(B, 1, 3)

            dist = torch.sum((points - centroid) ** 2, -1).float()

            mask = dist < distance
            distance[mask] = dist[mask]

            farthest = torch.max(distance, -1)[1]
        centroids = centroids.cpu().numpy().reshape([num_sample, ])
        point = data[centroids.astype(np.int32)]
        return point, sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def main():
    p.connect(p.DIRECT)  # p.DIRECT
    number_list = [8, 10, 12, 14, 16, 18, 20]
    viewMat_list = [
        [1.0, 0.0, -0.0, 0.0, -0.0, 1.0, -0.00017452123574912548, 0.0, 0.0, 0.00017452123574912548, 1.0, 0.0, 0.5,
         -7.417152664856985e-05, -0.4266100227832794, 1.0],
        [1.0, 0.0, -0.0, 0.0, -0.0, 0.8660240173339844, -0.5000023245811462, 0.0, 0.0, 0.5000023245811462,
         0.8660240173339844, 0.0, 0.5, 2.925097942352295e-05, -0.426557332277298, 1.0],
        [0.017451846972107887, 0.8658873438835144, -0.49993473291397095, 0.0, -0.9998477697372437, 0.015113634057343006,
         -0.00872611254453659, 0.0, 0.0, 0.5000109076499939, 0.8660191893577576, 0.0, 0.012434440664947033,
         0.4329407513141632, -0.6765085458755493, 1.0],
        [-0.999847412109375, 0.015133202075958252, -0.008737090043723583, 0.0, -0.017474284395575523,
         -0.8658948540687561, 0.49992072582244873, 0.0, 0.0, 0.4999971091747284, 0.8660269975662231, 0.0,
         -0.4962104260921478, 0.007570326328277588, -0.4309096336364746, 1.0],
        [-0.01745261251926422, -0.8658953309059143, 0.4999208450317383, 0.0, 0.9998477697372437, -0.015114436857402325,
         0.008726253174245358, 0.0, 0.0, 0.49999701976776123, 0.8660271763801575, 0.0, -0.005017626099288464,
         -0.43294382095336914, -0.1765807718038559, 1.0]]

    projMat = [0.74999994, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.00002003, -1.0, 0.0, 0.0, -0.0200002, 0.0]
    heightmap_resolution = 0.002
    name_list = []
    texture_list = []
    collislion_list = []
    f = open("/home/potato/workplace/obj_model/name_list.txt")
    for line in f:
        name_list.append(line.strip())
    f.close()
    f = open("/home/potato/workplace/obj_model/texture_list.txt")
    for line in f:
        texture_list.append(line.strip())
    f.close()
    f = open("/home/potato/workplace/obj_model/collislion_list.txt")
    for line in f:
        collislion_list.append(line.strip())
    f.close()
    print(len(texture_list))
    print(len(collislion_list))
    base_path = "/home/potato/workplace/obj_model/urdf_model"
    base_path_copy = "/home/potato/workplace/obj_model/urdf_model_copy"
    base_path_0 = "/home/potato/workplace/dataset/data"
    label_list = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
    mean_scale_list = np.loadtxt("/home/potato/workplace/obj_model/mean_scale_cls.txt")

    workspace_limits = np.asarray([[-0.7, -0.50], [-0.12, 0.12], [0.1, 0.5]])


    num_of_obj = 0
    obj_name_id = 0

    for _ in tqdm(range(2400)):
        if num_of_obj >= 300:
            num_of_obj = 0
            obj_name_id = obj_name_id + 1
            if obj_name_id == len(label_list):
                exit()
        obj_class = label_list[obj_name_id]

        obj_name_list = [f for f in name_list if obj_class in f]
        print(obj_name_list)
        p.setGravity(0.000000, 0.000000, -9.800000)
        planeId = p.loadURDF(os.path.join(pd.getDataPath(), "plane.urdf"))
        wall_id_1, wall_id_2, wall_id_3, wall_id_4 = create_wall()
        number_one = random.choice(number_list)

        object_name_list = obj_name_list
        flags = p.URDF_USE_INERTIA_FROM_FILE

        obj_id_list = []
        object_mesh_id = []
        scale_list = []

        for object_idx in range(number_one):
            object_name = np.random.choice(object_name_list)
            obj_index = name_list.index(object_name)

            obj_name = '_'.join(object_name.split('_')[0:-1])
            object_mesh_id.append(label_list.index(obj_name) + 1)

            obj_texture = texture_list[obj_index]
            obj_collislion = collislion_list[obj_index]

            obj_file = os.path.join(base_path, object_name, obj_texture + ".obj")
            mesh = trimesh.load_mesh(obj_file)
            collislion_file = os.path.join(base_path, object_name, obj_collislion + ".obj")
            collislion_mesh = trimesh.load_mesh(collislion_file)
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0]) * np.random.random_sample() + \
                     workspace_limits[0][0]
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0]) * np.random.random_sample() + \
                     workspace_limits[1][0]
            drop_z = (workspace_limits[2][1] - workspace_limits[2][0]) * np.random.random_sample() + \
                     workspace_limits[2][0]

            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]

            if "cup" in object_name or "bowl" in object_name:
                new_mesh, collislion_mesh, scale = change_scale(mesh, collislion_mesh, object_name, obj_texture, obj_collislion)
                obj_file = os.path.join(base_path_copy, object_name)
                urdf_file = [f for f in os.listdir(obj_file) if f.endswith('.urdf')]

                obj_id = p.loadURDF(os.path.join(obj_file, urdf_file[0]), basePosition=[drop_x, drop_y, 0.22],
                    baseOrientation=[drop_x, drop_y, 0.22, object_orientation[0]], flags=flags)
                scale_list.append(np.asarray(scale, dtype=float).reshape([1, -1]))
            else:
                obj_file = os.path.join(base_path_copy, object_name)
                urdf_file = [f for f in os.listdir(obj_file) if f.endswith('.urdf')]
                mesh_scale = change_urdf_scale(os.path.join(obj_file, urdf_file[0]), mesh)
                obj_id = p.loadURDF(os.path.join(obj_file, urdf_file[0]), basePosition=[drop_x, drop_y, 0.22],
                    baseOrientation=[drop_x, drop_y, 0.22, object_orientation[0]], flags=flags)
                scale_list.append(np.asarray(mesh_scale, dtype=float).reshape([1, -1]))

            for _ in range(480):
                p.stepSimulation()
            obj_id_list.append(obj_id)

        scale_list = np.concatenate(scale_list, axis=0)

        p.removeBody(wall_id_1)
        p.removeBody(wall_id_2)
        p.removeBody(wall_id_3)
        p.removeBody(wall_id_4)

        for _ in range(200):
            p.stepSimulation()

        object_mesh_id.insert(0, 0)

        pts_list = []

        for view_id in tqdm(range(5)):
            viewMat = viewMat_list[view_id]
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=1440, height=1024, viewMatrix=viewMat,
                projectionMatrix=projMat, flags=flags)
            # 保存图像

            color_img = np.asarray(rgbImg)
            color_img = color_img[:, :, :3]
            color_img = color_img.astype(np.uint8)

            # color_img_print用来存储RGB图片
            #color_img_print = color_img.copy()
            if view_id == 0:
                color_img_print = color_img[:, :, ::-1]  # RGB-->BGR

            class_id = np.array(object_mesh_id).reshape(-1, 1)

            segImg = segImg - 4
            segImg[segImg <= 0] = 0
            segImg_cls = segImg.copy()
            segImg_copy = segImg.copy()

            if view_id == 0:
                depthImg_print = np.asarray(depthImg).copy()
            if view_id == 0:
                ints_image = Image.fromarray(np.uint8(segImg_copy))
                segImg_copy_print = segImg_copy.copy()
            #seg_label = class_id[segImg, :].reshape([segImg.shape[0], segImg.shape[1]])
            segImg_cls = class_id[segImg_cls, :].reshape([-1, 1])
            segImg = segImg.reshape([-1, 1])
            position = get_point(depthImg, viewMat, projMat)
            color_img = color_img.reshape([-1, 3])
            position = np.concatenate((position[:, :3], color_img), axis=1)
            position = np.concatenate((position, segImg_cls, segImg), axis=1)


            position = position[position[:, 6] > 0]
            # save_points(position, area_id, office_id, view_id, number_one, label_list)
            pts_list.append(position)


        data = np.concatenate(pts_list, axis=0)
        base_path_1 = "/home/potato/workplace/dataset/data/points"
        pts_all = []
        view_mat = viewMat_list[0]
        proj_mat = projMat
        for i in tqdm(range(1, number_one + 1)):
            pts_one = data[data[:, 7] == i][:, :7]
            if pts_one.shape[0] <= 100:
                continue
            cls = int(pts_one[0, 6].item())
            pts_one = change_scale_points(pts_one)
            pts_one = sample_data(pts_one, 4096)[0]
            pts_one = pts_one.reshape([-1, 7])


            pts_obj_id = i + 4
            img_base_path = os.path.join(base_path_0, obj_class, "1")
            img_path = img_base_path + "/" + str(num_of_obj)
            cv2.imwrite(img_path + '_rgb.png', color_img_print)
            dep_ortho = np.asarray(depthImg_print)
            dep_ortho = 5 * 0.1 / (5 - (5 - 0.1) * dep_ortho)  # 深度图数值转换为z轴数值
            cv2.imwrite(img_path + '_depth.png', np.asarray(dep_ortho * 1000, np.uint16))


            gts = {}
            gts['class_ids'] = np.asarray([cls])  # int list, 1 to 6

            gts['bboxes'] = get_bbox(np.asarray(segImg_copy_print), pts_obj_id - 4)  # np.array, [[x1, y1, x2, y2], ...]    # 根据分割图的mask找bounding box

            ints_image.save(img_path + '_seg.png')

            single_scale_obj = mean_scale_list[cls-1] - scale_list[i-1]
            gts['scales'] = np.asarray(single_scale_obj)
            obj_t, obj_r = p.getBasePositionAndOrientation(pts_obj_id)
            cam_pose = np.asarray(view_mat).reshape((4, 4), order='F') * np.asarray(
                (1, 1, 1, 1,  # 相机位姿
                 -1, -1, -1, -1,
                 -1, -1, -1, -1,
                 1, 1, 1, 1,)).reshape((4, 4))

            obj_r = np.asarray(p.getMatrixFromQuaternion(obj_r)).reshape((3, 3))
            obj_t = np.asarray(obj_t).reshape((3, 1))
            obj_pose = np.vstack((np.hstack((obj_r, obj_t)), [0, 0, 0, 1]))  # 物体在世界坐标系下的位姿矩阵
            obj_cam = cam_pose @ obj_pose  # 物体在相机坐标系下位姿

            gts['rotations'] = obj_cam[:3, :3].astype(np.float32)  # np.array, R
            gts['translations'] = obj_cam[:3, 3].astype(np.float32)  # np.array, T

            with open(img_path + '_label.pkl', 'wb') as f:  # gts写入label.pkl
                cPickle.dump(gts, f)

            pts_save_path = os.path.join(base_path_0, obj_class, "points", "pose"+'%08d.txt'%(num_of_obj))
            pts_label_save_path = os.path.join(base_path_0, obj_class, "points_labs", "lab"+'%08d.txt'%(num_of_obj))
            pts_one_save = pts_one[:, :3] * 1000
            np.savetxt(pts_save_path, pts_one_save, fmt='%.3f', delimiter=" ")
            pts_one_label = np.ones_like(pts_one_save)
            np.savetxt(pts_label_save_path, pts_one_label[:,0:1], fmt='%d', delimiter=" ")
            num_of_obj = num_of_obj + 1
        p.resetSimulation()

if __name__ == '__main__':
    # Parse arguments
    main()






















