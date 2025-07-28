import os
import trimesh
from pysdf import SDF

# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/lidar/haonan/IMavatar_haonan_rd_pca_100_100000_depth1_continue_optimizecontact/rgb/eval/rgb/epoch_120'
# # mesh2_name = 'surface_1041.ply'
# # mesh2_name = 'surface_1359.ply'
# # mesh2_name = 'surface_1357.ply'
# # mesh2_name = 'surface_1355.ply'
# mesh2_name = 'surface_1043.ply'

# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/lidar/linyi/IMavatar_linyi_rd_pca_100_100000_depth1_continue_optimizecontact/rgb/eval/rgb/epoch_120'
# # mesh2_name = 'surface_173.ply'
# # mesh2_name = 'surface_789.ply'
# mesh2_name = 'surface_793.ply'
# # mesh2_name = 'surface_177.ply'
# # mesh2_name = 'surface_733.ply'

# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/lidar/luocheng/IMavatar_luocheng_rd_pca_100_100000_depth1_continue_optimizecontact/rgb/eval/rgb/epoch_120'
# # # mesh2_name = 'surface_1050.ply'
# mesh2_name = 'surface_1560.ply'


# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/lidar/zirui/IMavatar_zirui_rd_pca_100_100000_depth4_continue_optimizecontact/rgb/eval/rgb/epoch_120'
# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/lidar/zirui/IMavatar_zirui_rd_pca_100_100000_depth4_continue_optimizecontact_v2/rgb/eval/rgb/epoch_120'
# mesh2_name = 'surface_767.ply'
# mesh2_name = 'surface_1103.ply'

# mesh_dir = '/home/haonan/data/IMAvatar_hand_head/data/experiments/rebuttal/haonan_2hands/IMavatar_rebuttal_haonan2hands_rd_pca_10_50000_depth1_continue_single_continue/rgb/eval/rgb/epoch_115'
# mesh2_name = 'surface_1271.ply'


mesh2_pth = os.path.join(mesh_dir, mesh2_name)
mesh1_name = mesh2_name.replace('.ply', '_left_hand.ply')
mesh1_pth = os.path.join(mesh_dir, mesh1_name)

# 加载两个网格
mesh1 = trimesh.load(mesh1_pth)
mesh2 = trimesh.load(mesh2_pth)

# 计算交集
# intersection = mesh1.intersection(mesh2)
f = SDF(mesh2.vertices, mesh2.faces)
insides = f.contains(mesh1.vertices)
contact_idx = f.nn(mesh1.vertices[insides])

# 给交集网格上色为红色
mesh2.visual.vertex_colors = [180, 180, 180, 255]
mesh2.visual.vertex_colors[contact_idx] = [255, 0, 0, 255]

mesh1_name = mesh2_name.replace('.ply', '_right_hand.ply')
mesh1_pth = os.path.join(mesh_dir, mesh1_name)

# 加载两个网格
mesh1 = trimesh.load(mesh1_pth)

# 计算交集
# intersection = mesh1.intersection(mesh2)
f = SDF(mesh2.vertices, mesh2.faces)
insides = f.contains(mesh1.vertices)
contact_idx = f.nn(mesh1.vertices[insides])

# 给交集网格上色为红色
mesh2.visual.vertex_colors[contact_idx] = [255, 0, 0, 255]

# 保存结果用于 MeshLab 查看
mesh2.export(os.path.join(mesh_dir, mesh2_name.replace('.ply', '_color.ply')))