import torch
import torch.nn.functional as F


def tbn(triangles):
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a), dim=-1)
    d = b - a

    X = F.normalize(torch.cross(d, n), dim=-1)
    Y = F.normalize(torch.cross(d, X), dim=-1)
    Z = F.normalize(d, dim=-1)

    return torch.stack([X, Y, Z], dim=3)


def triangle2projection(triangles):
    R = tbn(triangles)
    T = triangles.unbind(-2)[0]
    I = torch.repeat_interleave(torch.eye(4, device=triangles.device)[None, None, ...], R.shape[1], 1)

    I = I.repeat(R.shape[0], 1, 1, 1)

    I[:, :, 0:3, 0:3] = R
    I[:, :, 0:3, 3] = T

    return I


def calculate_centroid(tris, dim=2):
    c = tris.sum(dim) / 3
    return c


# def interpolate(xyzs, tris, neighbours, edge_mask):
#     # Currently disabled
#     # return triangle2projection(tris)[0]
#     N = xyzs.shape[0]
#     # factor = 4
#     factor = 0.0
#     c_closests = calculate_centroid(tris)
#     c_neighbours = calculate_centroid(neighbours)
#     dc = torch.exp(-torch.norm(xyzs - c_closests[0], dim=1, keepdim=True))
#     # dn = torch.exp(-factor * torch.norm(xyzs.repeat(1, 3).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask
#     dn = torch.exp(-torch.norm(xyzs.repeat(1, 3).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask.permute(1, 0)
#     dn = dn.reshape(N, 3, 1).squeeze(-1)
#     dn = dn * factor
#     # breakpoint()
#     # dn = factor * torch.exp(-torch.norm(xyzs[:,None,:].repeat(1, 3, 1) - c_neighbours[0].reshape(N, 3, 3), dim=-1, keepdim=False)) 
#     # edge_mask = edge_mask.squeeze(0).reshape(N, 3)
#     # dn = dn * edge_mask
#     distances = torch.cat([dc, dn], dim=1)
#     try:
#         triangles = torch.cat([triangle2projection(tris)[0][:, None, ...], triangle2projection(neighbours)[0][:, None, ...].reshape(N, -1, 4, 4)], dim=1)
#     except:
#         breakpoint()
#     normalization = distances.sum(-1, keepdim=True) + 1e-8
#     weights = distances / normalization

#     return (triangles * weights[..., None, None]).sum(1)

# def interpolate(xyzs, tris, neighbours, edge_mask):
#     # Currently disabled
#     # return triangle2projection(tris)[0]
#     N = xyzs.shape[0]
#     # factor = 4
#     factor = 0.0
#     c_closests = calculate_centroid(tris)
#     c_neighbours = calculate_centroid(neighbours)
#     dc = torch.exp(-torch.norm(xyzs - c_closests[0], dim=1, keepdim=True))

#     # dn = torch.exp(-torch.norm(xyzs.repeat(1, 3).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask.permute(1, 0)
#     # dn = dn.reshape(N, 3, 1).squeeze(-1)
#     # dn = dn * factor
#     # distances = torch.cat([dc, dn], dim=1)
#     distances = dc
#     try:
#         # triangles = torch.cat([triangle2projection(tris)[0][:, None, ...], triangle2projection(neighbours)[0][:, None, ...].reshape(N, -1, 4, 4)], dim=1)
#         triangles = triangle2projection(tris)[0][:, None, ...]
#     except:
#         breakpoint()
#     normalization = distances.sum(-1, keepdim=True) + 1e-8
#     weights = distances / normalization

#     return (triangles * weights[..., None, None]).sum(1)

# def interpolate(xyzs, tris, neighbours, edge_mask):
#     # Currently disabled
#     # return triangle2projection(tris)[0]
#     '''
#         xyzs: (N, 3)
#         tris: (1, N, 3, 3)
#         neighbours: (1, N*3, 3, 3)
#         edge_mask: (N*3, 1)
#     '''
#     N = xyzs.shape[0]
#     factor = 4
#     # factor = 1
#     # factor = 0.0
#     c_closests = calculate_centroid(tris)
#     c_neighbours = calculate_centroid(neighbours)
#     dc = torch.exp(- torch.norm(xyzs - c_closests[0], dim=1, keepdim=True))

#     dn = torch.exp(- factor * torch.norm(xyzs[:,None,:].repeat(1, 3, 1).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask#.permute(1, 0)
#     dn = dn.reshape(N, 3, 1).squeeze(-1)
#     # dn = dn * factor
#     distances = torch.cat([dc, dn], dim=1)

#     triangles = torch.cat([triangle2projection(tris)[0][:, None, ...], triangle2projection(neighbours)[0][:, None, ...].reshape(N, -1, 4, 4)], dim=1)

#     normalization = distances.sum(-1, keepdim=True) + 1e-8
#     weights = distances / normalization

#     return (triangles * weights[..., None, None]).sum(1)

def interpolate(xyzs, tris, neighbours, edge_mask):
    # Compute RT matrices for current triangles
    Rt_current = triangle2projection(tris)  # Shape: (1, N, 4, 4)
    N = Rt_current.shape[1]
    Rt_current = Rt_current.squeeze(0)  # Shape: (N, 4, 4)

    # Compute RT matrices for neighboring triangles
    Rt_neighbours = triangle2projection(neighbours)  # Shape: (1, N*3, 4, 4)
    Rt_neighbours = Rt_neighbours.squeeze(0).view(N, 3, 4, 4)  # Shape: (N, 3, 4, 4)

    # Reshape edge_mask to (N, 3, 1)
    edge_mask = edge_mask.view(N, 3, 1)

    # Calculate centroids for current and neighboring triangles
    c_current = calculate_centroid(tris, dim=2).squeeze(0)  # Shape: (N, 3)
    neighbours_reshaped = neighbours.view(N, 3, 3, 3)
    c_neighbours = calculate_centroid(neighbours_reshaped, dim=2)  # Shape: (N, 3, 3)

    # Compute distances from each point to centroids
    dist_current = torch.norm(xyzs - c_current, dim=-1)  # Shape: (N,)
    dist_neighbours = torch.norm(xyzs.unsqueeze(1) - c_neighbours, dim=-1)  # Shape: (N, 3)

    # Mask out invalid neighbors by setting their distance to a large value
    large_value = 1e10
    dist_neighbours = torch.where(edge_mask.squeeze(-1).bool(), dist_neighbours, torch.tensor(large_value, device=xyzs.device))

    # Combine distances and compute weights
    all_distances = torch.cat([dist_current.unsqueeze(1), dist_neighbours], dim=1)  # Shape: (N, 4)
    eps = 1e-8
    inv_distances = 1.0 / (all_distances + eps)
    weights = inv_distances / inv_distances.sum(dim=1, keepdim=True)  # Shape: (N, 4)

    # Combine current and neighbor RT matrices
    all_Rts = torch.cat([Rt_current.unsqueeze(1), Rt_neighbours], dim=1)  # Shape: (N, 4, 4, 4)

    # Decompose into rotation and translation components
    R = all_Rts[:, :, :3, :3]  # Shape: (N, 4, 3, 3)
    T = all_Rts[:, :, :3, 3]   # Shape: (N, 4, 3)

    # # Apply weights to R and T
    # weighted_R = R * weights.unsqueeze(-1).unsqueeze(-1)  # Broadcasting weights
    # sum_R = weighted_R.sum(dim=1)  # Shape: (N, 3, 3)

    weighted_T = T * weights.unsqueeze(-1)
    sum_T = weighted_T.sum(dim=1)  # Shape: (N, 3)

    # # Orthogonalize the summed rotation matrices using SVD
    # U, S, V = torch.svd(sum_R)
    # orthogonal_R = U @ V.transpose(-1, -2)
    
    sum_R = interpolate_rotations(R, weights)

    # Construct the interpolated RT matrix
    interpolated_Rt = torch.eye(4, device=xyzs.device).unsqueeze(0).repeat(N, 1, 1)
    # interpolated_Rt[:, :3, :3] = orthogonal_R
    interpolated_Rt[:, :3, :3] = sum_R
    interpolated_Rt[:, :3, 3] = sum_T

    return interpolated_Rt  # Shape: (N, 4, 4)

# Compute signed distance to triangle planes (more stable than centroid distances)
def point_to_plane_distance(xyz, tri):
    a, b, c = tri.unbind(-2)
    normal = F.normalize(torch.cross(b - a, c - a), dim=-1)
    # return torch.abs((xyz - a) * normal)  # Absolute distance to plane
    return torch.abs((xyz - a) * normal)  # Absolute distance to plane

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
def interpolate_rotations(R_matrices, weights):
    # Convert rotation matrices to quaternions
    quats = matrix_to_quaternion(R_matrices)
    
    # Normalize weights for slerp
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Use torch.lerp for small angles or implement slerp
    blended_quat = quats * weights.unsqueeze(-1)
    # blended_quat = F.normalize(blended_quat, dim=-1)
    blended_quat = blended_quat.sum(1)
    
    return quaternion_to_matrix(blended_quat)

# def interpolate(xyzs, tris, neighbours, edge_mask):
#     # Compute RT matrices for current triangles
#     Rt_current = triangle2projection(tris)  # Shape: (1, N, 4, 4)
#     N = Rt_current.shape[1]
#     Rt_current = Rt_current.squeeze(0)  # Shape: (N, 4, 4)

#     # Compute RT matrices for neighboring triangles
#     Rt_neighbours = triangle2projection(neighbours)  # Shape: (1, N*3, 4, 4)
#     Rt_neighbours = Rt_neighbours.squeeze(0).view(N, 3, 4, 4)  # Shape: (N, 3, 4, 4)

#     # Reshape edge_mask to (N, 3, 1)
#     edge_mask = edge_mask.view(N, 3, 1)

#     # Calculate centroids for current and neighboring triangles
#     c_current = calculate_centroid(tris, dim=2).squeeze(0)  # Shape: (N, 3)
#     neighbours_reshaped = neighbours.view(N, 3, 3, 3)
#     c_neighbours = calculate_centroid(neighbours_reshaped, dim=2)  # Shape: (N, 3, 3)

#     # Compute distances from each point to centroids
#     # dist_current = torch.norm(xyzs - c_current, dim=-1)  # Shape: (N,)
#     # dist_neighbours = torch.norm(xyzs.unsqueeze(1) - c_neighbours, dim=-1)  # Shape: (N, 3)
#     dist_current = point_to_plane_distance(xyzs, tris.squeeze(0))
#     dist_neighbours = point_to_plane_distance(xyzs.unsqueeze(1).repeat(1,3,1).reshape(-1,3), neighbours.view(-1, 3, 3))

#     # Mask out invalid neighbors by setting their distance to a large value
#     large_value = 1e10
#     breakpoint()
#     dist_neighbours = torch.where(edge_mask.repeat(1,1,3).permute(0,2,1).reshape(-1, 3).bool(), dist_neighbours, torch.tensor(large_value, device=xyzs.device))

#     # Combine distances and compute weights
#     all_distances = torch.cat([dist_current, dist_neighbours], dim=1)  # Shape: (N, 4)
#     eps = 1e-8
#     inv_distances = 1.0 / (all_distances + eps)
#     weights = inv_distances / inv_distances.sum(dim=1, keepdim=True)  # Shape: (N, 4)

#     # Combine current and neighbor RT matrices
#     all_Rts = torch.cat([Rt_current.unsqueeze(1), Rt_neighbours], dim=1)  # Shape: (N, 4, 4, 4)

#     # Decompose into rotation and translation components
#     R = all_Rts[:, :, :3, :3]  # Shape: (N, 4, 3, 3)
#     T = all_Rts[:, :, :3, 3]   # Shape: (N, 4, 3)

#     # # Apply weights to R and T
#     # weighted_R = R * weights.unsqueeze(-1).unsqueeze(-1)  # Broadcasting weights
#     # sum_R = weighted_R.sum(dim=1)  # Shape: (N, 3, 3)

#     weighted_T = T * weights.unsqueeze(-1)
#     sum_T = weighted_T.sum(dim=1)  # Shape: (N, 3)
    
#     sum_R = interpolate_rotations(R, weights)

#     # # Orthogonalize the summed rotation matrices using SVD
#     # U, S, V = torch.svd(sum_R)
#     # orthogonal_R = U @ V.transpose(-1, -2)

#     # Construct the interpolated RT matrix
#     interpolated_Rt = torch.eye(4, device=xyzs.device).unsqueeze(0).repeat(N, 1, 1)
#     # interpolated_Rt[:, :3, :3] = orthogonal_R
#     interpolated_Rt[:, :3, :3] = sum_R
#     interpolated_Rt[:, :3, 3] = sum_T

#     return interpolated_Rt  # Shape: (N, 4, 4)


def project_position_batch(xyzs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(xyzs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)
    Rt_canon = interpolate(xyzs, canonical_triangles, canonical_neighbours, edge_mask)

    # homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, 0:1]], dim=1)[:, :, None]
    homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, :, 0:1]], dim=-1)[:, :, :, None]

    def_local = torch.matmul(Rt_def, homo)
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, :, 0:3, 0].float()

def project_position(xyzs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(xyzs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)
    Rt_canon = interpolate(xyzs, canonical_triangles, canonical_neighbours, edge_mask)

    homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, 0:1]], dim=1)[:, :, None]

    def_local = torch.matmul(Rt_def, homo)
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()

# def interpolate(xyzs, tris):
#     # Currently disabled
#     return triangle2projection(tris)[0]

# def project_position(xyzs, deformed_triangles, canonical_triangles):
#     Rt = interpolate(xyzs, deformed_triangles)
#     Rt_def = torch.linalg.inv(Rt)
#     Rt_canon = interpolate(xyzs, canonical_triangles)

#     homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, 0:1]], dim=1)[:, :, None]

#     def_local = torch.matmul(Rt_def, homo)
#     def_canon = torch.matmul(Rt_canon, def_local)

#     return def_canon[:, 0:3, 0].float()

def project_position_fwd(canonical_xyzs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt_canon = interpolate(canonical_xyzs, canonical_triangles, canonical_neighbours, edge_mask)
    Rt_canon_inv = torch.linalg.inv(Rt_canon)
    Rt_def = interpolate(canonical_xyzs, deformed_triangles, deformed_neighbours, edge_mask)

    homo = torch.cat([canonical_xyzs, torch.ones_like(canonical_xyzs)[:, 0:1]], dim=1)[:, :, None]

    canon_local = torch.matmul(Rt_canon_inv, homo)
    deformed = torch.matmul(Rt_def, canon_local)

    return deformed[:, 0:3, 0].float()


def project_direction(dirs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(dirs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)[:, 0:3, 0:3]
    Rt_canon = interpolate(dirs, canonical_triangles, canonical_neighbours, edge_mask)[:, 0:3, 0:3]

    def_local = torch.matmul(Rt_def, dirs[:, :, None])
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()