import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    # look_at_rotation,
    PerspectiveCameras,
    PointLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.mesh.rasterizer import Fragments
from typing import Optional
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast, TensorProperties
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend

from .pbr_materials import PBRMaterials

from flare.core import Camera

def get_renderers(image_size, 
                  light_posi=((1.0, 1.0, -5.0),),
                  silh_sigma=1e-7, 
                  silh_gamma=1e-1, 
                  silh_faces_per_pixel=50,
                  device='cuda'):
    # From pytorch3D tutorial
    cameras = PerspectiveCameras(device=device, in_ndc=False)
    ### Shlhouette renderer ###
    # To blend the 50 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(background_color=(0,0,0), sigma=silh_sigma, gamma=silh_gamma)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=silh_faces_per_pixel, 
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params),
        
    )

    ### Color renderer ###
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, # 0, # 10, # 1, 
    )

    blend_params = BlendParams(background_color=(1.0,1.0,1.0))
    # For casted shadow
    lights = PointLights(device=device, location=light_posi,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.4, 0.4, 0.4),),
        specular_color=((0.1, 0.1, 0.1),))
        
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShaderPBR(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    ### Normal renderer ###
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=10, # 1, 
    )

    blend_params = BlendParams(background_color=(1.0,1.0,1.0))
    # We can add a point light in front of the object. 
    # lights = PointLights(device=device, location=((1.0, 1.0, -5.0),))
    lights = PointLights(device=device, location=light_posi)
    phong_normal_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=SoftPhongNormalShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    return phong_renderer, silhouette_renderer, phong_normal_renderer

# Copied directly from SoftPhongShader, the edit is in phong_shading_PBR
class SoftPhongShaderPBR(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading_PBR(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


def phong_shading_PBR(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    if materials.use_normal_map:
        pixel_normals = materials.apply_normal_map(
            pixel_normals,
            fragments,
            faces[: meshes.num_faces_per_mesh()[0]],
            verts[: meshes.num_verts_per_mesh()[0]],
        )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors


def render_rgb_test(phong_renderer, img_size, focal_length, mano_verts, mano_faces, camera_t, img_width, img_height, device):
    """
    Render RGB image given MANO vertex locations and MANO faces
    """
    verts_rgb_init = torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).tile([778, 1])
    textures = TexturesVertex(verts_features=[verts_rgb_init])
    # faces = mano_layer.th_faces
    mesh = Meshes(mano_verts.to(device), mano_faces.unsqueeze(0).to(device), textures.to(device))

    fx = -focal_length
    fy = -focal_length
    px = img_size / 2. # 112.0
    py = img_size / 2. # 112.0
    target_rgb = phong_renderer(mesh,
                    principal_point=((px, py), ),
                    focal_length=((fx, fy),),
                    T=camera_t,
                    image_size=((img_width, img_height),)
                 )[:, :, :, 0:3]

    return target_rgb


def phong_normal_shading(meshes, fragments, materials) -> torch.Tensor:
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals #  # ones
    )
    if materials.use_normal_map:
        pixel_normals = materials.apply_normal_map(
            pixel_normals,
            fragments,
            faces[: meshes.num_faces_per_mesh()[0]],
            verts[: meshes.num_verts_per_mesh()[0]],
        )
    pixel_normals[:, :, :, :, 1] = pixel_normals[:, :, :, :, 1] * -1.
    pixel_normals[:, :, :, :, 2] = pixel_normals[:, :, :, :, 2] * -1.
    pixel_normals = (pixel_normals + 1.0) / 2.0

    return pixel_normals


class SoftPhongNormalShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        # texels = meshes.sample_textures(fragments)
        # lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_normal_shading(
            meshes=meshes,
            fragments=fragments,
            materials=materials,
        )
        colors = colors.squeeze(-2)
        # colors
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


### for shadow rendering

class MeshRendererShadow(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self
    
    def create_arange2d(self, batch_size, max_len):
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.expand(max_len, max_len)
        tensor_2d = torch.stack([range_tensor.T, range_tensor], dim=2)
        tensor_2d = tensor_2d.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return tensor_2d

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments_from_light = self.rasterizer(meshes_world, **kwargs)
        kwargs_light = {}
        for (k,v) in kwargs.items():
            kwargs_light[k] = v
        # Clone cameras for later projection 
        light_cam_clone = self.rasterizer.cameras.clone()
        # There is a side effect in rasterizer that will set camera 'R' and 'T', which means the settings in the kwargs will be ignored in the shader
        kwargs['T'] = kwargs['cam_T']
        kwargs['R'] = kwargs['cam_R']
        fragments_from_cam = self.rasterizer(meshes_world, **kwargs)
        batch_size = fragments_from_light[0].shape[0]
        max_face = meshes_world[0].faces_packed().shape[0] + 1  # plus 1 to handle "-1"
        
        visibility_map = torch.zeros_like(fragments_from_cam[0], dtype=torch.float)

        #### get 3d coordinate of hit points from cam 
        verts = meshes_world.verts_packed()  # (V, 3)
        faces = meshes_world.faces_packed()  # (F, 3)
        faces_verts = verts[faces]
        # Get hit points from cam
        pixel_3d_cam_world = interpolate_face_attributes(fragments_from_cam[0], fragments_from_cam[2], faces_verts)
        (N, H, W, K, _) = pixel_3d_cam_world.shape
        pixel_3d_cam_world_flat = pixel_3d_cam_world.reshape([N, H * W * K, 3])
        # Transform them to NDC of the light cam

        # NOTE: Retaining view space z coordinate for now.
        # verts_view = light_cam_clone.get_world_to_view_transform(**kwargs).transform_points(
        #     pixel_3d_cam_world_flat, eps=None
        # )
        # Get depth in the light space
        # transform_points
        # pixel_3d_cam_world_flat_z = light_cam_clone.transform_points(pixel_3d_cam_world_flat, **kwargs_light)

        # Transform the points hit in from cam (now in world space) to light space
        # only care about z from the following transformation
        depth_cam_from_light = light_cam_clone.get_world_to_view_transform(**kwargs_light).transform_points(pixel_3d_cam_world_flat)
        depth_cam_from_light_reshape = depth_cam_from_light.reshape([N, H, W, K, 3])
        # From the test below, this appears to only transfrom x and y, x is not transform correctly
        hit_pixel_in_light_cam = light_cam_clone.transform_points_screen(pixel_3d_cam_world_flat, **kwargs_light)
        hit_pixel_in_light_cam = hit_pixel_in_light_cam.reshape([N, H, W, K, 3])
        
        xy_key = hit_pixel_in_light_cam[..., :2].round().long()
        xy_key = xy_key.reshape(N * H * W * K, 2)
        batch_key = torch.arange(batch_size).repeat_interleave(H * W * K).to(fragments_from_cam[0].device)
        xy_batch_key = torch.cat([batch_key.unsqueeze(1), xy_key], dim=1)

        # average shadow from [-1,1] around the projected (x,y)
        visibility_map = torch.zeros_like(fragments_from_light[1])
        image_size = kwargs['image_size'][0][0]
        filter_size = 1
        shadow_step = 1
        for ii in range(-filter_size, filter_size + 1):
            for jj in range(-filter_size, filter_size + 1):
                depth_at_xy = fragments_from_light[1][xy_batch_key[:, 0], 
                                                     (xy_batch_key[:, 2] + ii * shadow_step).clamp(0, image_size-1),
                                                     (xy_batch_key[:, 1] + jj * shadow_step).clamp(0, image_size-1)]
                # This is the depth from light of the pixel seen in light view
                # Those points are transformed into cam [x,y]
                depth_at_xy = depth_at_xy.reshape(N, H, W, K)
                aa = depth_cam_from_light_reshape[:, :, :, :, 2] - 0.008
                bb = depth_at_xy
                
                visibility_map += torch.sigmoid((bb - aa) * 1000.0) # 1000.0
        
        visibility_map = visibility_map / ((1 + 2*filter_size)**2)
        kwargs['vis_map'] = visibility_map
        images = self.shader(fragments_from_cam, meshes_world, **kwargs)

        return images



def get_shadow_renderers(image_size, 
                  light_posi=((1.0, 1.0, -5.0),),
                  silh_sigma=1e-7, 
                  silh_gamma=1e-1, 
                  silh_faces_per_pixel=50,
                  amb_ratio=0.6,
                  device='cuda'):

    cameras = PerspectiveCameras(device=device, in_ndc=False)

    # if want to use SoftPhong, need to split the rasterizer for shadow and shading
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )

    blend_params = BlendParams(background_color=(1.0,1.0,1.0))

    # For testing casted shadow
    ambient_color = torch.ones([1,3], device=device) * amb_ratio
    diffuse_color = 1.0 - ambient_color
    lights = PointLights(device=device, location=light_posi,
        ambient_color=ambient_color,  # ((amb_ratio, amb_ratio, amb_ratio),),
        diffuse_color=diffuse_color,  # ((diff_ratio, diff_ratio, diff_ratio),),
        specular_color=((0.0, 0.0, 0.0),))

    shadow_renderer = MeshRendererShadow(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShaderShadow(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    return shadow_renderer


# def process_info_for_shadow(cam, light_positions, hand_verts_center, image_size, focal_length, device='cuda'):
#     cam_T = torch.stack([-cam[:, 1], -cam[:, 2], 2 * focal_length/(image_size * cam[:, 0] +1e-9) ], dim=1).to(device)
#     cam_at_light_t = light_positions.to(device) # cam at [[0.0363, 0.0879, 1.1640]]
    
#     batch_size = cam.shape[0]
#     cam_R = torch.Tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).repeat(batch_size, 1, 1).to(device)
#     # NOTE: the normal is filp because pytorch3D camera is pointing to +Z, which make pixel normal pointing to -Z

#     # Project light position to the surface of a sphere around hand vertex center
#     radius = 1.5 # 1.0
#     cam_at_light_t_r = hand_verts_center + (cam_at_light_t - hand_verts_center) * (radius / torch.linalg.norm(cam_at_light_t - hand_verts_center, dim=1, keepdim=True))
    
#     light_R = look_at_rotation(cam_at_light_t_r, at=hand_verts_center, up=((0, 1, 0),)).to(device)  # (1, 3, 3)
#     light_T = -torch.bmm(light_R.transpose(1, 2), cam_at_light_t_r[:, :, None])[:, :, 0]   # (1, 3)
#     return light_R, light_T, cam_R, cam_T

def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = -F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R#.transpose(1, 2)

def process_info_for_shadow(cam, light_positions, hand_verts_center, device='cuda', radius = 4.5):
    # batch_size = cam.shape[0]
    batch_size = len(cam)
    cam_intrinsics = cam[0].K.squeeze(0)
    # cam_R, cam_T = [], []
    # for i in range(batch_size):
    #     R = cam[i].R.clone().detach()
    #     R[0][0] *= -1 # turn into pytorch3d coords
    #     cam_R.append(R)
    #     cam_T.append(cam[i].t.clone().detach())
    #     # cam_intrinsics.append(views_subset['camera'][i].K)
    # cam_R = torch.stack(cam_R, dim=0).to(device) 
    # cam_T = torch.stack(cam_T, dim=0).to(device)    
    
    # cam_T = torch.stack([-cam[:, 1], -cam[:, 2], 2 * focal_length/(image_size * cam[:, 0] +1e-9) ], dim=1).to(device)
    #  # cam at [[0.0363, 0.0879, 1.1640]]
    
    # batch_size = cam.shape[0]
    # cam_R = torch.Tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).repeat(batch_size, 1, 1).to(device)
    # # NOTE: the normal is filp because pytorch3D camera is pointing to +Z, which make pixel normal pointing to -Z
    
    # cam_at_light_t = light_positions.clone().detach().to(device)
    cam_center = []
    for i in range(batch_size):
        cam_center.append(cam[i].center)
    cam_center = torch.stack(cam_center, dim=0)

    light_cams = []
    for light_position in light_positions:
        cam_at_light_t = light_position.to(device)
        # cam_at_light_t[:,1] = cam_center[:,1].clone().detach()
        # Project light position to the surface of a sphere around hand vertex center
        # radius = 4.5 # 1.0
        # radius = 6.5 # 1.0
        cam_at_light_t_r = hand_verts_center + (cam_at_light_t - hand_verts_center) * (radius / torch.linalg.norm(cam_at_light_t - hand_verts_center, dim=1, keepdim=True))

        # cam_at_light_t_r = cam_at_light_t
        
        light_R = look_at_rotation(cam_at_light_t_r, at=((0, 0, 0),), up=((0, 1, 0),)).to(device)  # (1, 3, 3)
        # light_R = look_at_rotation(cam_at_light_t_r, at=hand_verts_center, up=((0, 1, 0),)).to(device)  # (1, 3, 3)
        # light_R[:,0,0] *= -1
        # light_R[:,1,1] *= -1
        # light_T = - cam_at_light_t
        # light_T = -torch.bmm(light_R.transpose(1, 2), cam_at_light_t_r[:, :, None])[:, :, 0]   # (1, 3)
        light_T = -torch.bmm(light_R, cam_at_light_t_r[:, :, None])[:, :, 0]
        # light_T = cam_at_light_t_r

        # gl_transform = torch.tensor([[-1., 0,  0],
        #                             [0,  1., 0],
        #                             [0,  0, 1.]], device=device)
        # light_R = gl_transform[None,:,:] @ light_R

        light_cam = []
        for i in range(batch_size):
            R = light_R[i]#.clone().detach()
            t = light_T[i]#.clone().detach()
            camera = Camera(cam_intrinsics, R, t, device=device)
            light_cam.append(camera)
        
        light_cams.append(light_cam) 

    # light_cam = cam.copy()
    # for i in range(batch_size):
    #     light_cam[i].R = light_R[i].clone().detach()
    #     light_cam[i].t = light_T[i].clone().detach()

    return light_cams#, light_R, light_T#, cam_R, cam_T


from pytorch3d.renderer.mesh.shading import _apply_lighting
def phong_shading_with_shadow(
    meshes, fragments, lights, cameras, materials, texels, vis_maps ##
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    # TODO: Add other material properties
    if materials.use_normal_map:
        pixel_normals = materials.apply_normal_map(
            pixel_normals,
            fragments,
            faces[: meshes.num_faces_per_mesh()[0]],
            verts[: meshes.num_verts_per_mesh()[0]],
        )

    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    # colors = (ambient + diffuse) * texels + specular
    vis_maps = vis_maps.unsqueeze(-2).repeat(1, 1, 1, diffuse.shape[3], 1)
    colors = (ambient + diffuse * vis_maps) * texels + specular
    # colors = (ambient + diffuse) * texels + specular
    # colors = (ambient) * texels + specular
    # colors = (diffuse) * texels + specular
    # colors = vis_maps
    return colors


class SoftPhongShaderShadow(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        vis_maps = kwargs["vis_map"]

        colors = phong_shading_with_shadow(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
            vis_maps=vis_maps
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


def render_image_with_RT(mesh, light_t, light_r, cam_t, cam_r, batch_size, renderer, img_size, focal_length,
    silhouette=False, materials_properties=dict(), device='cuda'): # cam_r

    materials = PBRMaterials(
        device=device,
        shininess=0.0,
        **materials_properties,
    )
    fx = focal_length
    px = img_size / 2.
    py = img_size / 2.
    cam_t = cam_t.to(device)
    cam_r = cam_r.to(device)
    light_t = light_t.to(device)
    light_r = light_r.to(device)

    rendered_img = renderer(mesh,
                    principal_point=torch.Tensor([(px, py)]), # pp_batch,
                    focal_length=fx, # f_batch,  # ((fx, fy),),
                    T=light_t,
                    R=light_r, # R_batch,
                    cam_T=cam_t,
                    cam_R=cam_r,
                    materials=materials, 
                    image_size=torch.Tensor([(img_size, img_size)]), # ((img_width, img_height),)
                )
    if silhouette:
        rendered_img = rendered_img[:, :, :, 3]
    else:
        # color
        rendered_img = rendered_img[:, :, :, 0:3]
    return rendered_img

def prepare_mesh(params, fid, mano_layer, verts_textures, mesh_subdivider, global_pose, configs, device='cuda', 
        vis_normal=False, shared_texture=True, use_arm=False):
    batch_size = fid.shape[0]  # params['pose'].shape[0]

    from pytorch3d.renderer import (
        TexturesUV,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    uv_map = params['texture'][None, 0].repeat(batch_size, 1, 1, 1).to(device) 
    textures = TexturesUV(maps=uv_map,
                        faces_uvs=params['faces_uvs'].repeat(batch_size, 1, 1).type(torch.long).to(device),
                        verts_uvs=params['verts_uvs'].repeat(batch_size, 1, 1).to(device))
    mesh = Meshes(hand_verts.to(device), faces.to(device), textures.to(device))
    return hand_joints.to(device), hand_verts.to(device), faces.to(device), textures.to(device) # mesh


def prepare_materials(params, batch_size, shared_texture=True, device='cuda'):
    if 'normal_map' in params:
        if shared_texture:
            normal_maps = params['normal_map'][None, 0].repeat(batch_size, 1, 1, 1).to(device)
        else:
            normal_maps = params['normal_map'].to(device)

        normal_maps = torch.nn.functional.normalize(normal_maps, dim=-1)
        normal_maps = TexturesUV(maps=normal_maps,
                                faces_uvs=params['faces_uvs'].repeat(batch_size, 1, 1).type(torch.long).to(device),
                                verts_uvs=params['verts_uvs'].repeat(batch_size, 1, 1).to(device))
    else:
        normal_maps = None

    materials_prop = {
        "normal_maps": normal_maps,
    }
    return materials_prop