import os
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import trimesh
import cv2
import torchvision
import torchvision.models as models
from PIL import Image

import segmentation_models_pytorch as smp

# io utils
from pytorch3d.io import load_obj, load_objs_as_meshes

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    TexturesUV, TexturesAtlas
)


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def get_features(image, model, layers=None):
    """提取在指定层的特征"""
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def tensor_gray_scale(img):
    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    weights = weights.to(img)
    gray_img = torch.sum(img * weights, dim=1, keepdim=True)
    return gray_img

obj_filename = "/home/zcb/self_code_training/pose_refinement_from_real_3dmodel_2_8/glb_data/mclaren_senna_glb/obj_coarse/mclaren_senna_glb_coarse.obj"
verts, faces, aux = load_obj(
    obj_filename,
    device = device,
    load_textures = False,
    create_texture_atlas = True,
    texture_atlas_size = 4,
    texture_wrap = 'repeat'
)

atlas = aux.texture_atlas

mesh = Meshes(
    verts = [verts],
    faces = [faces.verts_idx],
    textures = TexturesAtlas(atlas=[atlas])
)
# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)


raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=100
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

# Select the viewpoint using spherical angles  
distance = 1   # distance from camera to the object
elevation = 50.0   # angle of elevation in degrees
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# x = distance * np.cos(elevation) * np.sin(azimuth)
# y = distance * np.sin(elevation)
# z = distance * np.cos(elevation) * np.cos(azimuth)
# print('x_gt: ',x, ' y_gt: ', y, ' z_gt: ', z)

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
# print('R_gt: ', R, ' T_gt: ', T)

# Render the teapot providing the values of R and T. 
# silhouette = silhouette_renderer(meshes_world=mesh, R=R, T=T)
# image_ref = phong_renderer(meshes_world=mesh, R=R, T=T)

# silhouette = silhouette.cpu().numpy()
# image_ref = image_ref.cpu().numpy() # 1 * 256 * 256 * 4

camera_target_pose = torch.from_numpy(np.array([0.0, 0.5, 2.0], dtype=np.float32)).to(device)
R = look_at_rotation(camera_target_pose[None, :], device=device)  # (1, 3, 3)
T = -torch.bmm(R.transpose(1, 2), camera_target_pose[None, :, None])[:, :, 0]   # (1, 3)
        
image_ref = phong_renderer(meshes_world=mesh, R=R, T=T).cpu().numpy()

# Write the reference image
images = torch.from_numpy(image_ref).permute(0,3,1,2)
torchvision.utils.save_image(images, "target.png")





class Model(nn.Module):
    def __init__(self, meshes, renderer):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        # self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([2.0, 2.0, 5.0], dtype=np.float32)).to(meshes.device))

    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        return image
    
# Save images periodically and compose them into a GIF.
filename_output = "./car_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# Using VGG as feature extractor network
vgg = models.vgg19(pretrained=True).features
vgg.eval().cuda()

# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=mesh, renderer=silhouette_renderer).to(device)

segementation_model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

# Using MSE loss
mse_loss = torch.nn.MSELoss()

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# plt.figure(figsize=(10, 10))

# _, image_init = model()

# Visualize the input image and target image
# plt.subplot(1, 2, 1)
# plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
# plt.grid(False)
# plt.title("Starting position")

# plt.subplot(1, 2, 2)
# plt.imshow(model.image_ref.cpu().numpy().squeeze())
# plt.grid(False)
# plt.title("Reference silhouette")
# plt.savefig('target.png')
# plt.close()

loop = tqdm(range(200))


# image_ref = cv2.imread('/home/zcb/self_code_training/pose_refinement_from_real_3dmodel_2_8/target.png', cv2.IMREAD_UNCHANGED)
transform = transforms.Compose([
    transforms.ToTensor()
])

# image_ref_train = transform(image_ref).permute(1,2,0).unsqueeze(0).to(device)
image_ref_train = torch.from_numpy(image_ref).to(device)

img = model().permute(0,3,1,2)
torchvision.utils.save_image(img, "initial.png")


for i in loop:
    optimizer.zero_grad()
    img = model()  # batches * h * w * channels
    loss_img = mse_loss(img, image_ref_train)

    img_rgb = img[...,3]
    image_ref_train_rgb = image_ref_train[...,3]
    loss_img = mse_loss(img_rgb, image_ref_train_rgb)


    # feature loss
    # features1 = get_features(img_rgb.permute(0,3,1,2), vgg)
    # features2 = get_features(image_ref_train_rgb.permute(0,3,1,2), vgg)
    # feature_loss = 0
    # for feature1, feature2 in zip(features1.values(), features2.values()):
    #     feature_loss += mse_loss(feature1, feature2)

    loss = loss_img

    # if i >= 20:
    #     loss = loss_img
    loss.backward()
    optimizer.step()
    
    loop.set_description('Optimizing (loss %.4f)' % loss.data)
    
    # if loss.item() < 200:
    #     break
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        R = look_at_rotation(model.camera_position[None, :], device=model.device)
        T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]   # (1, 3)

        # visualize the R and T in the refinement process
        print('position_refined: ',model.camera_position)

        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)
        
        plt.figure()
        plt.imshow(image[..., :3])
        plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        plt.axis("off")
        plt.savefig('training_output/rendered_image'+str(i)+'.png')
        plt.close()

        print('loss: ', loss.item())
    
writer.close()