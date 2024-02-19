# Pose-Refinement-with-Predicted-3D-Model
This project aims to get a precise pose from an initial pose with predicted 3d model generated by [One-2-3-45++](http://sudo.ai/3dgen).
The pipeline is as follow:
1. Get images of different views from ground truth 3d model.(Obtained from Objaverse, Sketchfeb)
2. Use one of the images to get the predicted 3d model from [One-2-3-45++](http://sudo.ai/3dgen).
3. Use the predicted model and an image rendered from the ground truth model(with the initial coarse pose) to get a precise pose of the image.

We use Pytorch3d as the framework.

First, we try to refine the pose from the ground truth model to the ground truth model.(or from predicted model to the predicted model)
The silhouette render is effective for this problem. We can use the silhoutte render, and take the loss as the MSE loss between the alpha channels of the initial image and the target image to optimize the pose(where the initial image is the image rendered with the initial coarse pose, and the target image is the image rendered with the target pose). Since only the alpha channel works for the optimization, texture of the 3d model is not needed. Therefore, we refine the pose based on the 3d model without texture.

Here is an example:
The ground truth model is:

![rgb_0006](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/fc9e7697-bf34-4dd4-b5e3-138a43563715)
![rgb_0005](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/88cba9a8-e7dd-4e72-a62c-3db7363a9693)
![rgb_0004](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/8b54333b-8268-4a83-acff-570e879d09c5)
![rgb_0003](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/92988530-9517-40bb-88ae-827ebd0344e9)
![rgb_0002](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/fd8f1a6e-d943-4502-abe3-e344fe92fba3)
![rgb_0001](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/c21e8de1-ac0d-4433-97f5-951b0210d837)
![rgb_0000](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/8ba9f775-b893-47d7-bd85-a38c10d87b4e)

The predicted model is:

![rgb_0006](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/cce3eace-1ca2-4aab-9576-e70dedc578cb)
![rgb_0005](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/9f84cf9e-856e-4a1d-90b1-060d618ce9ac)
![rgb_0004](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/a9fc60ac-cb05-473f-9ebc-9c0443953ba0)
![rgb_0003](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/ce2f588d-4ca2-4542-b02f-b6e1c1392b67)
![rgb_0002](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/71c802e9-3e4d-41da-9a4c-4acbbc243481)
![rgb_0001](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/411d2544-c2b7-4c8c-986c-6dc154c8bf24)
![rgb_0000](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/958c0074-6f4a-421a-9faf-7716f31e4007)


The initial image:

![initial_pre_gray](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/fafea8a6-25c5-43bc-a070-dea8e75fa677)

The target image:

![target_gt_gray](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/2f5801ec-07c1-44c1-9e15-c5ddd08f21b9)

The refinement process is visualized as:

![siholette_demo](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/1317faea-32e1-4608-9bf5-2d7b09ff00ed)

Here, whatever the initial pose is set, it can always be refined to the target pose. The pose always converges to the value and the error is very small.
target: [0, 0.5, 2.0] initial： any
result: [0.0189, 0.4826, 2.0482]
Here we use the coordinates xyz, and the direction is always set to face the model.


However, using silhoutte render and taking alpha channel as loss can not refinement the pose with the predicted 3d model and the ground truth image. The reason is that the predicted model is not perfect. The geometry and the texture are all different. The gap is large enough that optimization cannot converge. Therefore, phong render is what we needed because we can extract feature and some other high level information from the predicted model and the ground truth model. Now the loss is the MSE loss between the rgb channels of the initial image and the target image.

There is one thing worth to mention: if we use the rgb channel as the loss, the silhoutte render can not backward the parameters. Only if taking the alpha channel as the loss can make the optimization continue. On the contrast, the phong render can only backward the parameters during the optimization only when taking the rgb channel as the loss. The alpha channel can not work for the phong render.

We found that, when the difference between the initial pose and target pose is not very huge, the optimization can converge.
Here is an example:

The initial image:

![initial_pre_0 2](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/6c77a511-3409-4a54-b249-eef10c87d04e)

The target image:

![target_pre_0 2](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/f1959b03-cc78-4df5-96d2-87459c79fe96)

The refinement process is:

![pre_0 2_nofeature](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/6f380b5c-a2c6-4339-8e53-a9867e7b891f)

target: [0, 0.5, 2.0] initial: [0.0, 0.7, 2.2]
result: [-0.0118,  0.4876,  2.2365]

The result is not satisfactory obivously. We know that only considering the loss between the rgb channels can not work perfectly. Therefore, we introduce the feature loss(using VGG as the backbone) and set the loss = img_loss + 0.005 * feature loss.

The refinement process is：

![pre_0 2_withfeature](https://github.com/bobojiang26/Pose-Refinement-with-Predicted-3D-Model/assets/91231457/083e570a-aaf4-4b31-8545-7a00e1a54604)

target： [0, 0.5, 2.0] initial： [0.0, 0.7, 2.2]
result: [-0.0394,  0.5082,  2.0196]

We can find that the error is less than that of optimization without the feature loss.


Problem to solve:
1. The sizes and the poses of the ground truth 3d model and the predicted one are different(not aligned).
2. Find out when the difference between the initial image(the predicted model) and the target image(the ground truth model) is not big, whether the pose can be refined well.
3. If the pose can not be refined well based on the present method, we will try to add the correspondence loss.
