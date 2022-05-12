import argparse
from tkinter.messagebox import NO

import cv2
import numpy as np
import timm
import torch
from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM, FullGrad,
                              GradCAM, GradCAMPlusPlus,
                              GuidedBackpropReLUModel, ScoreCAM, XGradCAM)
from pytorch_grad_cam.utils.image import (deprocess_image, preprocess_image,
                                          show_cam_on_image)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models

from modeling.ResNet import resnet18, resnet34

# from torchvision.models import resnet50


rgb_img = cv2.imread('dog_cat.jfif', 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224,224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

# model = models.resnet18(pretrained=True)
# model = torch.load('models/resnet18.pth')
import timm

# lambda_resnet50ts lambda_resnet26t
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
# model = timm.create_model('resnet18', pretrained=True)

# model = resnet18()
# model.load_state_dict(torch.load('models/LambdaR18_0_lambda2_r_None.pt',map_location=torch.device('cpu')))

# model.stages[-1][-1]
# [layer[-1] for layer in model.stages] 
# target_layers = [model.stages[-1][-1]] 
target_layers = model.blocks[-1].norm1
# target_layers = [model.layer4[-1]] 
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#  FullGrad, GradCAM


# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
# targets = None

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True,)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# cam_image = show_cam_on_image(rgb_img, grayscale_cam)

# gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
# gb = gb_model(input_tensor, target_category=None)

# cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
# cam_gb = deprocess_image(cam_mask * gb)
# gb = deprocess_image(gb)

targets = None

# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets ,
                    eigen_smooth=True,)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam)

cv2.imwrite('cam.jpg', cam_image)

