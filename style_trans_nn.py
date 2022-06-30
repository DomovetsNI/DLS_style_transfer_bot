"""
This algotithm allows to transfer stype from style_image
to original_image.
"""
from uuid import uuid4

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision.utils import save_image

# try to use new technology for Mac M1
# device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# statistic for normalization
stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

# name of layers from which we will take fearutes
layers = ['0', '5', '10', '19', '28']

# learning params
epochs = 100
lr = 0.3
alpha = 1
beta = 1e2


# func to open image and transform it
def image_loader(path, stats=stats):
    image_raw = Image.open(path).convert("RGB")
    loader = transforms.Compose([transforms.Resize((512, 512)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(stats[0], stats[1])])
    image = loader(image_raw).unsqueeze(0)
    return image.to(device, torch.float), np.array(image_raw).shape


# denormolize image after work of NN
def img_denorm(img):
    mean = np.asarray(stats[0])
    std = np.asarray(stats[1])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(img)
    res = torch.clamp(res, 0, 1)

    return(res)


# Loading the preprocessed model based on vgg19
def prepare_model():
    model = torch.load("./model.pth", map_location=torch.device(device))
    for param in model.parameters():
        # we will learn not our model, just our generated image
        model.requires_grad_(False)

    # change MaxPool to AvgPool(it was mentioned in the Net as "small trick")
    for i, layer in enumerate(model):
        if isinstance(layer, nn.MaxPool2d):
            model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    return model.to(device).eval()


# get features from certain layers
def get_features(image, model):
    features = []
    x = image
    for name, layer in enumerate(model):
        x = layer(x)
        if str(name) in layers:
            features.append(x)

    return features


# calculating content loss
def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat)**2)

    return content_l


# function for calculating gramm matrix for style loss
def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram


# calculating style loss
def calc_style_loss(gen, style, layer_weight):
    """Calcultating the style loss of each layer by calculating the MSE
    between the gram matrix of the style image and the generated image
    and adding it to style loss with certain style layer weight"""

    G = gram_matrix(gen)
    A = gram_matrix(style)
    style_l = layer_weight * torch.mean((G-A)**2)

    return style_l


def calculate_loss(gen_features,
                   orig_feautes,
                   style_featues,
                   alpha,
                   beta,
                   layer_weights=[1, 1, 0.5, 0.3, 0, 0.2]):
    # layer_weights define the "importance" of each style feature
    style_loss = 0
    content_loss = calc_content_loss(gen_features[4], orig_feautes[4])
    for gen, style, wg in zip(gen_features, style_featues, layer_weights):
        style_loss += calc_style_loss(gen, style, wg)

    # calculating the total loss with coefs
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


def style_transfer(orig_img_path, style_img_path):
    # loading the original and the style image
    original_image, orig_shape = image_loader(orig_img_path)
    style_image, _ = image_loader(style_img_path)
    # creating the generated image from the original image
    generated_image = original_image.clone().requires_grad_(True)

    model = prepare_model()

    # define optimizer
    optimizer = optim.Adam([generated_image], lr=lr)

    # run our train loop
    for epoch in range(epochs):
        # extracting the features of generated, content and the
        # original required for calculating the loss
        gen_features = get_features(generated_image, model)
        orig_feautes = get_features(original_image, model)
        style_featues = get_features(style_image, model)

        # iterating over the activation of each layer and calculate
        # the loss and add it to the content and the style loss
        total_loss = calculate_loss(gen_features,
                                    orig_feautes,
                                    style_featues,
                                    alpha,
                                    beta)
        # optimize the pixel values of the generated image and
        # backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    saving_file_name = f"./generated/{uuid4()}.png"
    pre_save_trans = transforms.Resize((orig_shape[0], orig_shape[1]))
    denormed_img = img_denorm(generated_image).cpu()
    save_image(pre_save_trans(denormed_img), saving_file_name)

    return saving_file_name
