#!/usr/bin/env python3

# ============================
# Description: a guided grad_cam example code.
# Author: Lin Zhi
# Date: 30 Oct 2020
# ============================

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
as_numpy = lambda x: x.detach().cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt

# define the preprocessing transform
image_shape = (224, 224)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

with open("data/imagenet_class_index.json") as f:
    indx2label = json.load(f)


def decode_predictions(preds, k=5):
    # return the top k results in the predictions
    return [
        [(*indx2label[str(i)], i, pred[i]) for i in pred.argsort()[::-1][:k]]
        for pred in as_numpy(preds)
    ]

class Probe:
    def get_hook(self,):
        self.data = []
        def hook(module, input, output):
            self.data.append(output)
        return hook


# load the image
print("loading the image...")
img = Image.open("./data/dog1.png")

x = transform(img)[None]  # transform and reshape it to [1, C, *image_shape]
x = x.to(device)

print("loading the model...")
### You can change the model here.
model = torchvision.models.resnet50(pretrained=True)
model.eval()
model.to(device)

def Guided_ReLU_hook(m, g_i, g_o):
    if isinstance(g_i, tuple):
        return tuple(g.clamp(min = 0) for g in g_i)
    return g_i.clamp(min = 0)

#add a probe to model
probe = Probe()
#probe will save the output of the layer4 during forward
handle = model.layer4.register_forward_hook(probe.get_hook())
#using guided relu for backprop
handle = [handle] + [
        m.register_backward_hook(Guided_ReLU_hook)
        for _, m in model.named_modules() if isinstance(m, torch.nn.ReLU)
        ]

x = x.requires_grad_()
x.retain_grad()
logits = model(x)
preds = logits.softmax(-1)

print("the prediction result:")
for tag, label, i, prob in decode_predictions(preds)[0]:
    print("{} {:16} {:5} {:6.2%}".format(tag, label, i, prob))

print("Calculating the saliency of the top prediction...")
target = preds.argmax().item()

### Grad_Cam
# get the last_conv_output
last_conv_output = probe.data[0]
last_conv_output.retain_grad() #make sure the intermediate result save its grad

#backprop
logits[0, target].backward()
for h in handle: h.remove()

grad = last_conv_output.grad 
#taking average on the H-W panel
weight = grad.mean(dim = (-1, -2), keepdim = True)
saliency = (last_conv_output * weight).sum(dim = 1, keepdim = True)
#relu
saliency = saliency.clamp(min = 0)

guided_saliency = x.grad.abs().max(dim = 1, keepdim = True).values
guided_saliency *= F.interpolate(saliency, size = guided_saliency.shape[-2:], mode = "bilinear")
plt.imshow(as_numpy(guided_saliency[0, 0]))
plt.show()
