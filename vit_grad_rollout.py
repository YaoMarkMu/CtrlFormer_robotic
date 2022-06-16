import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from math import *
import matplotlib.pyplot as plt
pool= torch.nn.MaxPool2d((2,2), stride=(2,2))
pool1= torch.nn.MaxPool1d(3, stride=3)
def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(192)
    for attention, grad in zip(attentions, gradients):      
        print("debug",attention.shape)
        print("debug",grad.shape)
        plt.imshow(255*attention[0].detach().cpu().numpy().transpose(1,2,0))
        plt.show()
        plt.pause(1)   

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor,detach=False)
        loss = (output).sum()
        loss.backward()
        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)