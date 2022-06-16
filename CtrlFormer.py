import torch
from torch import nn

from timm.models import vit_toy_patch6_84

class Timm_Encoder_toy(nn.Module):
    def __init__(self,obs_shape, feature_dim):
        super().__init__()
        self.num_step=int(obs_shape[0]/3)
        self.feature_dim = feature_dim
        self.image_encode = vit_toy_patch6_84()
        self.linear_map = nn.Linear(192, 50)
        self.byol_project = nn.Sequential(
            nn.Linear(192, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 96),
            nn.BatchNorm1d(96),
        )
        self.byol_predict = nn.Sequential(
            nn.Linear(96, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 96),
        )
        # self.build_decoder()

    def set_reuse(self):
        self.image_encode.copy_token()


    def forward_1(self,img_sequence,detach):

        latent = self.image_encode.forward_features2(img_sequence)
        policy_feature = self.linear_map(latent)
        if detach:
            policy_feature=policy_feature.detach()

        return policy_feature
    
    def forward_2(self,img_sequence,detach):

        latent = self.image_encode.forward_features3(img_sequence)
        policy_feature = self.linear_map(latent)
        if detach:
            policy_feature=policy_feature.detach()

        return policy_feature
    
    def forward_0(self,img_sequence,detach):

        latent = self.image_encode.forward_features1(img_sequence)
        policy_feature = self.linear_map(latent)
        if detach:
            policy_feature=policy_feature.detach()

        return policy_feature
    
    def get_rec(self, input):
        result = self.decoder_input(input)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward_rec(self,img_sequence):

        rec = self.image_encode.forward_reconstruction(img_sequence)
        # rec = self.get_rec(rec)
        return rec

    # def build_decoder(self):
    #     modules = []
    #     hidden_dims = [32, 128, 256, 512]
    #     hidden_dims.reverse()
    #     self.decoder_input = nn.Linear(192, 2048)
    #     for i in range(len(hidden_dims) - 1):
    #         modules.append(
    #             nn.Sequential(
    #                 nn.ConvTranspose2d(hidden_dims[i],
    #                                    hidden_dims[i + 1],
    #                                    kernel_size=3,
    #                                    stride=4,
    #                                    padding=1,
    #                                    output_padding=1),
    #                 nn.LeakyReLU())
    #         )
    #     self.decoder = nn.Sequential(*modules)
    #     self.final_layer = nn.Sequential(
    #         nn.Conv2d(hidden_dims[-1], out_channels=9,
    #                   kernel_size=3, padding=0)
    #     )
