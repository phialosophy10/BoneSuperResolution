## Packages
import torch
import torch.nn as nn
from torchvision.models import vgg19

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Loss functions
class VGGLoss(nn.Module):
    def __init__(self, layer_idx=35):
        super().__init__()
        self.layer_5_4 = layer_idx
        self.vgg_model = vgg19(pretrained=True).features[:self.layer_5_4].eval().to(DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):
        features_real = self.vgg_model(torch.tile(real_hi_res, (1, 3, 1, 1))) #torch.tile(real_hi_res, (3, 1, 1, 1)).permute(1, 0, 2, 3) #torch.cat((real_hi_res,real_hi_res,real_hi_res),1)
        features_fake = self.vgg_model(torch.tile(fake_hi_res, (1, 3, 1, 1))) #torch.tile(fake_hi_res, (3, 1, 1, 1)).permute(1, 0, 2, 3) #torch.cat((fake_hi_res,fake_hi_res,fake_hi_res),1)
        return self.loss(features_real, features_fake)

class VGGLoss3D(nn.Module):
    def __init__(self, layer_idx=36):
        super().__init__()
        self.layer_5_4 = layer_idx

        self.vgg_model = vgg19(pretrained=True).features[:self.layer_5_4].eval().to(DEVICE)
        self.loss_func = nn.MSELoss()

        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):
        loss = torch.zeros(1).to(DEVICE)
        for i in range(real_hi_res.shape[0]):
            real_patch = real_hi_res[i, :, :, :, :]
            fake_patch = fake_hi_res[i, :, :, :, :]
            for _ in range(3):
                # Since VGG accepts only RGB images, patches are tiled to (BATCH_SIZE, 3, H, W, D)
                loss = loss + self.loss_func(self.vgg_model(torch.tile(real_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3)),
                                             self.vgg_model(torch.tile(fake_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3)))
                real_patch = real_patch.permute(0, 3, 1, 2)
                fake_patch = fake_patch.permute(0, 3, 1, 2)

                # For the real and fake 3D patch do:
                # pass 3D patch through VGG network in coronal direction
                # rotate 3D patch and pass through axial direction
                # rotate 3D patch and pass through sagittal direction
                # Subtract the feature vectors of real and fake using MSELoss
                # Finally, sum over features losses for each direction.

        # Perhaps we need 1 more permute here to go back to the same patch orientation.
        return loss

