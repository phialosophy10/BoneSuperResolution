## Packages
import torch

## GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IN_C = 1
K_FACTOR = 12  # 6
K_SIZE = 3
PATCH_SIZE = 64

VALIDATION_FREQUENCY = 1

LOSS_WEIGHTS = {
    "MSE": 0, #5e-2,  # 1e-2 for WGAN-GP
    "L1": 1.0, #1.0,  # 1e-2 for WGAN-GP. L1 loss is used in mDCSRN-GAN paper as opposed to L2/MSE loss
    "BCE_Logistic": 0, #1.0,
    "BCE": 0, #1.0,
    "VGG": 0,  # 0.006
    "VGG3D": 6e-3, #0.006,  # 0
    "GRAD": 0,  # 0.6
    "LAPLACE": 0,
    "TV3D": 0,  # 0.05
    "TEXTURE3D": 0,  # 0.5
    "ADV": 1e-3,  # 10**-3,  # is 0.1 in mDCSRN-GAN paper, but uses WGAN-GP instead of vanilla GAN
    "STRUCTURE_TENSOR": 0  # 10**-7
}
