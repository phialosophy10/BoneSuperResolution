# %% Packages
import numpy as np
import matplotlib.pyplot as plt
import utils
from PIL import Image
import glob
import torch
import porespy as ps
ps.visualization.set_mpl_style()
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# %% Load images
dataset_path = "/work3/soeba/HALOS/Data/Images"
hr_paths_train = sorted(glob.glob(dataset_path + "/train_hi-res/hr/m01*.*"))
lr_paths_train = sorted(glob.glob(dataset_path + "/train_hi-res/lr/c01*.*"))

img_paths_train = []
for i in range(len(hr_paths_train)):
    img_paths_train.append([hr_paths_train[i], lr_paths_train[i]])

class ImageDataset(Dataset):
    def __init__(self, files, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.files = files
    
    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)])
        img_hr = Image.open(self.files[index][0])
        img_lr = Image.open(self.files[index][1])
        img_hr = self.hr_transform(img_hr)
        img_lr = self.lr_transform(img_lr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

# %% 
batch_size = 5
train_dataloader = DataLoader(ImageDataset(img_paths_train, hr_shape=(128, 128)), batch_size=batch_size, shuffle=True, num_workers=1)

# %% 
imgs = next(iter(train_dataloader))
lr_imgs = Variable(imgs["lr"].type(torch.Tensor))
hr_imgs = Variable(imgs["hr"].type(torch.Tensor))
sr_imgs = Variable(imgs["hr"].type(torch.Tensor))

# %% 
# thickness_hr = []
# thickness_sr = []

# %%
# for i in range(batch_size):
#     bin_im_hr = hr_imgs[i][0].detach().clone()
#     bin_im_hr[bin_im_hr<=0.40] = 0
#     bin_im_hr[bin_im_hr>0.40] = 1
#     thickness_hr.append(ps.filters.local_thickness(bin_im_hr.detach().numpy(), mode='dt'))
#     bin_im_sr = sr_imgs[i][0].detach().clone()
#     bin_im_sr[bin_im_sr<=0.40] = 0
#     bin_im_sr[bin_im_sr>0.40] = 1
#     thickness_sr.append(ps.filters.local_thickness(bin_im_sr.detach().numpy(), mode='dt'))

# %%
# fig, axs = plt.subplots(batch_size, 7, figsize=(15,12))
# for i in range(batch_size):
#     axs[i,0].imshow(lr_imgs[i][0],cmap='gray')
#     axs[i,0].axis('off')
#     axs[i,1].imshow(hr_imgs[i][0],cmap='gray')
#     axs[i,1].axis('off')
#     axs[i,2].imshow(thickness_hr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
#     axs[i,2].axis('off')
#     axs[i,3].hist(thickness_hr[i].ravel(), bins=15, density=True)
#     axs[i,4].imshow(sr_imgs[i][0],cmap='gray')
#     axs[i,4].axis('off')
#     axs[i,5].imshow(thickness_sr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
#     axs[i,5].axis('off')
#     axs[i,6].hist(thickness_sr[i].ravel(), bins=15, density=True)
# plt.show()

# %%
# hr_imgs = make_grid(hr_imgs, nrow=1, normalize=True)
# sr_imgs = make_grid(sr_imgs, nrow=1, normalize=True)
# lr_imgs = make_grid(lr_imgs, nrow=1, normalize=True)

# %%
# img_grid = torch.cat((lr_imgs, hr_imgs, sr_imgs), -1)

# plt.imshow(img_grid.permute(1, 2, 0))
# plt.show()

# %%
#def make_thickness_images_dif(hr_imgs, lr_imgs, sr_imgs):
#batch_size = len(hr_imgs)
thickness_hr = []
thickness_sr = []
dif_ims = torch.zeros(hr_imgs.shape[0],3,hr_imgs.shape[2],hr_imgs.shape[3])

# %%
for i in range(batch_size):
    bin_im_hr = hr_imgs[i][0].detach().clone()
    bin_im_hr[bin_im_hr<=0.40] = 0
    bin_im_hr[bin_im_hr>0.40] = 1
    thickness_hr.append(ps.filters.local_thickness(bin_im_hr.cpu().detach().numpy(), mode='dt'))
    bin_im_sr = sr_imgs[i][0].detach().clone()
    bin_im_sr[bin_im_sr<=0.40] = 0
    bin_im_sr[bin_im_sr>0.40] = 1
   # dif_im_tmp = np.zeros((bin_im_hr.shape[0],bin_im_hr.shape[1],3))
    dif_ims[:,0,:,:] = bin_im_hr
    dif_ims[:,1,:,:] = bin_im_sr
  #  dif_ims.append(dif_im_tmp)
    thickness_sr.append(ps.filters.local_thickness(bin_im_sr.cpu().detach().numpy(), mode='dt'))

# %%
imgs_lr = make_grid(lr_imgs[:batch_size], nrow=1, normalize=True)
imgs_hr = make_grid(hr_imgs[:batch_size], nrow=1, normalize=True)
imgs_sr = make_grid(sr_imgs[:batch_size], nrow=1, normalize=True)
#imgs_dif = make_grid(torch.from_numpy(np.transpose(np.stack(dif_ims,axis=0),(0,3,1,2))), nrow=1, normalize=True)
imgs_dif = make_grid(dif_ims, nrow=1, normalize=True)
img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr, imgs_dif), -1)

# fig1, axs = plt.subplots(batch_size, 4, figsize=(8, 10))
# for i in range(batch_size):
#     axs[i,0].imshow(lr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
#     axs[i,0].axis('off')
#     axs[i,1].imshow(hr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
#     axs[i,1].axis('off')
#     axs[i,2].imshow(sr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
#     axs[i,2].axis('off')
#     axs[i,3].imshow(dif_ims[i])
#     axs[i,3].axis('off')

# %%
fig2, axs = plt.subplots(batch_size, 4, figsize=(8, 10))
for i in range(batch_size):
    axs[i,0].imshow(thickness_hr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
    axs[i,0].axis('off')
    axs[i,1].hist(thickness_hr[i].ravel(), bins=15, density=True)
    axs[i,2].imshow(thickness_sr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
    axs[i,2].axis('off')
    axs[i,3].hist(thickness_sr[i].ravel(), bins=15, density=True)
    
#    return img_grid, fig2

# %%
plt.imshow(torch.permute(img_grid,(1,2,0)))
plt.show()
#fig2.show()
# %%
