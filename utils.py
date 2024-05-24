import porespy as ps
ps.visualization.set_mpl_style()
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import localthickness as lt
from skimage.filters import threshold_otsu
from torchvision.utils import save_image, make_grid
import torch
import monai
from monai.networks.nets import UNet
import glob
import argparse
import time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

# Function to take in command line arguments
def get_cmd_args():
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
        "model_type",
        type=str,
        choices=["ESRGAN", "SRGAN"],
        default="ESRGAN",
        help="model architecture (ESRGAN or SRGAN)",
    )
    CLI.add_argument(
        "data_type",
        type=str,
        choices=["real", "synth"],
        default="real",
        help="input data type (real or synthetic)",
    )
    CLI.add_argument(
        "n_epochs",
        type=int,
        default=5,
        help="Number of epochs to train the model",
    )
    CLI.add_argument(
        "batch_size",
        type=int,
        default=12,
        help="Batch size for training",
    )
    CLI.add_argument(
        "lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    CLI.add_argument(
        "pix_loss",
        type=str,
        choices=["L1", "MSE"],
        default="L1",
        help="Type of pixel-loss (L1 or MSE)"
    )
    CLI.add_argument(
        "--loss_coeffs",
        nargs=3,
        type=float,
        default=[0.001, 1.0, 0.006],
        help="Coefficients for the three loss terms: adv, pixel, cont (default: 0.001, 1.0, 0.006)"
    )
    CLI.add_argument(
        "--test",                   # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",                  # 0 or more values expected => creates a list
        type=str,
        default=["002", "086"],     # default if nothing is provided
        help="Bones to test on (images shown during training)"
    )
    CLI.add_argument(
        "--train",
        nargs="*",
        type=str,                   # any type/callable can be used here
        default=["013", "015", "021"],
        help="Bones to train on"
    )
    CLI.add_argument(
        "--valid",
        nargs="*",
        type=str,                  
        default=["002", "086", "138"],
        help="Bones to validate on (calculations after finished training)"
    )
    args = CLI.parse_args()
    
    return args

# Function to make image grid with hr, lr, sr and thickness image and histogram
def make_thickness_images(hr_imgs, lr_imgs, sr_imgs):
    batch_size = len(hr_imgs)
    thickness_hr = []
    thickness_sr = []

    for i in range(batch_size):
        bin_im_hr = hr_imgs[i][0].detach().clone()
        bin_im_hr[bin_im_hr<=0.40] = 0
        bin_im_hr[bin_im_hr>0.40] = 1
        thickness_hr.append(ps.filters.local_thickness(bin_im_hr.cpu().detach().numpy(), mode='dt'))
        bin_im_sr = sr_imgs[i][0].detach().clone()
        bin_im_sr[bin_im_sr<=0.40] = 0
        bin_im_sr[bin_im_sr>0.40] = 1
        thickness_sr.append(ps.filters.local_thickness(bin_im_sr.cpu().detach().numpy(), mode='dt'))

    fig, axs = plt.subplots(batch_size, 7, figsize=(15,12))
    for i in range(batch_size):
        axs[i,0].imshow(lr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,0].axis('off')
        axs[i,1].imshow(hr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,1].axis('off')
        axs[i,2].imshow(thickness_hr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,2].axis('off')
        axs[i,3].hist(thickness_hr[i].ravel(), bins=15, density=True)
        axs[i,4].imshow(sr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,4].axis('off')
        axs[i,5].imshow(thickness_sr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,5].axis('off')
        axs[i,6].hist(thickness_sr[i].ravel(), bins=15, density=True)
    
    return fig

def make_thickness_images_dif(hr_imgs, lr_imgs, sr_imgs):
    batch_size = len(hr_imgs)
    thickness_hr = []
    thickness_sr = []
    dif_ims = []

    for i in range(batch_size):
        bin_im_hr = hr_imgs[i][0].detach().clone()
        bin_im_hr[bin_im_hr<=0.40] = 0
        bin_im_hr[bin_im_hr>0.40] = 1
        thickness_hr.append(ps.filters.local_thickness(bin_im_hr.cpu().detach().numpy(), mode='dt'))
        bin_im_sr = sr_imgs[i][0].detach().clone()
        bin_im_sr[bin_im_sr<=0.40] = 0
        bin_im_sr[bin_im_sr>0.40] = 1
        dif_im_tmp = np.zeros((bin_im_hr.shape[0],bin_im_hr.shape[1],3))
        dif_im_tmp[:,:,0] = hr_imgs[i][0].cpu().detach().numpy() #bin_im_hr.cpu().detach().numpy()
        dif_im_tmp[:,:,1] = sr_imgs[i][0].cpu().detach().numpy() #bin_im_sr.cpu().detach().numpy()
        dif_ims.append(dif_im_tmp)
        thickness_sr.append(ps.filters.local_thickness(bin_im_sr.cpu().detach().numpy(), mode='dt'))

    fig, axs = plt.subplots(batch_size, 8, figsize=(15,12))
    for i in range(batch_size):
        axs[i,0].imshow(lr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,0].axis('off')
        axs[i,1].imshow(hr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,1].axis('off')
        axs[i,2].imshow(thickness_hr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,2].axis('off')
        axs[i,3].hist(thickness_hr[i].ravel(), bins=15, density=True)
        axs[i,4].imshow(sr_imgs[i][0].cpu().detach().numpy(),cmap='gray')
        axs[i,4].axis('off')
        axs[i,5].imshow(thickness_sr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,5].axis('off')
        axs[i,6].hist(thickness_sr[i].ravel(), bins=15, density=True)
        axs[i,7].imshow(dif_ims[i])
        axs[i,7].axis('off')
    
    return fig

def make_thickness_images_dif2(hr_imgs, lr_imgs, sr_imgs):
    batch_size = len(hr_imgs)
    thickness_hr = []
    thick_hist_hr = []
    thickness_sr = []
    thick_hist_sr = []
    dif_ims = torch.zeros(hr_imgs.shape[0],3,hr_imgs.shape[2],hr_imgs.shape[3])

    for i in range(batch_size):
        bin_im_hr = hr_imgs[i][0].detach().clone()
        bin_im_hr[bin_im_hr<=0.40] = 0
        bin_im_hr[bin_im_hr>0.40] = 1
        thick_tmp = ps.filters.local_thickness(bin_im_hr.cpu().detach().numpy(), mode='dt')
        thickness_hr.append(thick_tmp)
        thick_hist_hr.append(ps.metrics.pore_size_distribution(im=thick_tmp))
        bin_im_sr = sr_imgs[i][0].detach().clone()
        bin_im_sr[bin_im_sr<=0.40] = 0
        bin_im_sr[bin_im_sr>0.40] = 1
        dif_ims[i,0,:,:] = bin_im_hr #hr_imgs[i][0].detach()
        dif_ims[i,1,:,:] = bin_im_sr #sr_imgs[i][0].detach()
        thick_tmp = ps.filters.local_thickness(bin_im_sr.cpu().detach().numpy(), mode='dt')
        thickness_sr.append(thick_tmp)
        thick_hist_sr.append(ps.metrics.pore_size_distribution(im=thick_tmp))

    imgs_lr = make_grid(lr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_hr = make_grid(hr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_sr = make_grid(sr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_dif = make_grid(dif_ims, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr, imgs_dif), -1)

    fig, axs = plt.subplots(batch_size, 4, figsize=(16,10))
    for i in range(batch_size):
        axs[i,0].imshow(thickness_hr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,0].axis('off')
        axs[i,1].bar(x=thick_hist_hr[i].LogR, height=thick_hist_hr[i].pdf, width=thick_hist_hr[i].bin_widths, edgecolor='k')
        axs[i,2].imshow(thickness_sr[i], interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,2].axis('off')
        axs[i,3].bar(x=thick_hist_sr[i].LogR, height=thick_hist_sr[i].pdf, width=thick_hist_sr[i].bin_widths, edgecolor='k')
    
    return img_grid, fig

def make_dif_thck_ims_v3(hr_imgs, lr_imgs, sr_imgs):
    batch_size = len(hr_imgs)
    thick_hr = []
    thick_sr = []
    dif_ims = np.zeros((hr_imgs.shape[0],3,hr_imgs.shape[2],hr_imgs.shape[3]))

    for i in range(batch_size):
        bin_im_hr = hr_imgs[i][0].cpu().detach().clone().numpy()
        #_, bin_im_hr = cv.threshold(bin_im_hr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        bin_im_hr = bin_im_hr > threshold_otsu(bin_im_hr)
        thick_tmp = lt.local_thickness(bin_im_hr, scale=0.5)
        thick_hr.append(thick_tmp)
        bin_im_sr = sr_imgs[i][0].cpu().detach().clone().numpy()
        #_, bin_im_sr = cv.threshold(bin_im_sr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        bin_im_sr = bin_im_sr > threshold_otsu(bin_im_sr)
        dif_ims[i,0,:,:] = bin_im_hr 
        dif_ims[i,1,:,:] = bin_im_sr 
        thick_tmp = lt.local_thickness(bin_im_sr, scale=0.5)
        thick_sr.append(thick_tmp)

    imgs_lr = make_grid(lr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_hr = make_grid(hr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_sr = make_grid(sr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_dif = make_grid(torch.from_numpy(dif_ims), nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr, imgs_dif), -1)

    fig, axs = plt.subplots(batch_size, 2, figsize=(6,10))
    for i in range(batch_size):
        axs[i,0].imshow(thick_hr[i], interpolation='none', origin='upper', cmap=lt.black_plasma())
        axs[i,0].axis('off')
        axs[i,1].imshow(thick_sr[i], interpolation='none', origin='upper', cmap=lt.black_plasma())
        axs[i,1].axis('off')
    
    return img_grid, fig

def show_dif_thck_ims_v3(hr_imgs, lr_imgs, sr_imgs):
    batch_size = len(hr_imgs)
    thick_hr = []
    thick_sr = []
    dif_ims = np.zeros((hr_imgs.shape[0],3,hr_imgs.shape[2],hr_imgs.shape[3]))

    for i in range(batch_size):
        bin_im_hr = hr_imgs[i][0].cpu().detach().clone().numpy()
        #_, bin_im_hr = cv.threshold(bin_im_hr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        bin_im_hr = bin_im_hr > threshold_otsu(bin_im_hr)
        thick_tmp = lt.local_thickness(bin_im_hr, scale=0.5)
        thick_hr.append(thick_tmp)
        bin_im_sr = sr_imgs[i][0].cpu().detach().clone().numpy()
        #_, bin_im_sr = cv.threshold(bin_im_sr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        bin_im_sr = bin_im_sr > threshold_otsu(bin_im_sr)
        dif_ims[i,0,:,:] = bin_im_hr 
        dif_ims[i,1,:,:] = bin_im_sr 
        thick_tmp = lt.local_thickness(bin_im_sr, scale=0.5)
        thick_sr.append(thick_tmp)

    imgs_lr = make_grid(lr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_hr = make_grid(hr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_sr = make_grid(sr_imgs[:batch_size].cpu().detach(), nrow=1, normalize=True)
    imgs_dif = make_grid(torch.from_numpy(dif_ims), nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr, imgs_dif), -1)
    plt.imshow(img_grid.transpose(1, 2, 0))
    
    fig, axs = plt.subplots(1, 4, figsize=(20,10))
    axs[0,0].imshow(imgs_hr, cmap='gray')
    axs[0,0].axis('off')
    axs[0,1].imshow(imgs_lr, cmap='gray')
    axs[0,1].axis('off')
    axs[0,2].imshow(imgs_sr, cmap='gray')
    axs[0,2].axis('off')
    axs[0,3].imshow(imgs_dif)
    axs[0,3].axis('off')
    plt.show()

    fig, axs = plt.subplots(batch_size, 2, figsize=(6,10))
    for i in range(batch_size):
        axs[i,0].imshow(thick_hr[i], interpolation='none', origin='upper', cmap=lt.black_plasma())
        axs[i,0].axis('off')
        axs[i,1].imshow(thick_sr[i], interpolation='none', origin='upper', cmap=lt.black_plasma())
        axs[i,1].axis('off')
    plt.show()

def make_images(hr_imgs, lr_imgs, sr_imgs):
    imgs_lr = make_grid(lr_imgs.cpu().detach(), nrow=1, normalize=True)
    imgs_hr = make_grid(hr_imgs.cpu().detach(), nrow=1, normalize=True)
    imgs_sr = make_grid(sr_imgs.cpu().detach(), nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr), -1)
    
    return img_grid

def make_images_from_vol(hr_imgs, lr_imgs, sr_imgs):
    n_slices = hr_imgs.shape[-1]
    middle_slice = int(n_slices/2)
    hr_imgs = hr_imgs[..., middle_slice]
    lr_imgs = lr_imgs[..., middle_slice]
    sr_imgs = sr_imgs[..., middle_slice]
    imgs_lr = make_grid(lr_imgs.cpu().detach(), nrow=1, normalize=True)
    imgs_hr = make_grid(hr_imgs.cpu().detach(), nrow=1, normalize=True)
    imgs_sr = make_grid(sr_imgs.cpu().detach(), nrow=1, normalize=True)
    img_grid = torch.cat((imgs_hr, imgs_lr, imgs_sr), -1)
    
    return img_grid

def make_images2(target, input, output):
    batch_size = len(target)
    fig, axs = plt.subplots(batch_size, 3, figsize=(8,10))
    for i in range(batch_size):
        axs[i,0].imshow(target[i][0].cpu().detach(), interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,0].axis('off')
        axs[i,1].imshow(input[i][0].cpu().detach(), cmap='gray')
        axs[i,1].axis('off')
        axs[i,2].imshow(output[i][0].cpu().detach(), interpolation='none', origin='upper', cmap=plt.cm.jet)
        axs[i,2].axis('off')
    
    return fig

def make_monai_net(layer_specs = (4, 8, 16, 32, 64, 128, 256), r_drop = 0.0, norm_type = 'instance', num_res = 2,kernel=7):
    # dropout rate              [0.0; 0.3]
    # norm                      {'instance','batch'}
    # no. of residual units     {2, 4}
    stride = (2,) * (len(layer_specs)-1)
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=layer_specs,
        kernel_size=kernel,
        up_kernel_size=kernel,
        strides=stride,
        dropout=r_drop,
        norm=norm_type,
        num_res_units=num_res
    )
    return model

def patches2slice(femur_no = "01", femur_type = "HR", slice_no = 800):
    # femur_no.: "01", "15", "21", "74"
    # femur_type: "HR", "LR", "SR", "mask"
    x_dim, y_dim = 128, 128
    dataset_path = "/work3/soeba/HALOS/Data/Images"

    if femur_no == "01":
        block_x = 4
        block_y = 5
    elif femur_no == "15":
        block_x = 4
        block_y = 6
    elif femur_no == "21":
        block_x = 5
        block_y = 6
    elif femur_no == "74":
        block_x = 4
        block_y = 6

    if femur_type == "HR":
        dataset_path += '/micro/'
    elif femur_type == "LR":
        dataset_path += '/clinical/low-res/linear/'
    elif femur_type == "SR":
        dataset_path += '/SR/low-res/linear/'
    elif femur_type == "mask":
        dataset_path += '/masks/'

    bone_slice = np.zeros((block_x*x_dim,block_y*y_dim))

    paths = sorted(glob.glob(dataset_path + femur_no + '?_' + str(slice_no).zfill(4) + "*.*"))
    for i in range(len(paths)):
        patch_no = int(paths[i][-6:-4])
       # print(f'patch no.: {patch_no}')
        patch_x_ind = patch_no // block_y
        patch_y_ind = patch_no % block_y
       # print(f'patch x-ind: {patch_x_ind}, y-ind: {patch_y_ind}')
        im = np.load(paths[i])
       # print(f'patch shape: {im.shape}')
       # print(f'slice: [{patch_x_ind*x_dim} : {(patch_x_ind+1)*x_dim} , {patch_y_ind*y_dim} : {(patch_y_ind+1)*y_dim}]')
        bone_slice[patch_x_ind*x_dim:(patch_x_ind+1)*x_dim,patch_y_ind*y_dim:(patch_y_ind+1)*y_dim] = im

    return bone_slice