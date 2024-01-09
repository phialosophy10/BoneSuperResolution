# %% Packages
import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d
import matplotlib.pyplot as plt
import glob
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# %% Functions
def calculate_angles(vec, class_vectors):

    shape = class_vectors.shape[:1] + vec.shape[1:]

    # Calculate dot product between each class vector and ST vectors.
    vec_dots = np.einsum('ij,oi->oj', vec.reshape(2, -1), class_vectors).reshape(shape)
    np.abs(vec_dots, out=vec_dots)

    # Determine classes.
    vec_class = np.empty(vec_dots.shape[1:], dtype=np.uint8)
    vec_class = np.argmax(vec_dots, axis=0, out=vec_class)

    # Get angle from x-axis.
    cos_theta = vec_dots[0]

    # Calculate theta.
    theta = np.arccos(cos_theta, out=cos_theta)
    theta = np.degrees(theta, out=theta)

    return vec_class, theta

def fig_with_colorbar(d, o, title, alpha=0.5, cmap=None, vmin=None, vmax=None):
    """Creates a figure with data, overlay and color bar."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.imshow(d, cmap='gray')
    if np.issubdtype(o.dtype, np.integer):
        cmap = plt.get_cmap('gist_rainbow', len(class_names))
        im = ax.imshow(o, alpha=alpha, cmap=cmap, vmin=-.5, vmax=len(class_names) - .5)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.arange(len(class_names)))
        cbar.ax.set_yticklabels(class_names)
    else:
        im = ax.imshow(o, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(title)
   # add_scalebar(ax, scale=voxel_size / 1000)
    plt.show()

def show_metrics(data, vec, fiber_threshold=None):
    
  #  for i, (d, v) in enumerate(zip(data_slices, vec_slices)):
    vec_class, theta = calculate_angles(vec, class_vectors)

    if fiber_threshold is not None:
        fiber_mask = data <= fiber_threshold

        vec_class = np.ma.masked_where(fiber_mask, vec_class)          
        theta[fiber_mask] = np.nan

    fig_with_colorbar(data, vec_class, 'Class', alpha=0.7)
   # fig_with_colorbar(data, theta, 'Angle from X (0-90 deg.)', alpha=0.5, vmin=0, vmax=90)

# Names of the four fiber classes.
class_names = ['0', '45', '-45', '90']

# Unit vectors representing each of the four fiber classes.
class_vectors = np.array([[0, 1], [1, 1], [-1, 1], [1, 0]], dtype=np.float64)
class_vectors /= np.linalg.norm(class_vectors, axis=-1)[..., np.newaxis]

# %% Load slice
femur_no = "01"
femur_type = "SR"
slice_no = 250

patch = utils.patches2slice(femur_no = femur_no, femur_type = femur_type, slice_no = slice_no)
mask = utils.patches2slice(femur_no = femur_no, femur_type = "mask", slice_no = slice_no)

im = np.zeros(patch.shape)
im[mask==1] = patch[mask==1]

# %% Show patch
plt.figure()
plt.imshow(im,cmap='gray')
plt.show()

# %% Compute structure tensors
sigma = 3.5      #noise scale: structures smaller than this will be removed by smoothing
rho = 15.5       #integration scale: size of neighborhood to analyze orientation

S = structure_tensor_2d(im, sigma, rho)
val, vec = eig_special_2d(S)

# %% Orientation analysis
show_metrics(im, vec, fiber_threshold=0.35)

# %%
