import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import util as util
import time

model_path = '79_130k.pth'
test_img_folder = 'LR'

model = arch.SRResNet(3, 3, 64, 16, upscale=4, norm_type=None, act_type='relu', \
            mode='CNA', res_scale=1, upsample_mode='pixelshuffle')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('testing...')

total_time = 0
idx = 0
for path in glob.glob(test_img_folder + '/*'):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.cuda()

    start = time.time()
    with torch.no_grad():
        output = model(img_LR).data
    total_time += time.time() - start
    output = util.tensor2img_np(output.squeeze())
    util.save_img_np(output, os.path.join('results', base + '.png'))

print('Time for each image: {:.2e}s'.format(total_time/100))