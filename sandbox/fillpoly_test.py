# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %% Lets first create a contour to use in example
cir = np.zeros((255,255))
cv2.circle(cir,(128,128),10,1)
res = cv2.findContours(cir.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours = res[-2] # for cv2 v3 and v4+ compatibility

# %% An open circle; the points are in contours[0]
plt.figure()
plt.imshow(cir)

# %% Option 1: Using fillPoly
img_pl = np.zeros((255,255))
cv2.fillPoly(img_pl,pts=contours,color=(255,255,255))
plt.figure()
plt.imshow(img_pl)

plt.show()
# %%
