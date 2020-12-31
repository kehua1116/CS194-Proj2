import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
from skimage import color
import skimage.io as skio
import cv2 
from scipy import signal



def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2



def low_high_frequency(im, sigma1, sigma2):
    gaussian = cv2.getGaussianKernel(8, 2) 
    gaussian_2d = np.outer(gaussian, gaussian.T)
    im_low = sigma1 * (signal.convolve2d(im, gaussian_2d, "same"))
    im_high = sigma2 * (im - im_low)
    return im_low, im_high


def hybrid_image_gray(im1, im2, sigma1, sigma2, save=False):
    im1 = color.rgb2gray(im1)
    im2 = color.rgb2gray(im2)
    im1_low, _ = low_high_frequency(im1, sigma1, sigma2)
    _, im2_high = low_high_frequency(im2, sigma1, sigma2)
    if save:
        skio.imsave("low_frequency.jpg", im1_low)
        skio.imsave("high_frequency.jpg", im2_high)
    im_hybrid = (im1_low + im2_high) / 2
    im_hybrid = np.clip(im_hybrid, 0, 1)
    return im_hybrid


def hybrid_image(im1, im2, sigma1, sigma2):
    im1_low = np.zeros(im1.shape)
    im2_high = np.zeros(im1.shape)
    for i in range(3):      
        im1_low[:,:,i], _ = low_high_frequency(im1[:,:,i], sigma1, sigma2)
        _, im2_high[:,:,i] = low_high_frequency(im2[:,:,i], sigma1, sigma2)
    im_hybrid = (im1_low + im2_high) / 2
    im_hybrid = np.clip(im_hybrid, 0, 1)
    return im_hybrid


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!

    # high sf
    im1 = plt.imread('./DerekPicture.jpg')/255.
    # low sf
    im2 = plt.imread('./nutmeg.jpg')/255

    # Next align images (this code is provided, but may be improved)
    im2_aligned, im1_aligned = align_images(im2, im1)
    sigma1 = 0.4
    sigma2 = 1
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    hybrid_gray = hybrid_image_gray(im1_aligned, im2_aligned, sigma1, sigma2)

    skio.imsave("hybrid_cat_gray.jpg", hybrid_gray)
    skio.imsave("hybrid_cat.jpg", hybrid)

    

    # Create hybrid image for other images
    im1 = plt.imread('./rick1.jpg')/255.
    im2 = plt.imread('./rick2.jpg')/255
    im2_aligned, im1_aligned = align_images(im2, im1)
    sigma1 = 0.6
    sigma2 = 1
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    hybrid_gray = hybrid_image_gray(im1_aligned, im2_aligned, sigma1, sigma2)
    skio.imsave("hybrid_rick_gray.jpg", hybrid_gray)
    skio.imsave("hybrid_rick.jpg", hybrid)


    im1 = plt.imread('./thanos.jpg')/255.
    im2 = plt.imread('./trump.jpg')/255
    im2_aligned, im1_aligned = align_images(im2, im1)
    sigma1 = 0.8
    sigma2 = 1.2
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    hybrid_gray = hybrid_image_gray(im1_aligned, im2_aligned, sigma1, sigma2, True)
    skio.imsave("hybrid_trump_gray.jpg", hybrid_gray)
    skio.imsave("hybrid_trump.jpg", hybrid)


    im1 = plt.imread('./ice_cream.jpg')/255.
    im2 = plt.imread('./cake.jpg')/255
    im2_aligned, im1_aligned = align_images(im2, im1)
    sigma1 = 0.6
    sigma2 = 1
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    hybrid_gray = hybrid_image_gray(im1_aligned, im2_aligned, sigma1, sigma2)
    skio.imsave("hybrid_icecream_gray.jpg", hybrid_gray)
    skio.imsave("hybrid_icecream.jpg", hybrid)
