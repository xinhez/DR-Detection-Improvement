
import numpy as np
from PIL import Image

threshold=0.8 # setting a limit to estimate between good and bad images

# function to calculate the amount of overlap between the sample good image
# and the other images in the data set

def intersection(hist1, hist2):
    min = np.minimum(hist1, hist2)
    common = np.true_divide(np.sum(min), np.sum(hist2))
    return common

# Loading the two images to be compared
img = Image.open('data/good.png')
img1 = Image.open('data/bad.png')

# Creating their numpy arrays
arr = np.array(img)
arr1 = np.array(img1)

# Flattening the array
f_img=arr.flatten()
f_img1=arr1.flatten()

# Calculating histograms of the image arrays
hist = np.histogram(f_img,256,[0,256])[0]
hist1= np.histogram(f_img1,256,[0,256])[0]

# Condition to segregate the images
if intersection(hist, hist1)>threshold:
    print "This is a good image"
else:
    print "This is a bad image"



