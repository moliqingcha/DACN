import scipy
import scipy.misc
import numpy as np

def imsave(image, path):
    

    label_colours = [(0,0,0),(255,255,255)]
    
    images = np.ones(list(image.shape)+[3])
    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 2:
                images[j_, k_] = label_colours[int(k)]
    scipy.misc.imsave(path, images)


