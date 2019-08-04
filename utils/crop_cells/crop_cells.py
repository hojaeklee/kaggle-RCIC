from PIL import Image
import numpy as np
import os
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.measure import label, perimeter
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage.measurements import center_of_mass

import warnings
warnings.filterwarnings("ignore")

def crop_cells(sample_id, image_dir, save_dir, min_nuc_size = 100, cropsize = 32, threshold = 0.5):
    channel_names = ["w" + str(i) for i in range(1,7)]
    channels = {}
    for cn in channel_names:
        channels[cn] = np.array(Image.open(image_dir + sample_id + "_"+ cn +".png"))
        
    nuclei = channels["w1"]
    (width, height) = nuclei.shape
    
    smoothed_nuclei = gaussian(nuclei, sigma=1.0)
    val = threshold_otsu(smoothed_nuclei)

    binary_nuclei = smoothed_nuclei > val
    binary_nuclei = remove_small_holes(binary_nuclei, min_size=100)

    labeled_nuclei = label(binary_nuclei)
    labeled_nuclei = clear_border(labeled_nuclei)
    labeled_nuclei = remove_small_objects(labeled_nuclei, min_size=min_nuc_size)
    
    
    unique, counts = np.unique(labeled_nuclei, return_counts=True)
    background_index = np.argmax(counts)
    counts = counts[unique!=background_index]
    unique = unique[unique!=background_index]
    

    cells_use = unique > -1
    for i in range(unique.size):
        mask = (labeled_nuclei == unique[i])

        y, x = center_of_mass(mask)
        x = np.int(x)
        y = np.int(y)

        c1 = y - cropsize // 2
        c2 = y + cropsize // 2
        c3 = x - cropsize // 2
        c4 = x + cropsize // 2

        if c1 < 0 or c2 >= height or c3 < 0 or c4 >= width:
            cells_use[i] = False

    unique = unique[cells_use]
    counts = counts[cells_use]


    mask = np.isin(labeled_nuclei,unique[counts<=np.quantile(counts,threshold)])
    mask = 255*mask.astype("uint8")
    Image. fromarray(mask)

    cells_use = counts<=np.quantile(counts,threshold)
    unique = unique[cells_use]
    counts = counts[cells_use]

    for i in range(unique.size):
        mask = (labeled_nuclei == unique[i])

        y, x = center_of_mass(mask)
        x = np.int(x)
        y = np.int(y)

        c1 = y - cropsize // 2
        c2 = y + cropsize // 2
        c3 = x - cropsize // 2
        c4 = x + cropsize // 2



        for cn, img in channels.items():
            cropped = img[c1:c2, c3:c4]
            image_name = save_dir + sample_id + "_cid"  + str(i) + "_" + cn  + "_cx" + str(x) + "_cy" + str(y)+ ".png"
            Image.fromarray(cropped).save(image_name)
    return

