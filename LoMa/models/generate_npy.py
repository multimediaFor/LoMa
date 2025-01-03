import os
import numpy as np
import logging as logger
from tqdm import tqdm


def generate_flist(path_input, path_gt, nickname):
    # NOTE: The image and ground-truth should have the same name.
    # Example:
    # path_input = 'tampCOCO/sp_images/'
    # path_gt = 'tampCOCO/sp_masks/'
    # nickname = 'tampCOCO_sp'
    res = []
    flag = False
    image_files = sorted(os.listdir(path_input))
    mask_files = sorted(os.listdir(path_gt))
    assert len(image_files) == len(mask_files), "Images and masks count do not match!"
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        res.append((path_input + image_file, path_gt + mask_file))
    save_name = '%s_%s.npy' % (nickname, len(res))
    np.save('flist/' + save_name, np.array(res))
    if flag:
        logger.info('Note: The following score is meaningless since no ground-truth is provided.')
    return save_name


path_input = "/data/gk/dataset/OSN_dataset/CASIAv1_Facebook/"
path_gt = "/data/gk/dataset/CASIA1/gt/"
nickname = "CASIAv1_Facebook"
generate_flist(path_input, path_gt, nickname)