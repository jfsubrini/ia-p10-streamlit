# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member
"""
Created by Jean-François Subrini on the 26th of May 2023.
Utils functions and variables for making semantic segmentation mask,
using a HRNetV2 + OCR model (created in the Notebook 2 Scripts).
"""
### IMPORTS ###
import cv2
import numpy as np
from tensorflow.keras.models import load_model



# List of images and masks.
NAME_LIST = ["frankfurt_1", "frankfurt_2", "frankfurt_3", "lindau_1",
            "lindau_2", "malaga", "munich_1", "munster_1", "munster_2"
            ]
IMG_LIST = ["frankfurt_000000_000294_leftImg8bit.png",
            "frankfurt_000000_014480_leftImg8bit.png",
            "frankfurt_000001_004859_leftImg8bit.png",
            "lindau_000001_000019_leftImg8bit.png",
            "lindau_000037_000019_leftImg8bit.png",
            "Malaga.png",
            "munich_000073_000019_leftImg8bit.png",
            "munster_000026_000019_leftImg8bit.png",
            "munster_000172_000019_leftImg8bit.png"
            ]
MSK_LIST = ["frankfurt_000000_000294_gtFine_color.png",
            "frankfurt_000000_014480_gtFine_color.png",
            "frankfurt_000001_004859_gtFine_color.png",
            "lindau_000001_000019_gtFine_color.png",
            "lindau_000037_000019_gtFine_color.png",
            "Malaga.png",
            "munich_000073_000019_gtFine_color.png",
            "munster_000026_000019_gtFine_color.png",
            "munster_000172_000019_gtFine_color.png"
            ]

### LOADING MODEL & PREDICTION FUNCTIONS ###
# Loading the selected UNET model.
unet_model = load_model('model_unet', compile=False)  # TODO change the model
# Image size.
IMG_HEIGHT = 192
IMG_WIDTH = 384

def get_colored(pred_mask, n_classes):
    """Function to get the prediction mask colored as wanted
    for the different 8 classes."""
    # This color map has been used to display the results in the predicted mask.
    color_map = {
        0: (0, 0, 0),        # void (black)
        1: (160, 120, 50),   # flat (brown)
        2: (255, 200, 200),  # construction (pink)
        3: (255, 255, 120),  # object (yellow)
        4: (0, 150, 40),     # nature (green)
        5: (0, 180, 230),    # sky (sky blue)
        6: (255, 80, 80),    # human (red)
        7: (90, 40, 210)     # vehicule (blue purple)
    }
    mask_height = pred_mask.shape[0]
    mask_width = pred_mask.shape[1]
    pred_mask_c = np.zeros((mask_height, mask_width, 3))

    for cls in range(n_classes):
        pred_mask_rgb = pred_mask[:, :] == cls
        pred_mask_c[:, :, 0] += ((pred_mask_rgb) * (color_map[cls][0])).astype('uint8') # R
        pred_mask_c[:, :, 1] += ((pred_mask_rgb) * (color_map[cls][1])).astype('uint8') # G
        pred_mask_c[:, :, 2] += ((pred_mask_rgb) * (color_map[cls][2])).astype('uint8') # B

    return pred_mask_c.astype('uint8')

def mask_prediction(model, img_to_predict):
    """Function that makes, with a model, the mask prediction of an image.
    """
    img_resized = cv2.resize(img_to_predict, (IMG_WIDTH, IMG_HEIGHT))
    pred_mask = model.predict(np.expand_dims(img_resized, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = np.squeeze(pred_mask)
    pred_mask_colored = get_colored(pred_mask, 8)

    return pred_mask_colored
