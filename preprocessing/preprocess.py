import cv2
import numpy as np

def rgbToGray(observation):
    weights = [0.2989, 0.5870, 0.1140]
    grayscale_observation = np.dot(observation[...,:3], weights)
    return grayscale_observation

def resize(observation, new_width, new_height):
    downsized_observation = cv2.resize(observation, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return downsized_observation

def crop_bottom(observation, crop_height):
    """
    Rognage de la partie basse de l'image en coupant crop_height pixels.
    Parameters:
    observation: np.array
        L'observation (image) sous forme de tableau NumPy (RGB ou niveaux de gris).
    crop_height: int
        Le nombre de pixels à enlever à partir du bas de l'image.
    Returns:
    cropped_observation: np.array
        L'image rognée.
    """
    # Rognage en retirant crop_height pixels du bas
    if crop_height < observation.shape[0]:
        cropped_observation = observation[:-crop_height, :, :]  # On garde toutes les colonnes, mais on enlève crop_height lignes
    else:
        raise ValueError("Le crop_height est plus grand que la hauteur de l'image.")
    
    return cropped_observation

import numpy as np

def crop_bottom_grayscale(observation, crop_height):
    """
    Crop the bottom part of a grayscale image by removing crop_height pixels.
    
    Parameters:
    observation: np.array
        The grayscale observation (2D image) as a NumPy array, shape (height, width).
    crop_height: int
        The number of pixels to remove from the bottom of the image.
        
    Returns:
    cropped_observation: np.array
        The cropped grayscale image.
    """
    # Ensure crop_height is valid
    if crop_height < observation.shape[0]:
        # Crop by removing crop_height pixels from the bottom
        cropped_observation = observation[:-crop_height, :]  # Keep all columns, remove last crop_height rows
    else:
        raise ValueError("crop_height is larger than the image height.")
    
    return cropped_observation

def preprocess_observation(observation, new_width, new_height):
    gray_obs = rgbToGray(observation)
    downsized_obs = resize(gray_obs, new_width=new_width, new_height=new_height)
    return downsized_obs
