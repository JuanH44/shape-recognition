# This script is used to extract the Hu moments from the shapes' images and create a csv file with the data, so it can
# be used to train the model.

import cv2
import numpy as np
import pandas as pd
import os

#shapes' names
shapes = ['circle', 'square', 'star', 'triangle']
folder = r'.\archive\shapes\\'


# Receives a path and returns an array of Hu moments
def get_hu_moments(path):
    # Read the image
    img = cv2.imread(path, 0)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(binary_img)
    hu_moments = cv2.HuMoments(moments).flatten().tolist()

    return hu_moments


# Receives a path and the shape's name, takes the first 250 images of the shape and
# returns an array of Hu moments appended with the shape's name
def get_hu_moments_from_path(path, shape):
    hu_moments_list = []
    for i in range(250):
        hu_moments = get_hu_moments(os.path.join(path, shape, str(i) + '.png'))
        hu_moments.append(shape)
        hu_moments_list.append(hu_moments)
    return hu_moments_list


if __name__ == '__main__':
    print("Starting...")

    four_shapes_hu_moments = []

    for shape in shapes:
        four_shapes_hu_moments.extend(get_hu_moments_from_path(folder, shape))

    # Define the columns' names
    columns = ['hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 'shape']

    # Create the dataframe
    df = pd.DataFrame(four_shapes_hu_moments, columns=columns)

    # Save the dataframe to a csv file
    df.to_csv('hu_moments.csv', index=False)
    print("Done!")


