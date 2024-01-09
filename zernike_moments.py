# This script is used to extract the Zernike moments from the shapes' images and create a csv file with the data,
# so it can be used to train the model.
# four shapes dataset
import cv2
import numpy as np
import pandas as pd
import os

import mahotas as mt

radio = 65


def get_zernike_moments(path):
    # Read the image
    img = cv2.imread(path, 0)
    _,  binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    moments = mt.features.zernike_moments(binary_img, radius=radio, degree=3)
    moments = moments.flatten().tolist()

    return moments


def get_zernike_moments_from_path(path, shape):
    zernike_moments_list = []
    for i in range(250):
        zernike_moments = get_zernike_moments(os.path.join(path, shape, str(i) + '.png'))
        zernike_moments.append(shape)
        zernike_moments_list.append(zernike_moments)
    return zernike_moments_list


if __name__ == '__main__':
    #
    # print(get_zernike_moments(r'.\archive\shapes\circle\0.png'))


    four_shapes_zernike_moments = []

    shapes = ['circle', 'square', 'star', 'triangle']
    folder = r'.\archive\shapes\\'

    for shape in shapes:
        four_shapes_zernike_moments.extend(get_zernike_moments_from_path(folder, shape))

    # Define the columns' names
    # zn = zernike moments, n = (degree + 1) * (degree + 2) / 2 = 5 * 6 / 2 = 15
    columns = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'shape']

    # Create the dataframe
    df = pd.DataFrame(four_shapes_zernike_moments, columns=columns)

    # Save the dataframe to a csv file
    df.to_csv('zernike_moments_2.csv', index=False)

    print("Done!")



