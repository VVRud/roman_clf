import os
import sys

sys.path.extend(['..'])

import random
from PIL import Image
from tqdm import tqdm
import numpy as np

import cv2
from imgaug import augmenters as iaa
import imgaug as ia

from utils.utils import get_args
from utils.config import process_config
from utils.dirs import create_dirs
from shutil import rmtree, copyfile


# Augmentation variables
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
aug_list = [
    sometimes(
        iaa.OneOf([
            iaa.Crop(px=(
                0, 5 #Crop away pixels at the sides of the image.
            )), 
            iaa.CropAndPad(percent=(-0.05, 0.05)),
            #Crop or pad each side(negative values result in cropping, positive in padding).
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            #Distort images locally by moving points around.
            iaa.Affine(
            #Affine transformations on images.
                scale={
                    "x": (0.8, 1.0),
                    "y": (0.8, 1.0)
                },
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1)
                },
                shear=(-8, 8),
                order=[0, 1],
                cval=255,
                mode=ia.ALL)
        ])),
    sometimes(
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 0.1)),
            #Blur each image with a gaussian kernel.
            iaa.AverageBlur(k=(2, 3)),
            #Blur each image using a mean over neihbourhoods.
            iaa.MedianBlur(k=(1, 3)),
            #Blur each image using a median over neihbourhoods.
            iaa.Dropout(p=(0, 0.005)),
            #Set a certain fraction of pixels in images to zero.
            iaa.Add((-5, 5)),
            #Add a value to all pixels in an image.
            iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.005),
            #Distort image locally by moving individual pixels around. 
            iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255))
            #Add gaussian noise.
        ])),
    iaa.ContrastNormalization((1.0, 2.0)),
    #Change and normalize the contrast of image.
    iaa.OneOf([
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        #Sharpen an image, then overlay the results with the original. 
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        #Emboss an image, then overlay the results with the original. 
    ]),
]


def resize_grey_and_save(filename, output_dir, size):
    """
    Resize one image, turn it gray and save.

    :param filename: Path to load image from. Can be relative or absolute.
    :param output_dir: Path to save file in. Can be relative or absolute.
    :param size: Size of the side(images are resized to be a square) for resulting image.

    :return None
    """

    image = Image.open(filename)
    # Converting to black and white data
    image = image.convert('L')
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)

    # Adding 2nd and 3rd channels
    array = np.asarray(image)
    array = np.stack((array, ) * 3, axis=-1)
    # Back to image
    image = Image.fromarray(array, )
    image.save(output_dir)


def clean_class(class_path, output_path, image_size):
    """
    An alias for 'resize_grey_and_save' to operate with full class folder.

    :param class_path: Path to the folder cntaining class images to process.
    :param output_path: Path to the folder to save processed images.
    :param image_size: Size of the side(images are resized to be a square) for resulting image.

    :return None
    """
    
    for img in tqdm(os.listdir(class_path), desc='Resizing'):
        f_name = os.path.join(class_path, img)
        out_name = os.path.join(output_path, img)
        resize_grey_and_save(f_name, out_name, image_size)


def augment_class(directory, cl, number_to_have):
    """
    Augment images of one class.

    :param directory: Path to the directory with class images.
    :param cl: Class name.
    :param number_to_have: Number of images to have in resuzlting directory.

    :return None
    """

    # Add transformations specific for classes 1, 2, 3 and 5
    transformations = aug_list.copy()
    if cl in ['1', '2', '3', '5']:
        transformations.append(iaa.Fliplr(0.5))
    if cl in ['1', '2', '3']:
        transformations.append(iaa.Flipud(0.2))

    # Build augmentation model
    seq = iaa.Sequential(transformations, random_order=True)
    
    # Read filenames from directory
    files = os.listdir(directory)
    # Set nex index to write image
    indicies = [int(name[:-4]) for name in files]
    indicies.sort()
    index = indicies[-1] + 1
    
    # Read images, calculate necessary steps 
    images = np.array([cv2.imread(os.path.join(directory, fname)) for fname in files])
    steps = int(np.ceil((number_to_have - len(files)) / len(files)))
    t = tqdm(range(steps))
    for step in t:
        # If last step
        if step + 1 == steps:
            # Shuffle data
            ind = list(range(len(files)))
            np.random.shuffle(ind)
            to_get = number_to_have - (step * len(files) + len(files))
            # Get random images to augment
            images = images[ind[:to_get]]
        
        # Augment images
        t.set_description('Augmenting...')
        images_aug = seq.augment_images(images)
        
        # Save images
        for image in images_aug:
            cv2.imwrite(os.path.join(directory, str(index) + '.jpg'), image)
            index += 1
    t.close()


def tt_split_class(directory, train_dir, test_dir, train_percentage):
    """
    Split images into train and test folders.

    :param directory: Path to directory to load images from.
    :param train_dir: Path to directory to save train images in.
    :param test_dir: Path to directory to save test images in.
    :param train_percentage: Percentage of train images to have.

    :return None
    """
    
    create_dirs([train_dir, test_dir])
    # Read filenames of images to split into train and test.
    filenames = [pict for pict in os.listdir(directory)
                if pict.endswith('.jpg')]

    # Shuffle names and split into train and test
    np.random.shuffle(filenames)
    split = int(len(filenames) * train_percentage)
    train = filenames[:split]
    test  = filenames[split:]

    # Save images
    for t_set, path_new in [(train, train_dir), (test, test_dir)]:
        for fname in t_set:
            copyfile(os.path.join(directory, fname), os.path.join(path_new, fname))


def main(config):
    # Set seeds for being able to have same results
    np.random.seed(config.seed if config.seed else 2019)
    random.seed(config.seed if config.seed else 2019)

    # Create variables for data paths
    original_data = os.path.join('..', 'data')
    clean_data = os.path.join('..' ,'data_clean')
    splitted_data = os.path.join('..', 'data_splitted')

    # Get filenames in directory with original data
    classes = [d for d in os.listdir(original_data)
                if os.path.isdir(os.path.join(original_data, d)) and not str.startswith(d, '.ipynb')]
    classes.sort()

    # Remove data if it was created before
    if os.path.isdir(clean_data) and os.path.isdir(splitted_data):
        print('\nWas found existing directory with clean and splitted data. Aaaaand...')
        print('\tUnfortunately, they was removed... \n\t\tcompletely removed...\n')
        rmtree(clean_data)
        rmtree(splitted_data)


    create_dirs([original_data, clean_data, splitted_data])
    #Iterate over classes
    t = tqdm(classes)    
    for cl in t:
        t.set_description('CLASS {}\t'.format(cl))
        class_path = os.path.join(original_data, cl)
        output_path = os.path.join(clean_data, cl)
        create_dirs([output_path])

        # Clean and augment data
        clean_class(class_path, output_path, config.image_size)
        augment_class(output_path, cl, config.number_to_have)

        # Split data
        train_path = os.path.join(splitted_data, 'train', cl)
        test_path = os.path.join(splitted_data, 'test', cl)
        tt_split_class(output_path, train_path, test_path, config.train_percentage)

    t.close()
    print('\n\nDone cleaning data')
    return 0 

if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print(e.with_traceback())
