import os
import sys

sys.path.extend(['..'])

import random
from PIL import Image
from tqdm import tqdm
import numpy as np

from utils.utils import get_args
from utils.config import process_config
from utils.dirs import create_dirs


def resize_grey_and_save(filename, output_dir, size):
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
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


def tt_split_class(directory, train_dir, test_dir, config):
    filenames = [os.path.join(directory, pict) for pict in os.listdir(directory) if pict.endswith('.jpg')]

    filenames.sort()
    random.shuffle(filenames)

    split = int(len(filenames) * config.train_percentage)
    train = filenames[:split]
    test  = filenames[split:]

    for data, path in [(train, train_dir), (test, test_dir)]:
        for f in tqdm(data):
            resize_grey_and_save(f, path, config.image_size)


def main(config):
    random.seed(config.seed if config.seed else 2019)

    original_data = os.path.join('..', 'data')
    train_data_dir = os.path.join('data_clean', 'train')
    test_data_dir = os.path.join('data_clean', 'test')

    print('Train/Test splitting started')
    # Get the filenames in each directory (train and test)
    classes = [d for d in os.listdir(original_data) if os.path.isdir(os.path.join(original_data, d))]
    classes.sort()
    t = tqdm(classes)
    for cl in t:
        t.set_description('CLASS: {}'.format(cl))

        tr_dir = os.path.join(train_data_dir, cl)
        ts_dir = os.path.join(test_data_dir, cl)
        create_dirs([tr_dir, ts_dir])

        tt_split_class(os.path.join(original_data, cl), tr_dir, ts_dir, config)

    print('\n\nDone building dataset')

if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print(e.with_traceback())
