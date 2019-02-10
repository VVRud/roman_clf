# Dataset for roman digits classification

**You are in _DATASET brench_**

## Introduction

1. [Config](#config)
2. [Data structure](#data-structure)
3. [Data description and cleaning process](#data-description-and-cleaning-process)
4. [Data augmentation and splitting](#data-augmentation-and-splitting)
5. [Script for data preparation](#script-for-data-preparation)
6. [Authors](#authors)

## Config

**_Config files must be located at `configs` folder._**

Parameters specific to data:

- `seed` - seed for random shuffling.
- `image_size` - size of the side for resulting image.
- `number_to_have` - number of images to have in `data_clean/_class_` directory.

## Data structure

`data` folder contains original _'dirty'_ data of roman images, splitted by classes. Number of images in single class folder can be any from 0 to Infinity.  
For example, folder `data/1` contains all images marked as class `1`.

`data_clean` folder contains resized, grayscaled and augmented images, splitted by classes. Number of images in single class folder must be set in config file.  
For example, folder `data_clean/1` contains resized, grayscaled and augmented images, marked as class `1`.

`data_splitted` folder contains data, splitted into train and test datasets. Number of single class in train dataset must be set in config file.  
For example, `data_clean/train/1` contains train images, marked as class `1`.

## Data desciption and cleaning process

It was decided to create about 120 images for each class (roman numbers from 1 to 8) and augment that data to get more variative dataset, so that neural network could be more robust to different variations of data. To achieve a goal of 120 images per class, every team member wrote his variations of roman digits with his/her unique handwriting multiple times. Later there were deleted images, which were wery dirty and neural network was not able to learn specific fetures for them.

Process of data cleaning includes resizing images, grayscaling them and saving to folder `data_clean`.

## Data augmentation and splitting

Data augmentation contains applying multiple transformations to data, such as:

- Cropping;
- Padding;
- Bluring;
- Different affine transformations

Data will be generated into folder `data_clean/_class_` near existing images.

Splitting data contains process of reading filenames of images and resaving them into folder `data_splitted/(train or test)/_class_`.

## Script for data preparation

This script will do all the stuff above fast and clean. To run it follow instructions below.

```sh
cd path/to/project/roman_clf
pip install -r requirements.txt
cd data
python prepare_data.py -c ../configs/roman.json
```

It is easy, isn't it? :)

## Authors

- [Vladyslav Rudenko](https://github.com/vvrud)
- [Vladyslav Zalevskyi](https://github.com/vivikar)
- [Pavlo Pyvovar](https://github.com/pavel-pyvovar)
- [Olga Pashneva](https://github.com/datacat01)
