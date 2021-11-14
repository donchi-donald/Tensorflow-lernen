import os.path
import re #regex
import shutil

import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

dataset = "C:\\archive\dataset\\"
val_txt = r'C:\archive\validation.txt'
train_dir = dataset + "train_data\\"
val_dir = dataset + "val_data\\"
test_dir = dataset + "test_data\\"
val_test_train_percent = 0.15

target_size = (256, 256)
batch_size = 128
class_mode = 'categorical'

with open(val_txt) as f:
    data = f.read().split('\n')[:-1]

data_points = {}
for i in data:
    key = i[:-9]
    if key in data_points:
        data_points[key].append(i)
    else:
        data_points[key] = [i]


data_size = {key : len(data_points[key]) for key in data_points.keys()}



def copy_img(start, end, dir):
    for key in data_points.keys():
        dest = os.path.join(dir, key)
        for i in range(start, end):
            src = dataset + data_points[key][i]
            shutil.copy2(src, dest)

def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    for lego_block in data_points.keys():
        os.makedirs(os.path.join(dir, lego_block))

def copy_images_and_create_directories():
    create_dir(train_dir)
    create_dir(val_dir)
    create_dir(test_dir)

    train_data_amount = int(data_size[list(data_size.keys())[0]]*(1-2*val_test_train_percent))
    val_test_data_amount = int((data_size[list(data_size.keys())[0]] - train_data_amount)/2)
    copy_img(0, train_data_amount, train_dir)
    copy_img(train_data_amount, train_data_amount + val_test_data_amount, test_dir)
    copy_img(train_data_amount + val_test_data_amount,train_data_amount + 2*val_test_data_amount , val_dir)

#copy_images_and_create_directories()

img_generator = ImageDataGenerator(rescale=1./255.,
                   rotation_range=45,
                   width_shift_range=0.2,
                   height_shift_range=0.2,
                   zoom_range=0.2,
                   horizontal_flip=True,
                   vertical_flip=True,
                   fill_mode='nearest')

train_data_gen = img_generator.flow_from_directory(train_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode=class_mode)

val_data_gen = img_generator.flow_from_directory(val_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 class_mode=class_mode)

test_data_gen = ImageDataGenerator(rescale=1./255.).flow_from_directory(test_dir,
                                                                        target_size=target_size,
                                                                        batch_size=batch_size,
                                                                        class_mode=class_mode)

print(train_data_gen)
print(val_data_gen)
print(test_data_gen)