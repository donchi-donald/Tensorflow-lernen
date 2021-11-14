import re #regex
import shutil

import matplotlib.pyplot as plt
dataset = "C:\\archive\dataset\\"
val_txt = r'C:\archive\validation.txt'
train_dir = dataset + "train_data\\"
val_dir = dataset + "val_data\\"
test_dir = dataset + "test_data\\"
val_test_train_percent = 0.15


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
        for i in range(start, end):
            src = dataset + data_points[key][i]
            shutil.copy2(src, dir)

train_data_amount = int(data_size[list(data_size.keys())[0]]*(1-2*val_test_train_percent))
val_test_data_amount = int((data_size[list(data_size.keys())[0]] - train_data_amount)/2)
copy_img(0, train_data_amount, train_dir)
copy_img(train_data_amount, train_data_amount + val_test_data_amount, test_dir)
copy_img(train_data_amount + val_test_data_amount,train_data_amount + 2*val_test_data_amount , val_dir)
print(train_data_amount)
print(val_test_data_amount)


