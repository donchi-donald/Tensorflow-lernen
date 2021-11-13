import re #regex
import matplotlib.pyplot as plt
dataset = r'C:\archive\dataset'
val_txt = r'C:\archive\validation.txt'

with open(val_txt) as f:
    data = f.read().split('\n')[:-1]

data_points = {}
for i in data:
    key = i[:-9]
    if key in data_points:
        data_points[key].append(i)
    else:
        data_points[key] = [i]

for i in data_points.keys():
    print(i)

data_size = {key : len(data_points[key]) for key in data_points.keys()}

plt.bar(data_size.keys(), data_size.values())
plt.show()


