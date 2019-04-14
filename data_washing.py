import numpy as np
import cv2
from zipfile import ZipFile
zipped_images = ZipFile("train_images.zip")

middle_name = ["off_edge", "on_edge", "on_corner"] # [0, 1, 2]

data = []
lens = []
label = []
for i in range(3):
    txt_file_name = "train_images/edge/" + middle_name[i] + ".txt"
    txtstr = zipped_images.read(txt_file_name)
    txtlist = txtstr.split()
    for image_name in txtlist:
        bmpname = "train_images/edge/" + middle_name[i] + "/4/" + str(image_name, encoding="utf-8")
        bmp_byte_str = zipped_images.read(bmpname)
        bmp_byte_array = np.frombuffer(bmp_byte_str, dtype=np.uint8)
        image = cv2.imdecode(bmp_byte_array, cv2.IMREAD_GRAYSCALE)
        data.append(image)
    
    lens.append(len(txtlist))
    label.extend([i] * lens[-1])

data = np.array(data)
data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
label = np.array(label)
label = label.reshape((label.shape[0], 1))
print(lens) # [100390, 110910, 47060]
print(data.shape) # (258360, 25)
print(label.shape) # (258360, 1)
overall_data = np.hstack((data, label))
print(overall_data.shape)
np.save('train_data', overall_data)

