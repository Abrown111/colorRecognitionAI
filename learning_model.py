import tensorflow as tf
from PIL import Image
import os
import numpy as np
import csv
import random
import time


directory = 'C:/Users/alebr/Documents/PixelPictures'
dir = 'C:/Users/alebr/Documents/testpics'
list_colors = []
pixel_array = []
test_array = []
names_array = []

image = Image.open("C:/Users/alebr/Documents/testpics/#1BDEFA.png")
pixels = list(image.getdata())
width, height = image.size
x_test = np.array([pixels[i * width:(i + 1) * width] for i in range(height)])

for filename in os.listdir(directory):

    list_colors.append(filename)
    im = Image.open(f'C:/Users/alebr/Documents/PixelPictures/{filename}')
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    pixel_array.append(pixels)
    im.close()

for filename in os.listdir(dir):

    print(filename)
    names_array.append(filename)
    im = Image.open(f'C:/Users/alebr/Documents/testpics/{filename}')
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    test_array.append(pixels)
    im.close()

color_vals = []

with open("C:/Users/alebr/Downloads/images_sorted.csv", 'r') as f:
    reader = csv.reader(f)
    for row in f:
        color_vals.append(int(row.split(",")[1]) if int(row.split(",")[1]) != 9 else 7)


x_train = np.array(pixel_array)
y_train = np.array(color_vals)
x_test = np.array(test_array)

# mnist = tf.keras.datasets.mnist
# (x_trains, s), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(240, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(120, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6)

predictions = model.predict(x_test)


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


for i in range(50):
    # pic = input("Test color: ")
    color_array_system = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Grey', 'Brown']
    num = random.randint(0, 119)# names_array.index(pic)
    image_name = names_array[num]
    ima = Image.open(f'C:/Users/alebr/Documents/testpics/{image_name}')
    print(color_array_system[np.argmax(predictions[num])-1], np.argmax(predictions[num]))
    ima = ima.resize((500, 500), Image.Resampling.LANCZOS)
    ima.show()
    time.sleep(3)
