from PIL import Image
import numpy as np
import random

for i in range(120):
    RED = random.randint(0, 255)
    GREEN = random.randint(0, 255)
    BLUE = random.randint(0, 255)

    pixel_touple = (RED, GREEN, BLUE)
    row = []
    total_array = []
    for i in range(11):
        row.append(pixel_touple)

    for j in range(11):
        total_array.append(row)

    nparray = np.array(total_array, dtype=np.uint8)

    red_string = hex(RED).split('x')[1].upper() if len(hex(RED).split('x')[1].upper()) == 2 else f'0{hex(RED).split('x')[1].upper()}'
    green_string = hex(GREEN).split('x')[1].upper() if len(hex(GREEN).split('x')[1].upper()) == 2 else f'0{hex(GREEN).split('x')[1].upper()}'
    blue_string = hex(BLUE).split('x')[1].upper() if len(hex(BLUE).split('x')[1].upper()) == 2 else f'0{hex(BLUE).split('x')[1].upper()}'

    new_image = Image.fromarray(nparray)
    file_string = f'#{red_string}{green_string}{blue_string}'
    new_image.save(f'C:/Users/alebr/Documents/PixelPictures/testpics/{file_string}.png')
