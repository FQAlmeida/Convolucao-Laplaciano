import numpy as np
from PIL import Image
from multiprocessing import Pool

image = Image.open("./data/11_test.png")

image = image.convert("L")

# image.show()

kernel = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
)

image_array = np.array(image)
image_arrayb = np.pad(image_array, 1, "constant")

image_array_conv = image_array.copy()

# print(image_array)


def f(i: int):
    pixels = list()
    for j, px in enumerate(image_arrayb[i][1:-1]):
        soma = 0
        for desloc_i in range(3):
            for desloc_j in range(3):
                soma += (
                    kernel[desloc_i][desloc_j]
                    * image_arrayb[i - 1 + desloc_i][j - 1 + desloc_j]
                )
        pixels.append(px + soma)
    return np.array(pixels)


with Pool() as pool:
    result = pool.map(f, range(1, image_arrayb.shape[0] - 1))
    result = np.array(result)
    image_array_conv = result

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_array = image_array + image_array_conv

image_final_array_min = image_final_array - np.min(image_final_array)
image_final_array = np.rint(
    255 * (image_final_array_min / np.max(image_final_array_min))
)

print(np.max(image_final_array))
print(np.min(image_final_array))

image_final = Image.fromarray(image_final_array)

image_final.show()
image.show()
