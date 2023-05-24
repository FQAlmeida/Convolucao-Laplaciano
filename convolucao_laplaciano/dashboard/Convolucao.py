import streamlit as st
from streamlit_toggle import st_toggle_switch as toggle
import numpy as np
from PIL import Image
from multiprocessing import Pool
from math import pi, e

st.title("Convolução Laplaciano")


image = Image.open("./data/Lua1_gray.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image)

st.markdown("#### Kernel")

col_1, col_2 = st.columns(2)

with col_2:
    size = int(
        st.number_input("Tamanho Kernel", min_value=3, max_value=None, value=3, step=2)
    )
    padding = (size - 1) // 2
    is_const = toggle("Constante", label_after=True)

divisor = 9 if is_const else (size**2)
kernel = np.ones((size, size)) / divisor

with col_1:
    st.write(kernel)


image_array = np.array(image)
image_arrayb = np.pad(image_array, padding, "edge")

image_array_conv = image_array.copy()


def f(i: int, padding: int, image_arrayb: np.ndarray, kernel: np.ndarray):
    pixels = list()
    for j in range(padding, len(image_arrayb[i]) - padding):
        soma = 0
        for desloc_i in range(len(kernel)):
            for desloc_j in range(len(kernel[desloc_i])):
                soma += (
                    kernel[desloc_i][desloc_j]
                    * image_arrayb[i - padding + desloc_i][j - padding + desloc_j]
                    # 466 - 1 + 4
                )
        pixels.append(soma)
    return np.array(pixels)


prog_bar_lua = st.progress(0, text="Conv Lua")
image_array_conv = list()
for i in range(padding, len(image_arrayb) - padding):
    # 464 + 2
    prog_bar_lua.progress(((i - padding) / (len(image_arrayb) - padding)))
    result = f(i, padding, image_arrayb, kernel)
    image_array_conv.append(result)
image_array_conv = np.array(image_array_conv)

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

# st.table(image_array_conv)

image_final_conv = Image.fromarray(image_array_conv)

st.image(image_final_conv.convert("L"))


# region gaussiano


def gauss(x, y, sigma: float):
    return 1 / (2 * pi * sigma**2) * e ** (-(x**2 + y**2) / (2 * sigma**2))


col_1, col_2 = st.columns(2)

with col_2:
    size = int(
        st.number_input(
            "Tamanho Kernel Gauss", min_value=3, max_value=None, value=5, step=2
        )
    )
    sigma = float(
        st.number_input("Alpha", min_value=0.1, max_value=2.0, value=1.0, step=0.01)
    )
    padding = (size - 1) // 2

grid_lin_space = np.linspace(
    -((size - 1) // 2), ((size - 1) // 2), size, dtype=np.int64
)
# st.write(grid_lin_space)
kernel_x, kernel_y = np.meshgrid(grid_lin_space, grid_lin_space)
# st.table(kernel_y)


kernel = gauss(
    kernel_x,
    kernel_y,
    sigma,
)

with col_1:
    st.write(kernel)

image_array_conv_gauss = image_array.copy()


prog_bar_lua_gauss = st.progress(0, text="Conv Lua Gauss")
image_array_conv_gauss = list()
for i in range(padding, len(image_arrayb) - padding):
    # 464 + 2
    prog_bar_lua_gauss.progress(((i - padding) / (len(image_arrayb) - padding)))
    result = f(i, padding, image_arrayb, kernel)
    result = np.array(result)
    image_array_conv_gauss.append(result)
image_array_conv_gauss = np.array(image_array_conv_gauss)

image_array_conv_min = image_array_conv_gauss - np.min(image_array_conv_gauss)
image_array_conv_gauss = np.rint(
    255 * (image_array_conv_min / np.max(image_array_conv_min))
)

# st.table(image_array_conv)

image_final_conv = Image.fromarray(image_array_conv_gauss)

st.image(image_final_conv.convert("L"))

# endregion

image = Image.open("./data/11_test.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image)

st.markdown("#### Kernel")

col_1, col_2 = st.columns(2)

with col_2:
    c = 1 if toggle("C", label_after=True) else -1
    st.write(f"C: {c}")
    dev_step = toggle("Option", label_after=True)

kernel_4 = [
    [+0, +1, +0],
    [+1, -4, +1],
    [+0, +1, +0],
]

kernel_8 = [
    [+1, +1, +1],
    [+1, -8, +1],
    [+1, +1, +1],
]

kernel = c * np.array(kernel_4 if dev_step else kernel_8)

with col_1:
    st.write(kernel)


image_array = np.array(image)
image_arrayb = np.pad(image_array, 1, "constant")

image_array_conv = image_array.copy()

prog_bar_eye = st.progress(0, "Conv Eye")
image_array_conv = list()
for i, _ in enumerate(image_arrayb[1:-1]):
    prog_bar_eye.progress(i / len(image_arrayb[1:-1]))
    result = f(i + 1, 1, image_arrayb, kernel)
    result = np.array(result)
    image_array_conv.append(result)
image_array_conv = np.array(image_array_conv)

image_final_array = image_array + image_array_conv

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_array_min = image_final_array - np.min(image_final_array)
image_final_array = np.rint(
    255 * (image_final_array_min / np.max(image_final_array_min))
)

# st.write(np.max(image_final_array))
# st.write(np.min(image_final_array))

image_final_conv = Image.fromarray(image_array_conv)
image_final = Image.fromarray(image_final_array)

st.image(image_final_conv.convert("L"))
st.image(image_final.convert("L"))

# image_final.show()
# image.show()

# region lua laplace

# TODO: Use a better image, without salt and pepper
image = Image.open("./data/Lua1_gray.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image)

st.markdown("#### Kernel")

col_1, col_2 = st.columns(2)

with col_2:
    c = 1 if toggle("C Lua", label_after=True) else -1
    st.write(f"C: {c}")
    dev_step = toggle("Option Lua", label_after=True)

kernel_4 = [
    [+0, +1, +0],
    [+1, -4, +1],
    [+0, +1, +0],
]

kernel_8 = [
    [+1, +1, +1],
    [+1, -8, +1],
    [+1, +1, +1],
]

kernel = c * np.array(kernel_4 if dev_step else kernel_8)

with col_1:
    st.write(kernel)


image_array = np.array(image)
image_arrayb = np.pad(image_array, 1, "constant")

image_array_conv = image_array.copy()

prog_bar_eye = st.progress(0, "Conv Lua Lap")
image_array_conv = list()
for i, _ in enumerate(image_arrayb[1:-1]):
    prog_bar_eye.progress(i / len(image_arrayb[1:-1]))
    result = f(i + 1, 1, image_arrayb, kernel)
    result = np.array(result)
    image_array_conv.append(result)
image_array_conv = np.array(image_array_conv)

image_final_array = image_array + image_array_conv

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_array_min = image_final_array - np.min(image_final_array)
image_final_array = np.rint(
    255 * (image_final_array_min / np.max(image_final_array_min))
)

# st.write(np.max(image_final_array))
# st.write(np.min(image_final_array))

image_final_conv = Image.fromarray(image_array_conv)
image_final = Image.fromarray(image_final_array)

st.image(image_final_conv.convert("L"))
st.image(image_final.convert("L"))

# endregion
