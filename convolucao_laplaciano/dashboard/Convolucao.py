from functools import partial
from math import e, pi
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_toggle import st_toggle_switch as toggle

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


def f(i: int, padding: int, image_arrayb: np.ndarray, kernel: np.ndarray):
    pixels = list()
    for j in range(padding, len(image_arrayb[i]) - padding):
        soma = np.sum(
            np.multiply(
                kernel,
                image_arrayb[
                    i - padding : i - padding + len(kernel),
                    j - padding : j - padding + len(kernel),
                ],
            )
        )
        pixels.append(soma)
    return np.array(pixels)


@st.cache_data
def conv(image_array: np.ndarray, kernel: np.ndarray):
    padding = (len(kernel) - 1) // 2
    image_arrayb = np.pad(image_array, padding, "edge")
    partial_f = partial(f, padding=padding, image_arrayb=image_arrayb, kernel=kernel)
    results = list()
    with Pool() as pool:
        results = pool.map_async(
            partial_f,
            range(padding, len(image_arrayb) - padding),
        )
        results = results.get()
    return np.array(results)


image_array_conv = conv(image_array, kernel)

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

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
        st.number_input("Sigma", min_value=0.1, max_value=5.0, value=2.0, step=0.01)
    )
    padding = (size - 1) // 2

grid_lin_space = np.linspace(
    -((size - 1) // 2), ((size - 1) // 2), size, dtype=np.int64
)
kernel_x, kernel_y = np.meshgrid(grid_lin_space, grid_lin_space)
kernel = gauss(
    kernel_x,
    kernel_y,
    sigma,
)

with col_1:
    st.write(kernel)

image_array_conv_gauss = conv(image_array, kernel)
image_array_conv_min = image_array_conv_gauss - np.min(image_array_conv_gauss)
image_array_conv_gauss = np.rint(
    255 * (image_array_conv_min / np.max(image_array_conv_min))
)

image_final_conv_lua_gauss = Image.fromarray(image_array_conv_gauss)

st.image(image_final_conv_lua_gauss.convert("L"))

# endregion

image = Image.open("./data/11_test.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image)

st.markdown("#### Kernel")

col_1, col_2 = st.columns(2)

with col_2:
    c = 1 if toggle("C", label_after=True, default_value=True) else -1
    dev_step = toggle("Option", label_after=True, default_value=True)

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
image_array_conv = conv(image_array, kernel)

image_final_array = np.add(image_array, image_array_conv)

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_array_min = image_final_array - np.min(image_final_array)
image_final_array = np.rint(
    255 * (image_final_array_min / np.max(image_final_array_min))
)


image_final_conv = Image.fromarray(image_array_conv)
image_final = Image.fromarray(image_final_array)

st.image(image_final_conv.convert("L"))
st.image(image_final.convert("L"))
