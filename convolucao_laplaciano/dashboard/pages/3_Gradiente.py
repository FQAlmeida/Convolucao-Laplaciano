from typing import Literal, Union
import streamlit as st
from convolucao_laplaciano.dashboard.commom import convolution, gauss
from PIL import Image
import numpy as np

st.title("Convolução Laplaciano")

image_lua = Image.open("./data/Lua_clean.jpg")

image_lua = image_lua.convert("L")

st.image(image_lua)

image = Image.open("./data/chessboard_inv.png")

image = image.convert("L")

st.image(image)


@st.cache_data
def conv(image_array: np.ndarray, kernel: np.ndarray):
    return convolution(image_array, kernel)

image_array_lua = np.array(image_lua)
image_array = np.array(image)
kernel_baixa_size = 5
sigma = 3.0
grid_lin_space = np.linspace(
    -((kernel_baixa_size - 1) // 2),
    ((kernel_baixa_size - 1) // 2),
    kernel_baixa_size,
    dtype=np.int64,
)
kernel_x, kernel_y = np.meshgrid(grid_lin_space, grid_lin_space)
kernel_baixa = gauss(kernel_x, kernel_y, sigma)

image_baixa_lua = conv(image_array_lua, kernel_baixa)
image_baixa = conv(image_array, kernel_baixa)


kernel_alta = -1 * np.array(
    [
        [+1, +1, +1],
        [+1, -8, +1],
        [+1, +1, +1],
    ]
)

image_alta_lua = np.add(image_baixa_lua, conv(image_baixa_lua, kernel_alta))
image_alta = np.add(image_baixa, conv(image_baixa, kernel_alta))

kernel_prewitt_size = 3
grid_lin_space = np.linspace(
    -((kernel_prewitt_size - 1) // 2),
    ((kernel_prewitt_size - 1) // 2),
    kernel_prewitt_size,
    dtype=np.int64,
)
kernel_prewitt_x, kernel_prewitt_y = np.meshgrid(grid_lin_space, grid_lin_space)
kernel_sobel_x = np.array(
    [
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1],
    ]
)
kernel_sobel_y = kernel_sobel_x.T

kernel_scharr_x = np.array(
    [
        [-3, 0, +3],
        [-10, 0, +10],
        [-3, 0, +3],
    ]
)
kernel_scharr_y = kernel_scharr_x.T

selected_kernel = st.selectbox("Selecione o Kernel", ("Prewitt", "Sobel", "Scharr"))
if not selected_kernel:
    selected_kernel = "Prewitt"


def get_selected_kernels(
    selected_kernel: Union[Literal["Prewitt"], Literal["Sobel"], Literal["Scharr"]]
):
    if selected_kernel == "Prewitt":
        return (kernel_prewitt_x, kernel_prewitt_y)
    if selected_kernel == "Sobel":
        return (kernel_sobel_x, kernel_sobel_y)
    return (kernel_scharr_x, kernel_scharr_y)


kernel_grad_x, kernel_grad_y = get_selected_kernels(selected_kernel)

g_x = conv(image_alta, kernel_grad_x)
g_y = conv(image_alta, kernel_grad_y)

g_x_lua = conv(image_alta_lua, kernel_grad_x)
g_y_lua = conv(image_alta_lua, kernel_grad_y)

m_lua = np.sqrt(g_x_lua**2, g_y_lua**2)

m = np.sqrt(g_x**2, g_y**2)
erro = 10**-8
d = np.arctan2(g_y, g_x + erro)

d_lua = np.arctan2(g_y_lua, g_x_lua + erro)

image_final_array_m_lua = m_lua

image_final_array_min_m_lua = image_final_array_m_lua - np.min(image_final_array_m_lua)
image_final_array_m_lua = np.rint(
    255 * (image_final_array_min_m_lua / np.max(image_final_array_min_m_lua))
)

image_final_array_m = m

image_final_array_min_m = image_final_array_m - np.min(image_final_array_m)
image_final_array_m = np.rint(
    255 * (image_final_array_min_m / np.max(image_final_array_min_m))
)

image_final_m_lua = Image.fromarray(image_final_array_m_lua)
image_final_m = Image.fromarray(image_final_array_m)

st.image(image_final_m_lua.convert("L"))
st.image(image_final_m.convert("L"))

image_final_array_d_lua = d_lua

image_final_array_min_d_lua = image_final_array_d_lua - np.min(image_final_array_d_lua)
image_final_array_d_lua = np.rint(
    255 * (image_final_array_min_d_lua / np.max(image_final_array_min_d_lua))
)

image_final_d_lua = Image.fromarray(image_final_array_d_lua)

image_final_array_d = d

image_final_array_min_d = image_final_array_d - np.min(image_final_array_d)
image_final_array_d = np.rint(
    255 * (image_final_array_min_d / np.max(image_final_array_min_d))
)

image_final_d = Image.fromarray(image_final_array_d)

st.image(image_final_d_lua.convert("L"))
st.image(image_final_d.convert("L"))

#-------------------------------------------

