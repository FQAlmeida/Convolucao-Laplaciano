from math import pi
from typing import Literal, Union

import numpy as np
from PIL import Image

from convolucao_laplaciano.dashboard.commom import convolution, gauss

image = Image.open("./data/Lua1_gray.png")

image = image.convert("L")

image.show()

size = 5
padding = (size - 1) // 2
is_const = False

divisor = 9 if is_const else (size**2)
kernel = np.ones((size, size)) / divisor

print(kernel)

image_array = np.array(image)


def conv(image_array: np.ndarray, kernel: np.ndarray):
    return convolution(image_array, kernel)


image_array_conv = conv(image_array, kernel)

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_conv = Image.fromarray(image_array_conv)

# region gaussiano
size = 5
sigma = 1.0
padding = (size - 1) // 2

grid_lin_space = np.linspace(
    -((size - 1) // 2), ((size - 1) // 2), size, dtype=np.int64
)
kernel_x, kernel_y = np.meshgrid(grid_lin_space, grid_lin_space)
kernel = gauss(kernel_x, kernel_y, sigma)

print(kernel)

image_array_conv_gauss = conv(image_array, kernel)
image_array_conv_min = image_array_conv_gauss - np.min(image_array_conv_gauss)
image_array_conv_gauss = np.rint(
    255 * (image_array_conv_min / np.max(image_array_conv_min))
)

image_final_conv_lua_gauss = Image.fromarray(image_array_conv_gauss)

print("Filtro de Média")
image_final_conv.convert("L").show()
print("Filtro Gaussiano")
image_final_conv_lua_gauss.convert("L").show()

# endregion

image = Image.open("./data/11_test.png")

image = image.convert("L")

image.show()

c = -1
dev_step = False

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

print(kernel)

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

image_final_conv.convert("L").show()
image_final.convert("L").show()

image_lua = Image.open("./data/Lua1_gray.png")

image_lua = image_lua.convert("L")


image_lua.show()

image = Image.open("./data/chessboard_inv.png")

image = image.convert("L")

image.show()

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

selected_kernel = "Prewitt"
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

image_final_m_lua.convert("L").show()
image_final_m.convert("L").show()

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


image_final_d_lua.convert("L").show()
image_final_d.convert("L").show()

# -------------------------------------------

matriz_selecao_lua = np.zeros(image_final_array_m_lua.shape)
matriz_selecao = np.zeros(image_final_array_m.shape)


def is_max_local(m: np.ndarray, d: np.ndarray, i: int, j: int):
    n1, n2 = get_colinear_neighbors(m, d, i, j)
    if max(m[i][j], n1, n2) == m[i][j]:
        return True
    return False


# 22.5o => pi / 8
# 45 => pi / 4
def get_colinear_neighbors(m: np.ndarray, d: np.ndarray, i: int, j: int):
    if pi / 8 < d[i][j] and d[i][j] <= 3 * (pi / 8):
        return (
            m[(i + 1) % m.shape[0]][(j - 1) % m.shape[1]],
            m[(i - 1) % m.shape[0]][(j + 1) % m.shape[1]],
        )
    if 3 * (pi / 8) < d[i][j] and d[i][j] <= 5 * (pi / 8):
        return m[(i + 1) % m.shape[0]][j], m[(i - 1) % m.shape[0]][j]
    if 5 * (pi / 8) < d[i][j] and d[i][j] <= 7 * (pi / 8):
        return (
            m[(i + 1) % m.shape[0]][(j + 1) % m.shape[1]],
            m[(i - 1) % m.shape[0]][(j - 1) % m.shape[1]],
        )
    if 7 * (pi / 8) < d[i][j] and d[i][j] <= pi:
        return m[i][(j + 1) % m.shape[1]], m[i][(j - 1) % m.shape[1]]
    if -(pi / 8) < d[i][j] and d[i][j] <= pi / 8:
        return m[i][(j - 1) % m.shape[1]], m[i][(j + 1) % m.shape[1]]
    if -(3 * (pi / 8)) < d[i][j] and d[i][j] <= -(pi / 8):
        return (
            m[(i - 1) % m.shape[0]][(j - 1) % m.shape[1]],
            m[(i + 1) % m.shape[0]][(j + 1) % m.shape[1]],
        )
    if -(5 * (pi / 8)) < d[i][j] and d[i][j] <= -(3 * (pi / 8)):
        return m[(i - 1) % m.shape[0]][j], m[(i + 1) % m.shape[0]][j]
    if -(7 * (pi / 8)) < d[i][j] and d[i][j] <= -(5 * (pi / 8)):
        return (
            m[(i - 1) % m.shape[0]][(j + 1) % m.shape[1]],
            m[(i + 1) % m.shape[0]][(j - 1) % m.shape[1]],
        )
    return m[i][(j + 1) % m.shape[1]], m[i][(j - 1) % m.shape[1]]


for i, linha in enumerate(m_lua):
    for j, pixel in enumerate(linha):
        if is_max_local(m_lua, d_lua, i, j):
            matriz_selecao_lua[i][j] = pixel

final_matriz_selecao_lua_m = matriz_selecao_lua - np.min(matriz_selecao_lua)
final_matriz_selecao_lua = np.rint(
    255 * (final_matriz_selecao_lua_m / np.max(final_matriz_selecao_lua_m))
)

image_final_matriz_selecao_lua = Image.fromarray(final_matriz_selecao_lua)

for i, linha in enumerate(m):
    for j, pixel in enumerate(linha):
        if is_max_local(m, d, i, j):
            matriz_selecao[i][j] = pixel

final_matriz_selecao_m = matriz_selecao - np.min(matriz_selecao)
final_matriz_selecao = np.rint(
    255 * (final_matriz_selecao_m / np.max(final_matriz_selecao_m))
)

image_final_matriz_selecao = Image.fromarray(final_matriz_selecao)

image_final_matriz_selecao_lua.convert("L").show()
image_final_matriz_selecao.convert("L").show()
