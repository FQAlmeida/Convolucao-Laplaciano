import numpy as np
import streamlit as st
from PIL import Image
import streamlit_toggle as toggle

from convolucao_laplaciano.dashboard.commom import convolution, gauss

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
    is_const = toggle.st_toggle_switch("Constante", label_after=True)

divisor = 9 if is_const else (size**2)
kernel = np.ones((size, size)) / divisor

with col_1:
    st.write(kernel)

image_array = np.array(image)


@st.cache_data
def conv(image_array: np.ndarray, kernel: np.ndarray):
    return convolution(image_array, kernel)


image_array_conv = conv(image_array, kernel)

image_array_conv_min = image_array_conv - np.min(image_array_conv)
image_array_conv = np.rint(255 * (image_array_conv_min / np.max(image_array_conv_min)))

image_final_conv = Image.fromarray(image_array_conv)

st.image(image_final_conv.convert("L"))


# region gaussiano


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
kernel = gauss(kernel_x, kernel_y, sigma)

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
