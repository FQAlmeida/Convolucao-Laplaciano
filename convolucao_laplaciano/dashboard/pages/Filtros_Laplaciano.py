import numpy as np
import streamlit as st
from PIL import Image
import streamlit_toggle as toggle
from convolucao_laplaciano.dashboard.commom import convolution

st.title("Convolução Laplaciano")

image = Image.open("./data/11_test.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image)

st.markdown("#### Kernel")

col_1, col_2 = st.columns(2)

with col_2:
    c = 1 if toggle.st_toggle_switch("C", label_after=True, default_value=True) else -1
    dev_step = toggle.st_toggle_switch("Option", label_after=True, default_value=True)

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


@st.cache_data
def conv(image_array: np.ndarray, kernel: np.ndarray):
    return convolution(image_array, kernel)


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
