import streamlit as st
from streamlit_toggle import st_toggle_switch as toggle
import numpy as np
from PIL import Image
from multiprocessing import Pool

st.title("Convolução Laplaciano")

image = Image.open("./data/11_test.png")

image = image.convert("L")

st.markdown("#### Imagem Original")
st.image(image, width=200)

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

st.write(np.max(image_final_array))
st.write(np.min(image_final_array))

image_final_conv = Image.fromarray(image_array_conv)

image_final = Image.fromarray(image_final_array)

st.image(image_final_conv.convert("L"), width=200)
st.image(image_final.convert("L"), width=200)

# image_final.show()
# image.show()
