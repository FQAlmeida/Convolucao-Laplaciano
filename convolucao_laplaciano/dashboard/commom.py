from math import pi, e
import numpy as np
from functools import partial
from multiprocessing.pool import ThreadPool as Pool


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


def convolution(image_array: np.ndarray, kernel: np.ndarray):
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


def gauss(x: np.ndarray, y: np.ndarray, sigma: float):
    return 1 / (2 * pi * sigma**2) * e ** (-(x**2 + y**2) / (2 * sigma**2))
