import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if (matrix_a.ndim != 2 and matrix_b.ndim != 2):
        raise ValueError("Аргументы должны быть двумерными матрицами")

    if (matrix_a.shape[1] != matrix_b.shape[0]):
        raise ValueError("Размеры матриц не позволяют перемножить")

    m, common, n = *matrix_a.shape, matrix_b.shape[1]
    res = np.zeros((m, n), dtype=np.int16)

    for i in range(m):
        for j in range(n):
            for k in range(common):
                res[i, j] += matrix_a[i, k] * matrix_b[k, j]

    return res


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # put your code here
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    pass
