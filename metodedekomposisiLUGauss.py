# Nama   : Naufal Yoga Pratama
# NIM    : 21120122130059
# Kelas  : Metode Numerik C / Teknik Komputer 


import numpy as np
import unittest

# Fungsi Dekomposisi LU menggunakan metode eliminasi Gauss
def lu_decomposition_gauss(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Mengisi bagian diagonal L dengan 1
        L[i][i] = 1

        # Menghitung elemen-elemen U
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        # Menghitung elemen-elemen L
        for k in range(i + 1, n):
            sum = 0
            for j in range(i):
                sum += (L[k][j] * U[j][i])
            L[k][i] = (matrix[k][i] - sum) / U[i][i]

    return L, U

# Menyelesaikan sistem persamaan linear dengan Dekomposisi LU
def solve_lu_decomposition(A, b):
    L, U = lu_decomposition_gauss(A)
    n = len(A)
    # Substitusi maju untuk mencari y
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    # Substitusi mundur untuk mencari x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

# Soal yang diberikan
A = np.array([[-3, 2, -1], [6, -6, 7], [3, -4, 4]])
b = np.array([-1, -7, -6])

# Langkah-langkah penyelesaian
print("Langkah-langkah penyelesaian:")
print("\nCara pertama menggunakan metode dekomposisi LU dengan metode eliminasi Gauss untuk matriks koefisien.")
L, U = lu_decomposition_gauss(A)
print("Matriks L:")
print(L)
print("Matriks U:")
print(U)

print("\nCara Kedua menggunakan substitusi maju dan mundur untuk mencari solusi dari sistem persamaan linear.")

# Menyelesaikan sistem persamaan linear
solution = solve_lu_decomposition(A, b)
print("\nSolusi:")
print("x =", solution[0])
print("y =", solution[1])
print("z =", solution[2])


class TestLUDecomposition(unittest.TestCase):
    def test_decomposition(self):
        A = np.array([[-3, 2, -1], [6, -6, 7], [3, -4, 4]])
        expected_L = np.array([[1., 0., 0.], [-2., 1., 0.], [-1., 1., 1.]])
        expected_U = np.array([[-3., 2., -1.], [0., -2., 5.], [0., 0., -2.]])
        L, U = lu_decomposition_gauss(A)
        np.testing.assert_array_almost_equal(L, expected_L)
        np.testing.assert_array_almost_equal(U, expected_U)


if __name__ == '__main__':
    unittest.main()