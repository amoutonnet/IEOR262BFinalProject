import numpy as np


PI = np.pi

LISTF = ["Rosenbrock", "Sphere", "Rastigrin", "Easom", "StyblinskiTang", "Holder", "CrossInTray"]
LISTG = ["Rosenbrock", "Sphere", "Rastigrin", "Easom", "StyblinskiTang"]
LISTH = ["Rosenbrock", "Sphere", "Rastigrin", "Easom", "StyblinskiTang"]


def fRosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def gRosenbrock(x, y):
    return np.array([
        [-400 * (y - x**2) * x - 2 * (2 - x)],
        [200 * (y - x**2)]
    ])


def hRosenbrock(x, y):
    return np.array([
        [1200 * x**2 - 400 * y + 2, -400 * x],
        [-400 * x, 200]
    ])


def fSphere(x, y):
    return x**2 + y**2


def gSphere(x, y):
    return np.array([
        [2 * x],
        [2 * y]
    ])


def hSphere(x, y):
    return 2 * np.eye(2)


def fRastigrin(x, y):
    return 20 + x**2 - 10 * np.cos(2 * PI * x) + y**2 - 10 * np.cos(2 * PI * y)


def gRastigrin(x, y):
    return np.array([
        [2 * x + 10 * 2 * PI * np.sin(2 * PI * x)],
        [2 * y + 10 * 2 * PI * np.sin(2 * PI * y)]
    ])


def hRastigrin(x, y):
    return np.array([
        [2 + 10 * 4 * (PI**2) * np.cos(2 * PI * x), 0],
        [0, 2 + 10 * 4 * (PI**2) * np.cos(2 * PI * y)]
    ])


def fEasom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - PI)**2 + (y - PI)**2))


def gEasom(x, y):
    return np.array([
        [(np.sin(x) + 2 * (x - PI) * np.cos(x)) * np.cos(y) * np.exp(-((x - PI)**2 + (y - PI)**2))],
        [(np.sin(y) + 2 * (y - PI) * np.cos(y)) * np.cos(x) * np.exp(-((x - PI)**2 + (y - PI)**2))]
    ])


def hEasom(x, y):
    h = np.zeros((2, 2))
    h[0][0] = -(-2 * x + 2 * PI)**2 * np.exp(-(x - PI)**2 - (y - PI)**2) * np.cos(x) * np.cos(y) + 2 * (-2 * x + 2 * PI) * \
        np.exp(-(x - PI)**2 - (y - PI)**2) * np.sin(x) * np.cos(y) + 3 * np.exp(-(x - PI)**2 - (y - PI)**2) * np.cos(x) * np.cos(y)
    h[0][1] = -(-2 * x + 2 * PI) * (-2 * y + 2 * PI) * np.exp(-(x - PI)**2 - (y - PI)**2) * np.cos(x) * np.cos(y) + (-2 * x + 2 * PI) * np.exp(-(x - PI)**2 - (y - PI)**2) * \
        np.sin(y) * np.cos(x) + (-2 * y + 2 * PI) * np.exp(-(x - PI)**2 - (y - PI)**2) * np.sin(x) * np.cos(y) - np.exp(-(x - PI)**2 - (y - PI)**2) * np.sin(x) * np.sin(y)
    h[1][0] = h[0][1]
    h[1][1] = -(-2 * y + 2 * PI)**2 * np.exp(-(x - PI)**2 - (y - PI)**2) * np.cos(x) * np.cos(y) + 2 * (-2 * y + 2 * PI) * \
        np.exp(-(x - PI)**2 - (y - PI)**2) * np.sin(y) * np.cos(x) + 3 * np.exp(-(x - PI)**2 - (y - PI)**2) * np.cos(x) * np.cos(y)
    return h


def fStyblinskiTang(x, y):
    return 0.5 * (x**4 - 16 * x**2 + 5.0 * x + y**4 - 16 * y**2 + 5.0 * y)


def gStyblinskiTang(x, y):
    return np.array([
        [-16.0 + 2.0 * x**3 + 2.5],
        [-16.0 + 2.0 * y**3 + 2.5]
    ])


def hStyblinskiTang(x, y):
    return np.array([
        [6 * x**2 - 16, 0],
        [0, 6 * y**2 - 16]
    ])


def fHolder(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / PI)))


def fCrossInTray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))) + 1)**0.1
