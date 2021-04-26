import numpy as np
from scipy import signal

def magnitude(img):
    
    # Initializing filters and matrices
    dx = np.array([[-1, 1]])
    dy = np.array([[-1], [1]])
    m, n, c = img.shape
    square = np.zeros((m, n))
    
    # Loop through colour channels
    for i in range(c):
        grad_x = signal.convolve2d(img[:, :, i], dx, boundary='symm', mode='same')
        grad_y = signal.convolve2d(img[:, :, i], dy, boundary='symm', mode='same')
        square += grad_x**2 + grad_y**2
    
    return np.sqrt(square)


def LoG(u, v, sigma):
    t1 = -1 / (np.pi * sigma**4)
    t2 = 1 - ((u**2 + v**2) / (2 * sigma**2))
    t3 = np.exp(-(u**2 + v**2) / (2 * sigma**2))
    
    return t1 * t2 * t3

def create_LoG_filter(sigma):
    n = round(6 * sigma)
    if (n%2 == 0):
        n += 1
    k = (n - 1) / 2
    F = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            F[i, j] = LoG(i - k, j - k, sigma)
    return F
