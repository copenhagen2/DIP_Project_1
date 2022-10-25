import cv2
import numpy as np 
import numba 

@numba.jit(nopython=True)
def conv(img, x, y, kernel):
    x_max = len(img)
    y_max = len(img[0])
    size = len(kernel)
    radius = (size-1) // 2
    res = 0
    for i in range(size):
        for j in range(size):
            if -1<x+i-radius<x_max  and -1<y+j-radius<y_max:
                res +=  img[x+i-radius][y+j-radius] * kernel[i][j]
    return np.floor(res)

@numba.jit(nopython=True)
def linear_process(img, kernel):
    x_max = len(img)
    y_max = len(img[0])
    res = np.zeros(img.shape)
    for i in range(x_max):
        for j in range(y_max):
            res[i][j] = conv(img, i, j, kernel)
    return res

@numba.jit(nopython=True)
def gradient(img:np.ndarray):
    kernel_x = np.array([[-1, 0 ,1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    g_x = linear_process(img, kernel_x)
    g_y = linear_process(img, kernel_y)
    g = np.sqrt(np.square(g_x)+np.square(g_y))
    return g

@numba.jit(nopython=True)
def laplacian(img):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    img_reinforce = linear_process(img, kernel)
    return img_reinforce

@numba.jit(nopython=True)
def boundary(src:np.ndarray, grad:np.ndarray, T):
    res = np.zeros(src.shape)
    x_max, y_max = src.shape
    for x in range(x_max):
        for y in range(y_max):
            if grad[x][y] <= T:
                continue

            if src[x][y] == 0:
                res[x][y] = 255
            # horizontial
            if -1<x+1<x_max and -1<x-1<x_max and src[x+1][y] * src[x-1][y] < 0:
                res[x][y] = 255

            # vertical
            if -1<y+1<y_max and -1<y-1<y_max and src[x][y+1] * src[x][y-1] < 0:
                res[x][y] = 255
    return res

@numba.jit(nopython=True)
def calculate_mean(src:np.ndarray, lap:np.ndarray, grad:np.ndarray, T):
    r, c = 0, 0
    x_max, y_max = src.shape
    for x in range(x_max):
        for y in range(y_max):
            if grad[x][y] >= T:
                if (-1<x+1<x_max and -1<x-1<x_max and lap[x+1][y] * lap[x-1][y] < 0) or (-1<y+1<y_max and -1<y-1<y_max and lap[x][y+1] * lap[x][y-1] < 0):
                    r += src[x][y]
                    c += 1
    return r/c

@numba.jit(nopython=True)
def binarize(src:np.ndarray, threshold):
    x_max, y_max = src.shape
    res = np.zeros(src.shape)
    for x in range(x_max):
        for y in range(y_max):
            if src[x][y] > threshold:
                res[x][y] = 255
    return res


if __name__ == "__main__":
    img = cv2.imread("PRLetter-images\\1_gray.bmp",cv2.IMREAD_GRAYSCALE)
    laplace_img = laplacian(img)
    grad_img = gradient(img)
    r = calculate_mean(img, laplace_img, grad_img, 200)
    bi_img = binarize(img, r)
    cv2.imshow("bi", bi_img)
    cv2.waitKey(0)
    