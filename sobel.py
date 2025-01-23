import cv2
import numpy as np

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Aplica um filtro gaussiano para suavizar a imagem."""
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    return cv2.filter2D(image, -1, kernel)

def sobel_gradients(image):
    """Calcula os gradientes da imagem usando operadores de Sobel."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T

    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)

    return magnitude, direction

def sobel_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = gaussian_filter(gray)
    sobel_magnitude, sobel_direction = sobel_gradients(blur)
    sobel_magnitude_norm = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    return sobel_magnitude_norm

if __name__ == "__main__":
    image = cv2.imread('360_F_299042079_vGBD7wIlSeNl7vOevWHiL93G4koMM967.jpg')
    sobel_result = sobel_filter(image)

    cv2.imwrite('ResultadoSobel.png', sobel_result)
    cv2.imshow("Original", image)
    cv2.imshow('Resultado Sobel', sobel_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()