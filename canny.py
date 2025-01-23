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


def non_maximum_suppression(magnitude, direction):
    """Aplica supressão não máxima para refinar as bordas."""
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.float32)
    angle = np.rad2deg(direction) % 180  # Converte ângulos para [0, 180]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            try:
                # Determina os vizinhos na direção do gradiente
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neighbors = (magnitude[i, j + 1], magnitude[i, j - 1])
                elif 22.5 <= angle[i, j] < 67.5:
                    neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
                elif 67.5 <= angle[i, j] < 112.5:
                    neighbors = (magnitude[i + 1, j], magnitude[i - 1, j])
                else:
                    neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

                # Mantém o valor somente se for um máximo local
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]
            except IndexError:
                pass
    return suppressed


def threshold_hysteresis(image, low_threshold, high_threshold):
    """Aplica limiarização por histerese."""
    strong = 255
    weak = 75

    output = np.zeros_like(image, dtype=np.uint8)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    # Conecta bordas fracas a fortes
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if output[i, j] == weak:
                if strong in output[i - 1:i + 2, j - 1:j + 2]:
                    output[i, j] = strong
                else:
                    output[i, j] = 0
    return output


# Pipeline do algoritmo Canny
def canny_filter(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, kernel_size=3, sigma=1.0)
    magnitude, direction = sobel_gradients(blurred)
    suppressed = non_maximum_suppression(magnitude, direction)
    edges = threshold_hysteresis(suppressed, low_threshold, high_threshold)
    return edges

if __name__ == "__main__":
    image = cv2.imread("360_F_299042079_vGBD7wIlSeNl7vOevWHiL93G4koMM967.jpg")  # Substitua pelo caminho da sua imagem
    canny_result = canny_filter(image, low_threshold=50, high_threshold=150)

    cv2.imwrite('ResultadoCanny.png', canny_result)
    cv2.imshow("Original", image)
    cv2.imshow("Bordas Canny", canny_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
