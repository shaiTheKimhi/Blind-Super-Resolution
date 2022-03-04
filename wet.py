from locale import normalize
from re import L
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.linalg import toeplitz
import scipy.misc
import sklearn
import sklearn.neighbors
import cv2

ALPHA = 3  # may choose alpha


def matrixForConv(shape, F):
    """
    Arg:
    I -- 2D numpy matrix
    F -- numpy 2D matrix
    Returns:
    output -- matrix that multiplying it with I will return conv(I, F)
    """
    # number of columns and rows of the filter
    FRowNum, FColNum = F.shape

    #  calculate the output dimensions
    outputRowNum = shape[0] + FRowNum - 1
    outputColNum = shape[1] + FColNum - 1

    # zero pad the filter
    F = np.pad(F, ((outputRowNum - FRowNum, 0),
                               (0, outputColNum - FColNum)),
                           'constant', constant_values=0)

    # Use each row of  F to creat toeplitz.
    toeplitzList = []
    for i in range(F.shape[0] - 1, -1, -1):  # iterate from last row to the first row
        c = F[i, :] 
        r = np.r_[c[0], np.zeros(shape[1] - 1)]  
        toeplitz_m = toeplitz(c, r)  
        toeplitzList.append(toeplitz_m)

    c = range(1, F.shape[0] + 1)
    r = np.r_[c[0], np.zeros(shape[0] - 1, dtype=int)]
    toeplitz_hw = toeplitz(c, r)

    toeplitzShape = toeplitzList[0].shape  
    h = toeplitzShape[0] * toeplitz_hw.shape[0]
    w = toeplitzShape[1] * toeplitz_hw.shape[1]
    doublyBlockedShape = [h, w]
    doublyBlocked = np.zeros(doublyBlockedShape)

    b_h, b_w = toeplitzShape 
    for i in range(toeplitz_hw.shape[0]):
        for j in range(toeplitz_hw.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doublyBlocked[start_i: end_i, start_j:end_j] = toeplitzList[toeplitz_hw[i, j] - 1]

    convolutionMatrix = []
    convolutionSize = shape[0] + FRowNum - 1
    size = (convolutionSize - FRowNum)
    start = int(np.math.floor(size / 2)) * convolutionSize + int(np.math.ceil(size / 2))
    index = start

    while len(convolutionMatrix) < FRowNum ** 2:
        for j in range(FRowNum):
            convolutionMatrix.append(doublyBlocked[index + j])
        index += convolutionSize

    convolutionMatrix = np.array(convolutionMatrix)
    return convolutionMatrix


def plotResults(restoredImg, blurredWith, restoredWith, origImg):
    plt.imshow(restoredImg, cmap='gray')
    PSNR = psnr(origImg[:,:], restoredImg[:,:])
    plt.title(f'image blurred with {blurredWith} and restored with {restoredWith}. PSNR={PSNR:.2f}')
    plt.savefig(blurredWith + " + " + restoredWith)
    plt.show()


def psnr(first, second):
    return 20 * np.log10(first.max() / (np.sqrt(np.mean((first - second) ** 2))))


class Image:
    def __init__(self, imgArr, windowSize=15):
        self.gaussianKernel = self.__gaussianMatrix(np.linspace(-(windowSize / 16), windowSize / 16, windowSize))
        plt.imshow(self.gaussianKernel, cmap='gray')
        plt.title("Real gaussian PSF")
        plt.show()
        self.sincKernel = self.__sincMatrix(np.linspace(-(windowSize / 4), windowSize / 4, windowSize))

        self.gaussianImg = signal.convolve2d(imgArr, self.gaussianKernel, mode='same', boundary='wrap')
        self.sincImg = signal.convolve2d(imgArr, self.sincKernel, mode='same', boundary='wrap')

        # first practical assignment:
        self.lowResGaussian = self.gaussianImg[::ALPHA, ::ALPHA]
        self.lowResSinc = self.sincImg[::ALPHA, ::ALPHA]

        plt.title('low-res gaus-image')
        plt.imshow(self.lowResGaussian, cmap='gray')
        plt.savefig("gaussLowRes")
        plt.show()

        plt.title('low-res sinc-image')
        plt.imshow(self.lowResSinc, cmap='gray')
        plt.savefig("sincLowRes")
        plt.show()


    def restoreSinc(self, k):
        return self.__wienerFilterToUpsample(self.lowResSinc, k)

    def restoreGaussian(self, k):
        return self.__wienerFilterToUpsample(self.lowResGaussian, k)

    def __gaussianMatrix(self, evenlySpacedUnion):
        mu = 0
        sigma = 1
        x, y = np.meshgrid(evenlySpacedUnion, evenlySpacedUnion)
        d = np.sqrt(x ** 2 + y ** 2)
        gaussian = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        return gaussian / gaussian.sum()

    def __sincMatrix(self, evenlySpacedUnion):
        xxMatrix = np.outer(evenlySpacedUnion, evenlySpacedUnion)
        sinc = np.sinc(xxMatrix)
        return sinc / sinc.sum()

    def __wienerFilterToUpsample(self, lowSampled, psf, k=0.1, alpha=ALPHA):
        if np.sum(psf):
            psf /= np.sum(psf)
        newSize = (int(lowSampled.shape[1] * alpha), int(lowSampled.shape[0] * alpha))
        img = cv2.resize(lowSampled, dsize=newSize, interpolation=cv2.INTER_CUBIC)
        psf = fftpack.fft2(psf, shape=img.shape)
        psf = np.conj(psf) / (np.abs(psf) ** 2 + k)
        return np.abs(fftpack.ifft2(fftpack.fft2(np.copy(img)) * psf))


class patches:
    def __init__(self, imgArr, patchSize):
        self.patchSize = patchSize
        self.r = self.__createPatches(imgArr, patchSize, 1)  # no use of alpha

        self.q = self.__createPatches(imgArr, patchSize)
        self.qVec = np.array([patch.reshape(patch.size) for patch in self.q])
        #self.Rj_calc = MatrixCalc(self.r)
        delta = fftpack.fftshift(scipy.signal.unit_impulse((patchSize, patchSize)))
        self.Rj = calcR(self.r, delta)    

    def __createPatches(self, imgArr, size, alpha=ALPHA):
        size = int(size / alpha)
        step = int(size * alpha / self.patchSize)
        patches = []
        for i in range(0, imgArr.shape[0] - size, step):
            for j in range(0, imgArr.shape[1] - size, step):
                patches.append(imgArr[i:i + size, j:j + size])
        return patches

    def calcSumElement(self, i, j):
        return self.Rj[j].T @ self.Rj[j], self.Rj[j].T @ self.qVec[i]

    def calculateWeights(self, k, sigmaNN):
        numNeighbors = 5
        rAlpha = np.array([element @ k for element in self.Rj])
        tree = sklearn.neighbors.BallTree(rAlpha, leaf_size=2)
        distWeights = np.zeros((len(self.q), len(self.r)))

        for i, qi in enumerate(self.qVec):
            _, neighbor_indices = tree.query(np.expand_dims(qi, 0), k=numNeighbors)
            for j in neighbor_indices:
                distWeights[i, j] = np.exp(-0.5 * (np.linalg.norm(qi - rAlpha[j]) ** 2) / (sigmaNN ** 2))

        #normalize weights
        n = distWeights.shape[0]
        return distWeights /  (np.sum(distWeights, axis=1).reshape(n,1) + 1e-6)



def calcR(rpatches, kernel, alpha=ALPHA):
    #applying matrixForConv for each patch and than downsampling in the first dimension
    return [matrixForConv(kernel.shape, patch)[::alpha ** 2, ::] for patch in rpatches]


def reconstruct_kernel(patches):
    psize = patches.r[0].__len__()
    csqr = laplacian(psize)
    csqr = csqr.T @ csqr
    delta = fftpack.fftshift(scipy.signal.unit_impulse((psize, psize)))
    k = delta.reshape(delta.size)
    for t in range(5):
        k = iterate(k, csqr, patches)
    return k.reshape((psize, psize))

def iterate(k, CSquared, patches):
    sigmaNN = 0.06
    neighborsWeights = patches.calculateWeights(k, sigmaNN)
    size = k.shape[0]
    matEpsilon = np.ones((size, size)) * 1e-10
    sumLeft = np.zeros((size, size))
    sumRight = np.zeros_like(k)
    for i in range(neighborsWeights.shape[0]):
        for j in range(neighborsWeights.shape[1]):
            if neighborsWeights[i, j]:
                R = patches.Rj
                left, right = R[j].T @ R[j], R[j].T @ patches.qVec[i] #changed
                sumLeft += neighborsWeights[i, j] * left + CSquared
                sumRight += neighborsWeights[i, j] * right

    return np.linalg.inv(sumLeft * (sigmaNN ** -2) + matEpsilon) @ sumRight

def laplacian(length):
    val1 = -1
    val2 = 4

    # digonals init
    diag1 = np.zeros((length, length))
    diag2 = np.zeros((length, length))

    for i in range(length):
        if (i + 1) in range(length):
            diag1[i, i + 1] = val1
        if (i - 1) in range(length):
            diag1[i, i - 1] = val1

    for i in range(length):
        diag1[i, i] = val2
        diag2[i, i] = val1

    # first diagonal
    start = 0
    end = length
    C = np.zeros((length ** 2, length ** 2))

    for num_matrices in range(length):
        C[start: end, start: end] = diag1
        start += length
        end += length

    # second diagonal
    startX = 0
    endXstartY = length
    endY = 2 * length

    for num_matrices in range(length - 1):
        C[startX: endXstartY, endXstartY: endY] = diag2
        C[endXstartY: endY, startX: endXstartY] = diag2
        startX += length
        endXstartY += length
        endY += length

    return C


def main():
    imgArr = np.array(plt.imread("DIPSourceHW2.png"))[:, :, 0]
    # imgArr = np.transpose(imgArr)
    imgArr /= imgArr.max()
    expandImg = np.zeros((imgArr.shape[0]+2, imgArr.shape[1]+2))
    expandImg[1:expandImg.shape[0] - 1, 1:expandImg.shape[1] - 1] = imgArr
    filteredImage = Image(expandImg)
    patchSize = 15

    low_res_G = filteredImage.lowResGaussian
    gaussianPatches = patches(low_res_G, patchSize)
    gaussianOptimalK = reconstruct_kernel(gaussianPatches)
    temp = np.copy(gaussianOptimalK)
    gaussianRestoredOptimal = filteredImage.restoreGaussian(gaussianOptimalK)
    gaussianOptimalK = np.copy(temp)
    sincRestoredNotOptimal = filteredImage.restoreSinc(gaussianOptimalK)
    gaussianOptimal = filteredImage.restoreGaussian(filteredImage.gaussianKernel) #get the optimal image using the true gaussian (not estimation)

    low_res_S = filteredImage.lowResSinc
    sincPatches = patches(low_res_S, patchSize)
    sincOptimalK = reconstruct_kernel(sincPatches)
    temp = np.copy(sincOptimalK)
    sincRestoredOptimal = filteredImage.restoreSinc(sincOptimalK)
    sincOptimalK = np.copy(temp)
    gaussianRestoredNotOptimal = filteredImage.restoreGaussian(sincOptimalK)
    sincOptimal = filteredImage.restoreSinc(filteredImage.sincKernel) #get the optimal image using the true sinc (not estimation)

    ## plot results and PSNR relative to original high-res image
    plotResults(gaussianRestoredOptimal, "Gauss-ker", "Gauss-ker", expandImg)
    plotResults(gaussianRestoredNotOptimal, "Gauss-ker", "Sinc-ker", expandImg)
    plotResults(gaussianOptimal, "Gauss-ker",  "Gauss-ker-True", expandImg)
    plotResults(sincRestoredOptimal, "Sinc-ker", "Sinc-ker", expandImg)
    plotResults(sincRestoredNotOptimal, "Sinc-ker", "Gauss-ker", expandImg)
    plotResults(sincOptimal, "Sinc-ker", "Sinc-ker-True", expandImg)
    
    plt.imshow(gaussianOptimalK, cmap='gray')
    plt.show()

    plt.imshow(sincOptimalK, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()