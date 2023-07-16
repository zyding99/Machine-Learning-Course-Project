import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def low_rank_approximation(M, rank):
    SVD = np.linalg.svd(M, full_matrices = False)
    U, S, V = SVD
    l_U, l_V = len(U), len(V)
    M_h = np.zeros((l_U, l_V))
    for i in range(rank):
        M_h += S[i] * np.outer(U.T[i], V[i])
    return M_h

def image_generator(filename):
    image = mpl.image.imread(filename)
    ranks = [5, 20, 100]
    images = {}
    for rank in ranks:
        images[rank] = low_rank_approximation(M = image, rank = rank)
        mpl.image.imsave(filename + '_rank%s.jpg' % rank, images[rank], cmap = 'gray')
    mpl.image.imsave(filename + '_origin.jpg', image, cmap = 'gray')

def show_images():
    image_generator('data/face.jpg')
    image_generator('data/sky.jpg')

def MSE(filename, rank):
    image = mpl.image.imread(filename)
    image_h = low_rank_approximation(M = image, rank = rank)
    # error = 0
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         error += (image[i,j] - image_h[i,j]) ** 2
    # return error
    MSE = np.sum(np.power((image - image_h), 2))
    return MSE


def plot_both():
    face_errors, sky_errors = [], []
    interval = range(1, 101)
    for i in interval:
        face_error, sky_error = MSE('data/face.jpg', i), MSE('data/sky.jpg', i)
        face_errors.append(face_error)
        sky_errors.append(sky_error)
        print(i)
    plot = plt.plot(interval, face_errors, 'r',
            interval, sky_errors, 'g')
    plt.legend(plot, ('error of face.jpg','error of sky.jpg'))
    plt.xlabel('rank')
    plt.ylabel('error')
    plt.show()

def plot_partial(filename):
    errors = []
    interval = range(2, 50)
    for i in interval:
        error = MSE('data/%s.jpg' % filename, i)
        errors.append(error)
        print(i, error)
    plt.plot(interval, errors, 'b')   
    plt.xlabel('rank')
    plt.ylabel('error')
    plt.show()


if __name__ == "__main__":
    # show_images()
    # plot_partial('face')
    # plot_partial('sky')
    plot_both()