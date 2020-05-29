import cv2  # Needs the package OpenCV to be installed. Check Anaconda Environments and Packages.
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import time
from multiprocessing import Pool
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import seaborn as sns
import pandas as pd

DATASET_ROOT = "./datasets"
DATASET_FACES94 = DATASET_ROOT + "/faces94"
DATASET_FACES94_MALE = DATASET_FACES94 + "/male"
DATASET_FACES94_FEMALE = DATASET_FACES94 + "/female"
DATASET_FACES94_MALESTAFF = DATASET_FACES94 + "/malestaff"
DATASET_FACES95 = DATASET_ROOT + "/faces95"
DATASET_FACES96 = DATASET_ROOT + "/faces96"
DATASET_GRIMACE = DATASET_ROOT + "/grimace"
DATASET_LANDSCAPE = DATASET_ROOT + "/naturalLandscapes"
MEDIAN_FILE_NAME = "median_data.dat"
DISTANCE_FILE_NAME = "distance_data.dat"


def readFaces94MaleFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_MALE, gray)


def readFaces94FemaleFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_FEMALE, gray)


def readFaces94MaleStaffFaces(gray=False):
    return readImagesFromDataset(DATASET_FACES94_MALESTAFF, gray)


def readFaces94AllFaces(gray=False):
    npMaleFaces = readFaces94MaleFaces(gray)
    npFemaleFaces = readFaces94FemaleFaces(gray)
    npMaleStaffFaces = readFaces94MaleStaffFaces(gray)

    return np.concatenate((npMaleFaces, npMaleStaffFaces, npFemaleFaces))


def readImagesFromDataset(datasetDir, gray=False):
    images = []
    directories = glob.glob(datasetDir + "/*")
    for directory in directories:
        images += readImagesFromDirectory(directory, gray)

    return np.array(images, dtype="float32")


def readImagesFromDirectory(directory, gray=False, size=(180, 200), verbose=False):
    images = []
    imageNames = glob.glob(directory + "/*.jpg")
    for imageName in imageNames:
        if verbose:
            print("Currently reading {}".format(imageName))
        image = cv2.resize(cv2.imread(imageName), size)
        # Convert to gray in order to reduce the dimensionality of the data set
        # only if stated by the parameter for gray
        images.append(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if gray else image
        )

    return images


## ======= Read images of natural landscape

def readLandsCapeImage(gray=False):
    return readImagesFromDirectory(DATASET_LANDSCAPE, gray)


## ============== Get distances to calculate median and spearman correlation of output
def dist_to_all(A: np.ndarray) -> np.ndarray:

    d = get_point_distances_parallel(A, n_processes=8)

    return d


def dist_to_all_bound(A: np.ndarray, start: int, finish: int, verbose: bool = False) -> np.ndarray:
    dist = []
    print("Start Calculating")
    for i in range(start, finish):
        if verbose:
            print("calculating distance for element {} vs all data".format(i))
        distances = np.linalg.norm(np.subtract(A[i], A), ord=2, axis=1)
        sum_dist = np.sum(distances, axis=0)
        dist.append(sum_dist)
    print("Calculated images from {} to {} for a total of {}".format(start, finish, finish - start))
    return np.array(dist)


def dist_from_all_bound(A: np.ndarray, start: int, finish: int, verbose: bool = False) -> np.ndarray:
    dist = np.zeros(A.shape[0])
    for i in range(start, finish):
        if verbose:
            print("calculating distance for all data from {}".format(i))
            print("Current size of dist: {}".format(dist.shape))
        dist += np.linalg.norm(np.subtract(A[i], A), ord=2, axis=1)
    return dist


def dist_from_all_parallel(A: np.ndarray, start: int, finish: int, n_processes: int = 4) -> np.ndarray:
    if n_processes < 1:
        n_processes = 8

    num_ind = int((finish - start) / n_processes)

    indices = [min(start + i * num_ind, finish) for i in range(n_processes + 1)]
    args = [(A / 255, indices[i], indices[i + 1] + 1, True) for i in range(len(indices) - 1)]

    with Pool(n_processes) as p:
        print("Started computing distances")
        start_time = time.time()
        results = p.starmap(dist_from_all_bound, args)
        print(results)
        d = np.sum(results, axis=0)
        finish_time = time.time() - start_time
        print("Total time elapsed in minutes: {}".format(finish_time / 60))
        return d


def dist_from_all(A: np.ndarray, start: int, finish: int, filename: str) -> np.ndarray:
    if os.path.exists(filename):
        print("File already exists, returning file")
        return np.genfromtxt(filename, delimiter=',')
    print("Getting distances")
    dist = dist_from_all_parallel(A, start, finish, n_processes=8)

    print("Saving file {}".format(filename))
    np.savetxt(filename, dist.reshape((1, -1)), delimiter=",")

    return dist


def get_median_image(A: np.ndarray) -> np.ndarray:
    if os.path.exists(MEDIAN_FILE_NAME):
        print("File already exists, returning file")
        return np.genfromtxt(MEDIAN_FILE_NAME, delimiter=',')
    dist = dist_to_all(A)

    index = np.argmin(dist)

    median = A[index]
    np.savetxt(MEDIAN_FILE_NAME, median.reshape((1, -1)), delimiter=",")
    return median


def get_point_distances_parallel(A: np.ndarray, n_processes: int = 4) -> np.ndarray:
    if n_processes < 1:
        n_processes = 8

    dataset_N = A.shape[0]

    num_ind = int(np.ceil(dataset_N / n_processes))

    indices = [min(i * num_ind, dataset_N - 1) for i in range(n_processes + 1)]
    args = [(A / 255, indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
    results = []

    with Pool(n_processes) as p:
        print("Started computing distances")
        start_time = time.time()
        results = p.starmap(dist_to_all_bound, args)
        d = np.concatenate(results)
        finish_time = time.time() - start_time
        print("Total time elapsed in minutes: {}".format(finish_time / 60))
        return d


def get_spearman_cov(A: np.ndarray) -> np.ndarray:
    mad_A = st.median_absolute_deviation(A, axis=1)
    sp_corr = get_spearman_corr(A)

    return calculate_covariance_matrix(sp_corr, mad_A, A.shape[0])


def calculate_covariance_matrix(corr_m: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = corr_m[i, j] * s[i] * s[j]
    return cov


def get_spearman_corr(A: np.ndarray) -> np.ndarray:
    print("Get rank matrix")
    rank_A = A.argsort(axis=1).argsort(axis=1)

    rank_mean = rank_A.mean()
    rank_std = rank_A.std()
    print("Normalize rank matrix")
    rank_A = (rank_A - rank_mean) / rank_std

    _corr = (1 / A.shape[1]) * np.matmul(rank_A, rank_A.T)

    return _corr


def get_num_variables_percentage(s, desired_percentage=0.85):
    var_percentages = s / s.sum()

    count = 0
    current_percentage = 0
    for vp in var_percentages:
        current_percentage += vp
        count += 1
        if current_percentage > desired_percentage:
            break
    return count


def show_eigen_space(eig: np.ndarray, height: int, width: int):
    cols = 4
    rows = 4

    plt.figure(figsize=(30, 20))

    for i in np.arange(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(eig[:, i].reshape(height, width), plt.cm.gray)


def show_boxplot(d, figsize=(15, 4)):
    plt.figure(figsize=figsize)
    plt.title('Boxplot')
    plt.boxplot(d, 0, 'rs', 0);
    plt.show()


def show_histogram(d, figsize=(15, 4)):
    plt.figure(figsize=figsize)
    plt.title('Histogram')
    plt.grid(True)
    plt.hist(d)
    plt.show()


def get_depth_test(dist_A: np.ndarray, dist_B: np.ndarray):
    get_depth = lambda x: 1 - x / np.sum(x)
    depth_A = get_depth(dist_A)
    depth_B = get_depth(dist_B)
    results = sm.OLS(depth_B, sm.add_constant(depth_A)).fit()

    print(results.summary())
    plt.figure(figsize=(30, 20))
    plt.scatter(depth_A, depth_B)
    plt.plot(depth_A, depth_A * results.params[1] + results.params[0], 'r-')
    plt.show()


def split_indices(indices: np.ndarray, labels: np.ndarray, val_size: float = 0.2, test_size: float = 0.1) -> tuple:
    N = indices.shape[0]
    train_size = 1 - test_size
    true_val_size = val_size / (train_size)
    ind_train1, ind_test, y_train1, _ = train_test_split(indices, labels, test_size=test_size, random_state=0,
                                                         stratify=labels)
    ind_train, ind_val, _, _ = train_test_split(ind_train1, y_train1, test_size=true_val_size, random_state=0,
                                                stratify=y_train1)

    return ind_train, ind_val, ind_test


def show_items_class(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(15, 4))
    plt.bar(unique, counts, 0.9)
    plt.show()


def get_classification_results(y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
    print("Accuracy: {}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))

    prec_score = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    for i, class_name in enumerate(class_names):
        print("Precision for class {}: {}".format(class_name, prec_score[i]))

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names))

    classes_dict = {'Actual': y_true.tolist(), 'Predicted': y_pred.tolist()}
    classes_df = pd.DataFrame(classes_dict, columns=["Actual", "Predicted"])
    conf_matrix = pd.crosstab(classes_df['Actual'], classes_df['Predicted'], rownames=['Actual'],
                              colnames=['Predicted'])

    plt.figure(figsize=(12, 10))
    plt.title("Heatmap")
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.0f')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    ax.invert_yaxis()
