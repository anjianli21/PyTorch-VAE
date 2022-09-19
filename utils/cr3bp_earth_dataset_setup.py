import numpy as np
import pickle
import os
import warnings
from sklearn import preprocessing

def cr3bp_earth_dataset_setup(data_dir_list, data_distribution, train_size, val_size):

    if data_distribution == "cr3bp_earth_local_optimal":
        file_path = "Data/cr3bp_earth/default_local_optimal_solution_0911_0912.pickle"
    else:
        warnings.warn("incorrect data type")
        exit()
    if not os.path.exists(file_path):
        data_file_list = []

        for data_dir in data_dir_list:
            for root, dirs, files in os.walk(data_dir):
                files = [root + '/' + file for file in files]
                data_file_list.extend(files)

        data = []
        for file in data_file_list:
            f = open(file, 'rb')
            data_list = pickle.load(f)
            for data_point in data_list:
                if data_point["snopt_inform"] == 1 and data_point["feasibility"]:
                    data.append(data_point["results.control"])
        with open(file_path, "wb") as fp:  # write pickle
            pickle.dump(data, fp)
    else:
        with open(file_path, "rb") as f:  # load pickle
            data = pickle.load(f)

    data = np.asarray(data)
    if np.shape(data)[0] < train_size + val_size:
        warnings.warn("not enough data!")
        exit()

    # Normalize the data
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    print(f"min of data is {np.min(data, axis=0)}, max of data is {np.max(data, axis=0)}")

    # Save min and max of the data
    min_max_data_path = "Data/cr3bp_earth/default_local_optimal_solution_0911_0912_min_max.pickle"
    min_max_data = {"data_min": np.min(data, axis=0),
                    "data_max": np.max(data, axis=0)}
    with open(min_max_data_path, "wb") as fp:  # write pickle
        pickle.dump(min_max_data, fp)

    train_data = normalized_data[:train_size, :]
    val_data = normalized_data[train_size:train_size + val_size, :]

    return train_data, val_data


def filter_raw_snopt_control_evaluation(raw_eval):
    distance_matrix = get_distance_matrix(raw_eval, raw_eval)

    index_set = []
    for i in range(np.shape(distance_matrix)[0]):
        if i == 0:
            index_set.append(i)
            continue
        else:
            if np.min(distance_matrix[:i, i]) < 1:
                continue
            else:
                index_set.append(i)

    return raw_eval[index_set, :]


def get_distance_matrix(A, B, squared=False):
    ## adpted from https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared