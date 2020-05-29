from ImageUtils import *
faces94_male = readFaces94MaleFaces(gray=True)
faces94_female = readFaces94FemaleFaces(gray=True)
faces94_malestaff = readFaces94MaleStaffFaces(gray=True)
landscapes = np.array(readLandsCapeImage(gray=True))

dataset = np.vstack((faces94_male, faces94_female, faces94_malestaff, landscapes))

#class 0: Landscapes
#class 1: male
#class 2: female

labels = np.concatenate((
    np.ones(faces94_male.shape[0]),
    np.full(faces94_female.shape[0], 2),
    np.ones(faces94_malestaff.shape[0]),
    np.zeros(landscapes.shape[0])
))

dataset_N, height, width = dataset.shape

A = dataset.reshape((dataset_N, height*width))

landscapes_start = dataset_N - landscapes.shape[0]

if __name__ == "__main__":
    dist_landscapes = dist_from_all(A, landscapes_start, dataset_N, "dist_landscapes.dat")
    print(dist_landscapes.shape)