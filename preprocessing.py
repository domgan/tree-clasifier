import glob
import numpy as np
from PIL import Image


def load_images(directory):
    result = []
    for f in glob.iglob(directory + "/*"):
        img = np.asarray(Image.open(f))
        if not img.shape == (224, 224, 3):
            continue
        img = img/255
        result.append(img)
    result = np.stack(result, axis=0)
    return result


def labels(images, label):
    result = np.zeros(images.shape[0], dtype=int) + label
    return result


def labels_matrix(labels_list, labels_no):
    result = np.zeros((labels_no, len(labels_list)))
    for i in range(len(labels_list)):
        result[labels_list[i]][i] = 1
    return result


chinatree_images = load_images("test_cut")
chinatree_labels = labels(chinatree_images, 0)

palm_images = load_images("test2_cut")
palm_labels = labels(palm_images, 1)

all_images = np.concatenate((chinatree_images, palm_images), 0)
all_labels = np.concatenate((chinatree_labels, palm_labels), 0)

shuffler = np.random.permutation(all_images.shape[0])
all_images_shuffled = all_images[shuffler]
all_labels_shuffled = labels_matrix(all_labels[shuffler], 2).T
