import torch

def resize_dataset(dataset, n_samples, n_cls=10):

    imgs_per_class = [[] for i in range(n_cls)]
    labels_per_class = [[] for i in range(n_cls)]

    for img, label in dataset:
        if(len(imgs_per_class[label]) < n_samples):
            imgs_per_class[label].append(img)
            labels_per_class[label].append(label)

    images_set = torch.stack(imgs_per_class)
    labels_set = torch.tensor(labels_per_class)

    return images_set, labels_set