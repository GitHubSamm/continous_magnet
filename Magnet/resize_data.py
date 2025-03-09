import torch

def resize_dataset(dataset, n_samples, n_cls=10):
    """This function aims to resize the CIFAR-10 dataset with a given number 
    of samples per class.
    It assumes that the labels are digits from 0 to n_class"""
    imgs_per_class = [[] for i in range(n_cls)]
    labels_per_class = [[] for i in range(n_cls)]

    for img, label in dataset:
        if(len(imgs_per_class[label]) < n_samples):
            imgs_per_class[label].append(img)
            labels_per_class[label].append(label)

    imgs_per_class = [torch.stack(imgs) for imgs in imgs_per_class]
    labels_per_class = [torch.tensor(labels) for labels in labels_per_class]

    images_set = torch.cat(imgs_per_class)
    labels_set = torch.cat(labels_per_class)

    return images_set, labels_set