import torch
import numpy as np

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

def split_train_data(train_data, train_labels, n_increments):

    train_data_per_increment = []
    train_label_per_increment = []
    n_cls = len(torch.unique(train_labels))
    n_class_per_increment = n_cls // n_increments

    if n_cls % n_increments != 0:
        raise ValueError("n_increments should divide the number of classes in the dataset")

    # Assuming class labels goes from 0 to n_classes
    for incr in range(n_increments):
        cls_idx = torch.tensor([i+incr*n_class_per_increment for i in range(n_class_per_increment)])
        incr_idx = torch.isin(train_labels, cls_idx)
        train_label_per_increment.append(train_labels[incr_idx])
        train_data_per_increment.append(train_data[incr_idx]) 

    return train_data_per_increment, train_label_per_increment

def split_test_data(test_data, test_labels, n_increments):

  test_data_per_increment = []
  test_label_per_increment = []

  n_cls = len(torch.unique(test_labels))
  n_class_per_increment = n_cls // n_increments

  if n_cls % n_increments != 0:
        raise ValueError("n_increments should divide the number of classes in the dataset")

  for incr in range(1, n_increments+1):
    cls_idx = torch.tensor([i for i in range(n_class_per_increment * incr)])
    incr_idx = torch.isin(test_labels, cls_idx)
    test_label_per_increment.append(test_labels[incr_idx])
    test_data_per_increment.append(test_data[incr_idx])

  return test_data_per_increment, test_label_per_increment
