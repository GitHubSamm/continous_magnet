import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resize_data import resize_dataset

import create_model
from learning_function import learning_function
from config import Facnet_config


##########################################################################################
## ------------------ Import train/test data --------------##
##########################################################################################

print("Importing data...")
# Define a transform (here no resizing is needed for 32x32 images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the CIFAR-10 training and test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

images_train, labels_train = resize_dataset(train_dataset, 2000, 10)
images_test, labels_test   = resize_dataset(test_dataset, 400, 10)

print(f"images_train size: {images_train.shape}")
print(f"labels_train size: {labels_train.shape}")
print(f"images_test size: {images_test.shape}")
print(f"labels_test size: {labels_test.shape}")


##########################################################################################
## ------------------------------------- Creating the model-----------------------------##
##########################################################################################
learning_rate       = Facnet_config['learning_rate']
width               = Facnet_config['width']

model     = create_model.WRN2(width)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device=device, dtype=torch.float)
model.eval()

##########################################################################################
## ---------------------------------- Creating the optimizer----------------------------##
##########################################################################################
optimizer     = torch.optim.Adam(model.parameters(),lr = learning_rate)

learning_function(model,optimizer,device,images_train, labels_train,images_test,labels_test)
