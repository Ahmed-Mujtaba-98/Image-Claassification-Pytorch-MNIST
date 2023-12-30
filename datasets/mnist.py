import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

def mnist(batchsize=4, numworkers=2):
    """
    Load and prepare the MNIST dataset.

    Parameters:
    - batchsize (int): Number of images in each batch.
    - numworkers (int): Number of worker processes for data loading.

    Returns:
    - trainloader (torch.utils.data.DataLoader): DataLoader for the training set.
    - testloader (torch.utils.data.DataLoader): DataLoader for the test set.
    """

    # Define transformations for the training data and testing data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

    # Download and load the test data
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=numworkers)

    return trainloader, testloader

def fashion_mnist(batchsize=4):
   # Define a transform to normalize the data
   transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])
   
   # Download and load the training data
   trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
   train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

   # Download and load the test data
   testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
   test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True)

   return train_loader, test_loader