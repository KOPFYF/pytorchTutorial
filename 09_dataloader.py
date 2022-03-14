'''
https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks
In the neural network terminology:

one epoch = one forward pass and one backward pass of all the training examples
batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
For example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

for each epoch, iterations(n) = training examples / batch size
it means for 1 epoch, we need n iterations, with such batch size to go through the whole dataset once.



# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into **small batches**

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# **e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = WineDataset()

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print('sample: (features, labels)', features, labels)

'''
https://pytorch.org/docs/stable/data.html

At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class. 
It represents a Python iterable over a dataset, with support for

-map-style and iterable-style datasets,
-customizing data loading order,
-automatic batching,
-single- and multi-process data loading,
-automatic memory pinning.

DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
'''

if __name__ == '__main__':
    # Load whole dataset with DataLoader
    # shuffle: shuffle data, good for training
    # num_workers: faster loading with multiple subprocesses
    # !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
    train_loader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=0)

    # convert to an iterator and look at one random sample
    dataiter = iter(train_loader)
    data = dataiter.next()
    features, labels = data
    print(features, labels)

    # Dummy Training loop
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/4)
    print('total_samples, n_iterations:', total_samples, n_iterations)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            
            # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
            # Run your training process
            if (i+1) % 5 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

    # some famous datasets are available in torchvision.datasets
    # e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=torchvision.transforms.ToTensor(),  
                                            download=True)

    train_loader = DataLoader(dataset=train_dataset, 
                                            batch_size=3, 
                                            shuffle=True)

    # look at one random sample
    dataiter = iter(train_loader)
    data = dataiter.next()
    inputs, targets = data
    print(inputs.shape, targets.shape) # torch.Size([3, 1, 28, 28]) torch.Size([3])
