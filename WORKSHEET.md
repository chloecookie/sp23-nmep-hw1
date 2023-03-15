# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`YOUR GITHUB REPO HERE (or notice that you DMed us to share a private repo)`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

torch.nn.Module is the base class for all neural network modules. It allows you to define and organize parameters and layers of a neural network. It's used to create custom neural networks. It also includes a lot of built-in methods for managing the parameters. such as parameters(), zero_grad(), and .to(). 

torch.nn.functional is ma module that has pre-implemented functions such as convolution, pooling, activation functions, loss functions. It is a stateless function. 

torch.nn.Module is used for defining the architecture of a neural network , whereas torch.nn.functional already has pre-implemented functions that can be used in the forward pass. 

## -1.1 What is the difference between a Dataset and a DataLoader?

Dataset stores the data and the corresponding labels and DataLoader creates an iterable that wraps around the Dataset so you can easily manipulate the data. It also turns the training and testing data into minibatches. We want to have mini batches because it's more computationally efficient and we perform gradient descent more often so its faster to improve (once per minibatch instead of once per epoch). We usually set Dataset equal to a variable and then pass that into DataLoader. 

Dataset: 
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)


## -1.2 What does `@torch.no_grad()` above a function header do?

It temporarily sets all require_grad() flags to false. It reduces memory and speeds up computation. It's used to perform inference without gradient descent and ensure there's no leak test data in the model. 

# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

Build.py files contain the data loader builder and the model builder. 

## 0.1 Where would you define a new model?

We would define a new model by using the build_model function in models/build.py. 

## 0.2 How would you add support for a new dataset? What files would you need to change?

We would add support for a new dataset in data/datasets.py where dataset loaders are defined. 

## 0.3 Where is the actual training code?

The actual training code is in main.py. 

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

Diagram: https://imgur.com/a/cxC1Uh8

# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.


## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

Build_Loader configures different datasets depending on the dataset (cifar 10 or medium_imagenet). It also sets the definitions of Dataloaders for training, validating, and testing. 

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

3 functions for training, validating, and testing that are each an instance of the corresponding dataset that it is using (cifar10 or medium_imagenet). 

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

The field that actually contains the dataset is in data/datasets.py 

self.dataset = CIFAR10(root="/data/cifar10", train=self.train, download=True). We do need to download it ahead of time. 

### 1.1.1 What is `self.train`? What is `self.transform`?

If self.train is toggled true, then data augmentations are applied to the images. This is used during training to improve the model's accuracy. Some transformations are flipping, rotating, and resizing. 

Self.transforms is implemented if self.train is true. It applies the transformations such as color jittering, random horizontal flips, normalizing, and resizing. 

### 1.1.2 What does `__getitem__` do? What is `index`?

__getitem__ is how we index into an instance of self.dataset. It takes out the image and the label. Index is a specific sampled image and label of the dataset (e.g. index-th image). The image then passes through the transform function which applied transformations to the image. What's returned is the new transformed image and the label. 

### 1.1.3 What does `__len__` do?

__len__ returns the length of the dataset. 

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

self._get_transforms is the function that applies the transformations.  There is an if statement because only certain transform functions are applied if self.train is True (ColorJitter and RandomHorizontalFlip). During training, self.train might be set to True. During inference, self.train might be set to False because you're not training but you might still want to preprocess the data. 

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

transforms 

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

The field that actually contains the data is the filepath: str = "/data/medium-imagenet/medium-imagenet-nmep-96.hdf5",


> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

_get_transforms is different because the one in CIFAR10Dataset applies a set of image augmentations to the dateaset. _get_transforms in MediumImagenet normalizes and resizes for the base layer and then if a set of conditions is fulfilled, then random horizontal flipping and color jittering is applid. 

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

__getitem__ in CIFAR10 gets the specific image and label at that index and then transforms it. The transformed image and label is returned. __getitem__ in MediumImagenet allows you to read the data as the entire file contents are not stored in memory. self.split  is used to determine which dataset split (train, validation, or test) the item belongs to. If the split is not "test", the label data is also retrieved from the file using the same index and split.
It returns a tuple containing the transformed image data and the label data as a PyTorch tensor object. If the split is "test", the label is set to -1 because the test dataset may not have labels.

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.


def imshow(img):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()

def main():
  transform = transforms.Compose( [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
      (0.5, 0.5, 0.5))])

  dataiter = iter(trainloader)
  imgs, lbls = next(dataiter)

  for i in range(10):  
      plt.imshow(torchvision.utils.make_grid(imgs[i]))
  

main()

Parts of the code is from :https://github.com/sanghoon/pytorch_imagenet/blob/master/toy_cifar.py

# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

models/build.py implemends the function that builds the model. It specifies different build parameters if the model is LeNet vs if the model is ResNet18.

Lenet.py creates a class that defines LeNet that contains 32x32 color images and 200 classes. 

Resnet.py implements a class that defines a residual block for the ResNet architecture and another class called ResNet18. 

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

Pytorch models inherit from nn.module. We need to implement the functions init and forward. Init implements the attributes that the neural network needs, such as Conv2d and Sigmoid. Forward implements the forward pass and flattens the image so the data can be processed better. 

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

The configs that we have set a few parameters such as batch size, image size, the model, number of classes, drop rate, epoch start, number of epochs, learning rate, and others. They use the swin model whcih is a transformer and they use a dataset that is called CIFAR. 


## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

Main() is a function that first sets the device and then builds a model with the preloaded parameters from the config file. The optimizer is also built using the preloaded parameters from the config file. The loss function/criterion is set to CrossEntropyLoss. For every epoch in the range, the train_one_epoch and validate function is called. 

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

The validate function sets the model to evaluate mode from trainign mode. It computes the output for each image that you feed into the model, measures accuracy and loss, and also tracks the elapsed time the computation takes.The loss and accuracy values are then stored in loss_meter and acc1_meter, respectively.  The evaluate function sets the model to evaluate mode too and returns the predictions of the images in a list which is concantenated into a numpy array.


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

AlexNet has substantially more parameters than LeNet with 60M parametesr. LeNet has 60k parameters. 

For the memory, I'm guessing that the memory usage is the green column that says x / y MB but for some reason my username isn't popping up next ot GPU 0 after I train the LeNet model (but it did successfully train). 

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

`YOUR ANSWER HERE`

# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

https://wandb.ai/chloechia/SP23-HW1?workspace=user-chloechia

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

(https://wandb.ai/chloechia/SP23-HW1?workspace=user-chloechia)

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

Prioritized training the Resnet portion of this homework so didn't get to this. 

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.


Prioritized training the Resnet portion of this homework so didn't get to this. 

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 


Prioritized training the Resnet portion of this homework so didn't get to this. 

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.


Prioritized training the Resnet portion of this homework so didn't get to this. 

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

Prioritized training the Resnet portion of this homework so didn't get to this. 

# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/models.py`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

ResNet 18: 70~% accuracy 

https://wandb.ai/chloechia/SP23-HW1?workspace=user-chloechia

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉
