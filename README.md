# 🏝️ IMAGE CLASSIFICATION USING PYTORCH 🔥️ - MNIST DATASET

**About MNIST**

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. It was created by "re-mixing" the samples from NIST's original datasets. The database contains 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image of a digit between 0 and 9 1. The MNIST database is widely used in machine learning research for benchmarking algorithms.

**About Fashion MNIST**

The Fashion MNIST dataset is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. These classes represent different articles of clothing, including T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

## ✍️ Prerequisites
- You should have anaconda package already installed on your host PC. If not, visit the [Anaconda official site](https://www.anaconda.com/download) to download and install the latest package.

## 👨‍🔧️ Environment Setup
Clone this repository by running the following commands on the terminal,

```
git clone <this_repository>
cd <this_repository>
```
Setup conda environment,

```
conda create -n my_env python=3.10
conda activate my_env
pip install -r requirements.txt
```

## 📌️ List of Supported Models and Datasets
List of models supported for training are,

    choices=[
        'Net',
        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
        'EfficientNetB0',
        'GoogLeNet',
        'LeNet',
        'MobileNet', 'MobileNetV2',
        'RegNetX_200MF', 'RegNetX_400MF', 'RegNetY_400MF',
        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
        'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
        'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG_mnist',
    ]

> "Net" is a simple CNN with few convolution and fully-connected layers. You can find all models architecture [here](./models/).

List of dataset choices supported for training are,

    choices=[
        'mnist',
        'fashion_mnist',
    ]

## 🌴️ Model Training
To starting training the models, run the below command,

```
python train.py --epochs 10 --dataset mnist --model Net
```

To save best model during training, run the below command,

```
python train.py --epochs 10 --dataset mnist --model Net --save_model
```

To resume training from saved checkpoint, run the below command,

```
python train.py --epochs 10 --dataset mnist --model Net --resume
```

### 📃️ Arguments Description

- **dataset**: Dataset to train, default='mnist'
- **epochs**: Total number of epochs, default=2
- **lr**: Learning rate for model trainig, default=0.001
- **batch_size**: Batch size for dataloader, default=4
- **num_workers**: Total number of workers, default=2
- **net_size**: Net size for shufflenetv2, default=2
- **save_model**: Save best model checkpoint
- **resume**: Resume training from checkpoint


## 🌍️ References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## 📢️ Other Repositories and Contribute
- Checkout this repository for image classification on CIFAR10 and CIFAR100 dataset.
- Feel free to contribute and create a pull request to add additional features. Also, open issues if you face any difficulty, I will happy to assist you in solving the problems.