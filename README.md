# Perturbative Neural Networks (PNN)
This is an attempt to reproduce results in Perturbative Neural Networks paper.
See original repo for details: https://github.com/juefeix/pnn.pytorch

##Results:
CIFAR-10:
ResNet18 (nfilters=128): 94% 
PNN using ResNet18 structure (nfilters=128): 79%
PNN using ResNet18 structure (nfilters=128) with the regular 3x3 convolutional first layer: 89%


## Implementation details
Three main changes from the original code:
1. Test accuracy in the original code was calculated incorrectly. As a result, their reported accuracies are much better than they really are. 
2. The original code used regular convolutional layer as the first layer in the model. 
3. The original code didn't implement fanout (number of noise masks per input channel)

This implementation uses 3 main arguments to setup model configuration:
--first_filter_size  zero (0) value for this argument converts the first layer of the model into a perturbation layer (addition of noise masks followed by 1x1 convolution), any other value will make this layer a regular convolutional layer with this filter size. 
--filter_size same as above, used for all layers except the first one
--nmasks this value is number of noise masks per input channel, if this value is 1, there will be as many noise masks as input channels. if it is more than one, --group argument can be used to perform perturbations of each input channel independently (just like described in the paper), if no --group argument is used, nmasks*nfilters noise masks will be used in all layers except the first one. The first layer will by default apply nfilters noise masks to each input channel before applying 1x1 convolution.
--use_act controls whether to apply activation function in the first layer after the perturbation and before 1x1 convolution

### Training Recipes
Pure PNN with fanout 1:
```
python main.py --net-type 'resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-
rate 1e-4 --first_filter_size 0 --filter_size 0 --nmasks 1 --use_act
```

PNN with the first layer using 3x3 convolution:
```
python main.py --net-type 'resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-
rate 1e-4 --first_filter_size 3 --filter_size 0 --nmasks 1 --use_act
```

Regular CNN (ResNet18):
```
python main.py --net-type 'resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-
rate 1e-4 --first_filter_size 3 --filter_size 3 --nmasks 1 --use_act
```

### Requirements
PyTorch 0.4.1
