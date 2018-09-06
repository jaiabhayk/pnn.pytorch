# Perturbative Neural Networks (PNN)
This is an attempt to reproduce results in Perturbative Neural Networks paper.
See [original repo](https://github.com/juefeix/pnn.pytorch) for details.

## Motivation
The original implementation used regular convolutions in the first layer, and the remaining layers used fanout of 1, which means each input channel was perturbed with a single noise mask. 

However, the biggest issue with the original implementation is that test accuracy was calculated incorrectly. Instead of the usual method of calculating ratio of correct samples to total samples in the test dataset, the authors calculated accuracy on per batch basis, and applied smoothing weight (test_accuracy = 0.7 * prev_batch_accuracy + 0.3 * current_batch_accuracy). 

Here's how [this method](https://github.com/juefeix/pnn.pytorch/blob/master/plugins/monitor.py#L31) (reported) compares to the [proper accuracy calculation](https://github.com/michaelklachko/pnn.pytorch/blob/master/main.py#L226-L230) (actual):

![img](https://s15.postimg.cc/vta2ku9nv/image.png)

For this model run (noiseresnet18 on CIFAR10), the code in original repo would report test accuracy 90.53%, while the actual test accuracy is 85.91%

After correcting this issue, I ran large number of experiments trying to see if perturbing input with noise masks would provide any benefit, and my conclusion is that it does not. 

## Implementation details
Most of the modifications are contained in the [PerturbLayer class.](https://github.com/michaelklachko/pnn.pytorch/blob/master/models.py#L15) Here are the main changes from the original code:

`--first_filter_size` and `--filter_size` arguments control the type of the layer. A value of 0 turns the layer into a perturbation layer, as described in the paper. Any value n > 0 will turn the layer into a regular convolutional layer with filter size n.

`--nmasks` specifies number of noise masks to apply to each input channel. This is "fanout" parameter mentioned in the paper. The original implementation used nmasks=1 hardcoded.

`--unique_masks` specifies whether to use different sets of nmasks noise masks for each input channel. --no-unique_masks forces the same set of nmasks to be used for all input channels.

`--train_masks` enables treating noise masks as regular parameters, and optimizes their values during training at the same time as model weights.

Other arguments allow changing noise type (uniform or normal), pooling type (max or avg), activation function (relu, rrelu, prelu, elu, selu, tanh, sigmoid), whether to apply activation function in the first layer (--use_act, immediately after perturbing the input RGB channels, this results in some information loss), whether to scale noise level in the first layer, and --debug argument prints out values of input, noise, and output for every update step to verify that noise is being applied correctly.

Three different models are supported: `--net_type perturb_resnet18`, `--net_type cifarnet` (6 conv layers, followed by a fully connected layer), and `--net_type lenet` (3 conv. layers followed by a fully connected layer). In addition, I included the baseline ResNet-18 model `--net-type resnet18` taken from [here](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py), and `--net_type noiseresnet18` model from the original repo. Note that `perturb_resnet18` model is flexible enough to replace both baseline and `noiseresnet18` models, using appropriate arguments.

## Results
CIFAR-10:

1. Baseline (regular ResNet18 with 3x3 convolutions) Test Accuracy: 94.2% 
```
python main.py --net-type 'resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 10 --learning-rate 1e-3
```

2. Original implementation (this is equivalent to using the code in the original repo). Test Accuracy: 85.91%
```
python main.py --net-type 'noiseresnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 10 --learning-rate 1e-4 --first_filter_size 7
```

3. Same as above, but changing `first_filter_size` argument to 3 improves the accuracy to 86.21%

4. PNN with all uniform noise in all layers (including the first layer). Test Accuracy: 74.6%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 0 --filter_size 0 --nmasks 1 
```

5. Same as above, but without any noise (this is equivalent to regular CNN with all convolutions changed to 1x1). Test Accuracy: 74.3%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 1 --filter_size 1 
```

6. PNN with noise masks in all layers except the first layer, which uses regular 3x3 convolutions. Test Accuracy: 85.8%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 3 --filter_size 0 --nmasks 1 
```

7. Same as above, but without any noise (regular CNN with all 3x3 convolutions in the first layer, and 1x1 in remaining layers). Test Accuracy: 85.5%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 3 --filter_size 1 
```

8. PNN with noise masks in all layers except the first layer, which uses regular 3x3 convolutions with fanout=64. Internally fanout is implemented with grouped 1x1 convolutions. Note --unique_masks arg, which creates unique set of masks for each input channel, in every layer. Test Accuracy: 84.7%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 3 --filter_size 0 --nmasks 64 --unique_masks
```

9. Same as above, but with --no-unique_masks argument, which means that the same set of masks is used for each input channel. Test Accuracy: 84.4%
```
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 3 --filter_size 0 --nmasks 64 --no-unique_masks
```

Experiments 8 and 9 are the closest to what was described in the paper.

10. Finally, I tried to train the noise masks (they are being updated each batch, at the same time as regular convolution weights). Test Accuracy: 85.9%

`python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 128 --batch-size 16 --learning-rate 1e-3 --first_filter_size 3 --filter_size 0 --nmasks 64 --no-unique_masks --train_masks`


## Conclusion
Unfortunately, it appears that perturbing layer inputs with noise does not provide any significant benefit. Simple 1x1 convolutions without noise masks provide similar performance. No matter how we apply noise masks, the accuracy drop resulting from using 1x1 filters is severe (~5% on CIFAR-10 even when not modifying the first layer). If we modify the first layer, the accuracy drops by ~20%. 

