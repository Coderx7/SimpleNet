## Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures (2016).

![GitHub Logo](/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg)

This repository contains the architectures, Models, logs, etc pertaining to the SimpleNet Paper 
(Lets keep it simple: Using simple architectures to outperform deeper architectures ) : https://arxiv.org/abs/1608.06037       
(Check the successor of this architecture at [Towards Principled Design of Deep Convolutional Networks: Introducing SimpNet](https://github.com/Coderx7/SimpNet))

SimpleNet-V1 outperforms deeper and heavier architectures such as AlexNet, VGGNet,ResNet,GoogleNet,etc in a series of benchmark datasets, such as CIFAR10/100, MNIST, SVHN. 
It also achievs a higher accuracy (currently [71.14/89.75](https://github.com/Coderx7/SimpleNet_Pytorch#imagenet-result)) in imagenet, more than VGGNet, ResNet, MobileNet, AlexNet, NIN, Squeezenet, etc with only 5.7M parameters.
Slimer versions of the architecture work very decently against more complex architectures such as ResNet and WRN as well.

## Citation
If you find SimpleNet useful in your research, please consider citing:

    @article{hasanpour2016lets,
      title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
      author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
      journal={arXiv preprint arXiv:1608.06037},
      year={2016}
    }

## Other Implementations : 

**Pytorch** :
For using Pytorch implemnetation click [Pytorch implementation](https://github.com/Coderx7/SimpleNet_Pytorch) 


## Results Overview :
ImageNet result was achieved using simple SGD without hyper parameter tuning for 100 epochs(single crop). no multicrop techniques were used. no dense evaluation or combinations of such techniques were used unlike all other architectures. the models will be uploaded when the training is finished. 

| Dataset | Accuracy |
|------------|----------|
| Cifar10    | **95.51** |
| CIFAR100*  | **78.37**|
| MNIST      | 99.75    |
| SVHN       | 98.21    |
| ImageNet   | **71.14/89.75**  |

* Achieved using Pytorch implementation 

#### Top CIFAR10/100 results:


| **Method**                   | **\#Params** |  **CIFAR10**  | **CIFAR100** |
| :--------------------------- | :----------: | :-----------: | :----------: |
| VGGNet(16L) /Enhanced        |     138m     | 91.4 / 92.45  |      \-      |
| ResNet-110L / 1202L  \*      |  1.7/10.2m   | 93.57 / 92.07 | 74.84/72.18  |
| SD-110L / 1202L              |  1.7/10.2m   | 94.77 / 95.09 |  75.42 / -   |
| WRN-(16/8)/(28/10)           |    11/36m    | 95.19 / 95.83 |  77.11/79.5  |
| Highway Network              |     N/A      |     92.40     |    67.76     |
| FitNet                       |      1M      |     91.61     |    64.96     |
| FMP\* (1 tests)              |     12M      |     95.50     |    73.61     |
| Max-out(k=2)                 |      6M      |     90.62     |    65.46     |
| Network in Network           |      1M      |     91.19     |    64.32     |
| DSN                          |      1M      |     92.03     |    65.43     |
| Max-out NIN                  |      \-      |     93.25     |    71.14     |
| LSUV                         |     N/A      |     94.16     |     N/A      |
| SimpleNet-Arch 1\(۞)        |    5.48M     |   **94.75**   |      \-      |
| SimpleNet-Arch 2 \(۩) |    5.48M     |   **95.51**   |  **78.37**   |

\*Note that the Fractional max pooling[13] uses deeper architectures and also uses extreme data augmentation.۞  means No zero-padding or normalization with dropout and ۩ means Standard data-augmentation- with dropout. To our knowledge, our architecture has the state of the art result, without aforementioned data-augmentations.


#### MNIST results:

| **Method**                                   | **Error rate** |
| :------------------------------------------- | :------------: |
| DropConnect\*\*                              |     0.21%      |
| Multi-column DNN for Image Classiﬁcation\*\* |     0.23%      |
| APAC\*\*                                     |     0.23%      |
| Generalizing Pooling Functions in CNN\*\*    |     0.29%      |
| Fractional Max-Pooling\*\*                   |     0.32%      |
| Batch-normalized Max-out NIN                 |     0.24%      |
| Max-out network (k=2)                        |     0.45%      |
| Network In Network                           |     0.45%      |
| Deeply Supervised Network                    |     0.39%      |
| RCNN-96                                      |     0.31%      |
| **SimpleNet \***                             |   **0.25%**    |

\*Note that we didn’t intend on achieving the state of the art
performance here as we are using a single optimization policy without
fine-tuning hyper parameters or data-augmentation for a specific task,
and still we nearly achieved state-of-the-art on MNIST. \*\*Results
achieved using an ensemble or extreme data-augmentation

#### Top SVHN results:

| **Method**                   | **Error rate** |
| :--------------------------- | :------------: |
| Network in Network           |      2.35      |
| Deeply Supervised Net        |      1.92      |
| ResNet (reported by  (2016)) |      2.01      |
| ResNet with Stochastic Depth |      1.75      |
| Wide ResNet                  |      1.64      |
| **SimpleNet**                |    **1.79**    |



#### ImageNet2012 results:

| **Method**         | **T1/T5 Accuracy Rate** |
| :----------------- | :---------------------: |
| AlexNet(60M)       |        57.2/80.3        |
| VGGNet16(138M)     |          70.5           |
| GoogleNet(8M)      |          68.7           |
| Wide ResNet(11.7M) |       69.6/89.07        |
| SimpleNet(5.4M)    |     **67.17/87.44**     |


Table 6-Slimmed version Results on Different Datasets

| **Model** | **Ours** | **Maxout** | **DSN** | **ALLCNN** | **dasNet** | **ResNet(32)** | **WRN** | **NIN** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **#Param** | **310K** | **460K** | **6M** | **1M** | **1.3M** | **6M** | **475K** | **600K** | **1M** |
| **CIFAR10** | **91.98** | **92.33** | **90.62** | **92.03** | **92.75** | **90.78** | **91.6** | **93.15** | **91.19** |
| **CIFAR100** | **64.68**|**66.82**|**65.46**|**65.43**|**66.29**|**66.22**|**67.37**|**69.11**|**-** |

| Other datasets | Our result |
| --- | --- |
| MNIST(310K)\* | **99.72** |
| SVHN(310K)\* | **97.63** |

\*Since we presented their results in their respective sections, we avoided mentioning the results here again. 



#### Cifar10 extended results:

| **Method**                              | **Accuracy** | **\#Params** |
| :-------------------------------------- | :----------: | :----------: |
| VGGNet(16L)                             |     91.4     |     138m     |
| VGGNET(Enhanced-16L)\*                  |    92.45     |     138m     |
| ResNet-110\*                            |    93.57     |     1.7m     |
| ResNet-1202                             |    92.07     |    10.2m     |
| Stochastic depth-110L                   |    94.77     |     1.7m     |
| Stochastic depth-1202L                  |    95.09     |    10.2m     |
| Wide Residual Net                       |    95.19     |     11m      |
| Wide Residual Net                       |    95.83     |     36m      |
| Highway Network                         |    92.40     |      \-      |
| FitNet                                  |    91.61     |      1M      |
| SqueezNet-(tested by us)                |    79.58     |     1.3M     |
| ALLCNN                                  |    92.75     |      \-      |
| Fractional Max-pooling\* (1 tests)      |    95.50     |     12M      |
| Max-out(k=2)                            |    90.62     |      6M      |
| Network in Network                      |    91.19     |      1M      |
| Deeply Supervised Network               |    92.03     |      1M      |
| Batch normalized Max-out NIN            |    93.25     |      \-      |
| All you need is a good init (LSUV)      |    94.16     |      \-      |
| Generalizing Pooling Functions in CNN   |    93.95     |      \-      |
| Spatially-Sparse CNNs                   |    93.72     |      \-      |
|                                         |    93.63     |      \-      |
| Recurrent CNN for Object Recognition    |    92.91     |      \-      |
| RCNN-160                                |    92.91     |      \-      |
| SimpleNet-Arch1                         |    94.75     |     5.4m     |
| SimpleNet-Arch1 using data augmentation |    95.32     |     5.4m     |

#### CIFAR100 Extended results:

| **Method**                                | **Accuracy** |
| :---------------------------------------- | :----------: |
| GoogleNet with ELU\*                      |    75.72     |
| Spatially-sparse CNNs                     |     75.7     |
| Fractional Max-Pooling(12M)               |    73.61     |
| Scalable Bayesian Optimization Using DNNs |    72.60     |
| All you need is a good init               |    72.34     |
| Batch-normalized Max-out NIN(k=5)         |    71.14     |
| Network in Network                        |    64.32     |
| Deeply Supervised Network                 |    65.43     |
| ResNet-110L                               |    74.84     |
| ResNet-1202L                              |    72.18     |
| WRN                                       |  77.11/79.5  |
| Highway                                   |    67.76     |
| FitNet                                    |    64.96     |
| SimpleNet                                 |    77.83     |



** Achieved using several data-augmentation tricks

**Flops and Parameter Comparison:**
<span id="tab:Flops_appndx" label="tab:Flops_appndx">\[tab:Flops\_appndx\]</span>

| **Model**      | **MACC** | **COMP** | **ADD** | **DIV** | **Activations** | **Params** | **SIZE(MB)** |
| :------------- | :------: | :------: | :-----: | :-----: | :-------------: | :--------: | :----------: |
| SimpleNet      |   1.9G   |  1.82M   |  1.5M   |  1.5M   |      6.38M      |    6.4M    |     24.4     |
| SqueezeNet     | 861.34M  |  9.67M   |  226K   |  1.51M  |     12.58M      |   1.25M    |     4.7      |
| Inception v4\* |  12.27G  |  21.87M  | 53.42M  | 15.09M  |     72.56M      |   42.71M   |     163      |
| Inception v3\* |  5.72G   |  16.53M  | 25.94M  |  8.97M  |     41.33M      |   23.83M   |      91      |
| Incep-Resv2\*  |  13.18G  |  31.57M  | 38.81M  | 25.06M  |     117.8M      |   55.97M   |     214      |
| ResNet-152     |  11.3G   |  22.33M  | 35.27M  | 22.03M  |     100.11M     |   60.19M   |     230      |
| ResNet-50      |  3.87G   |  10.89M  | 16.21M  | 10.59M  |     46.72M      |   25.56M   |    97.70     |
| AlexNet        |  7.27G   |  17.69M  |  4.78M  |  9.55M  |     20.81M      |   60.97M   |    217.00    |
| GoogleNet      |  16.04G  | 161.07M  |  8.83M  | 16.64M  |     102.19M     |     7M     |      40      |
| NIN            |  11.06G  |  28.93M  |  380K   |   20K   |     38.79M      |    7.6M    |      29      |
| VGG16          |  154.7G  | 196.85M  |   10K   |   10K   |     288.03M     |  138.36M   |    512.2     |

Flops and Parameter Comparison of Models trained on ImageNet

    
\*Inception v3, v4 did not have any Caffe model, so we reported their
size related information from MXNet and Tensorflow respectively.
Inception-ResNet-V2 would take 60 days of training with 2 Titan X to
achieve the reported accuracy. Statistics are obtained using
<http://dgschwend.github.io/netscope>


1# Data-augmentation method used by stochastic depth paper: [https://github.com/Pastromhaug/caffe-stochastic-depth](https://github.com/Pastromhaug/caffe-stochastic-depth).

2# https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md

3# https://github.com/tensorflow/models/tree/master/slim

4# https://github.com/revilokeb/inception\_resnetv2\_caffe


