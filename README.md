SimpleNet: Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures. 

This repository contains the architectures, Models, logs, etc pertaining to the SimpleNet Paper 
(Lets keep it simple: Using simple architectures to outperform deeper architectures ) : https://arxiv.org/abs/1608.06037 

SimpleNet-V1 outperforms deeper and heavier architectures such as AlexNet, VGGNet,ResNet,GoogleNet,etc in a series of benchmark datasets, such as CIFAR10/100, MNIST, SVHN. 
It also achievs a higher accuracy (currently 60.97/83.54) in imagenet, more than AlexNet, NIN, Squeezenet, etc with only 5.4M parameters.
Slimer versions of the architecture work very decently against more complex architectures such as ResNet and WRN as well.

The files are being uploaded/updated.  


#### Results Overview :
ImageNet result was achieved using single-scale training(256x256 input). no multiscale, multicrop techniques were used. no dense evaluation or combinations of such techniques were used unlike all other architectures. 

| Dataset | Accuracy |
|------------|----------|
| Cifar10    | 94.75/95.32    |
| CIFAR100   | 73.42/74.86        |
| MNIST      | 99.75        |
| SVHN       | 98.21       |
| ImageNet   | 60.97/83.54        |

#### Comparison with other architectures 
Table 1 showing different architectures statistics

| **Model** | **AlexNet** | **GoogleNet** | **ResNet152** | **VGGNet16** | **NIN** | **Ours** |
| --- | --- | --- | --- | --- | --- | --- |
| **#Param** | 60M | 7M | 60M | 138M | 7.6M | 5.4M |
| **#OP** | 1140M | 1600M | 11300M | 15740M | 1100M | 652M |
| **Storage (MB)** | 217 | 51 | 230 | 512.24 | 29 | 20 |





Table 2 showing Top CIFAR10/100 results

| Method | #Params | CIFAR10 | CIFAR100 |
| --- | --- | --- | --- |
| VGGNet(original 16L)[49] Enhanced | 138m | 91.4 / 92.45 | - |
| ResNet-110L [4] / 1202L[4] \* | 1.7m/10.2m | 93.57 / 92.07 | 74.84 / 72.18 |
| Stochastic depth-110L [39] / 1202L[39] | 1.7m/10.2m | 94.77 / 95.09 | 75.42 / - |
| Wide Residual Net-(16/8)[31] / (28/10)[31] | 11m/36m | 95.19 / 95.83 | 77.11 / 79.5 |
| Highway Network [30] | - | 92.40 | 67.76 |
| FitNet [43]| 1M | 91.61 | 64.96 |
| Fractional Max-pooling\* (1 tests)[13]** | 12M | 95.50 | 73.61 |
| Max-out(k=2)[12] | 6M | 90.62 | 65.46 |
| Network in Network [38] | 1M | 91.19 | 64.32 |
| Deeply Supervised Network[50] | 1M | 92.03 | 65.43 |
| Max-out Network In Network [51] | - | 93.25 | 71.14 |
| All you need is a good init (LSUV)[52] | - | 94.16 | - |
| Our Architecture۞   | **5.48m** | **94.75** | **-** |
|Our Architecture ۩1 | **5.48m** | **95.32** | **73.42** |

\*Note that the Fractional max pooling[13] uses deeper architectures and also uses extreme data augmentation.۞  means No zero-padding or normalization with dropout and۩means Standard data-augmentation- with dropout. To our knowledge, our architecture has the state of the art result, without aforementioned data-augmentations.


Table 3 showing MNIST results

| Method | Error rate |
| --- | --- |
| **Regularization of Neural Networks using DropConnect** [16]\*\* | 0.21% |
| **Multi-column Deep Neural Networks for Image Classiﬁcation** [53]\*\* | 0.23% |
| **APAC: Augmented Pattern Classification with Neural Networks** [54]\*\* | 0.23% |
| **Batch-normalized Max-out Network in Network** [51] | 0.24% |
| **Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree [55] \*\* | 0.29% |
| **Fractional Max-Pooling [13]**\*\* | 0.32% |
| **Max-out network (k=2) [12]** | 0.45% |
| **Network In Network [38]** | 0.45% |
| **Deeply Supervised Network[50]** | 0.39% |
| **RCNN-96 [56]** | 0.31% |
| Our architecture \* | **0.25%** |

\*Note that we didn&#39;t intend on achieving the state of the art performance here, as we already pointed out in the prior sections, we wanted to experiment if the architecture does perform nicely without any changes

\*\*Achieved using an ensemble or extreme data-augmentation

Table 4 showing SVHN results

| Method | Error |
| --- | --- |
| **Network in Network** | **2.35** |
| Deeply Supervised Net | **1.92** |
| ResNet (reported by Huang et al. (2016)) | **2.01** |
| ResNet with Stochastic Depth | **1.75** |
| Wide ResNet | **1.64** |
| Our architecture | **1.79** |



Table 5 showing ImageNet2012 results

| Method | T1/T5 |
| --- | --- |
| AlexNet(60M) | 57.2/80.3 |
| Squeezenet(1.3M) | 57.5/80.3 |
| Network in Network(7.5M) | 59.36/- |
| VGGNet16(138M) | 70.5 |
| GoogleNet(8M) | 68.7 |
| Wide ResNet(11.7M) | 69.6/89.07 |
| Our architecture\* | **60.97/83.54** |

                         \*Trained only for 300K (still training)









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
| ImageNet(310K) \*\* | **37.34/63.4** |





\*Since we presented their results in their respective sections, we avoided mentioning the results here again. \*\*ImageNet result belongs to the latest iteration of a still ongoing training.





Cifar10 extended results:

| Method | Accuracy | #Params |
| --- | --- | --- |
| **VGGNet(original 16L)****[49]** | 91.4 | 138m |
| **VGGNET(Enhanced)****[49] ****\*** | 92.45 | 138m |
| **ResNet-110**** [4]**\* | 93.57 | 1.7m |
| **ResNet-1202**** [4]** | 92.07 | 10.2 |
| **Stochastic depth-110L**** [39]** | 94.77 | 1.7m |
| **Stochastic depth-1202L**** [39]** | 95.09 | 10.2m |
| **Wide Residual Net**** [31]** | 95.19 | 11m |
| **Wide Residual Net**** [31]** | 95.83 | 36m |
| **Highway Network**** [30]** | 92.40 | - |
| **FitNet**** [43]** | 91.61 | 1M |
| **SqueezNet**** [45]****-(tested by us)** | 79.58 | 1.3M |
| **ALLCNN**** [44]** | 92.75 | - |
| **Fractional Max-pooling\* (1 tests)****[13]** | 95.50 | 12M |
| **Max-out(k=2)****[12]** | 90.62 | 6M |
| **Network in Network**** [38]** | 91.19 | 1M |
| **Deeply Supervised Network**** [50]** | 92.03 | 1M |
| **Batch normalized Max-out Network In Network**** [51]** | 93.25 | - |
| **All you need is a good init (LSUV)****[52]** | 94.16 | - |
| **Generalizing Pooling Functions in Convolutional Neural Networks:** **[55]** | 93.95 | - |
| **Spatially-sparse convolutional neural networks**** [59]** | 93.72 | - |
| **Scalable Bayesian Optimization Using Deep Neural Networks**** [60]** | 93.63 | - |
| **Recurrent Convolutional Neural Network for Object Recognition**** [57]** | 92.91 | - |
| **RCNN-160**** [56]** | 92.91 | - |
| Our Architecture1 | **94.75** | **5.4** |
| Our Architecture1 using data augmentation | **95.32** | **5.4** |

CIFAR100 Extended results:

| Method | Accuracy |
| --- | --- |
| **GoogleNet with ELU**** [19]**\* | 75.72 |
| **Spatially-sparse convolutional neural networks**** [59]** | 75.7 |
| **Fractional Max-Pooling(12M)** **[13]** | 73.61 |
| **Scalable Bayesian Optimization Using Deep Neural Networks**** [60]** | 72.60 |
| **All you need is a good init**** [52]** | 72.34 |
| **Batch-normalized Max-out Network In Network(k=5)****[51]** | 71.14 |
| **Network in Network**** [38]** | 64.32 |
| **Deeply Supervised Network**** [50]** | 65.43 |
| **ResNet-110L**** [4]** | 74.84 |
| **ResNet-1202L**** [4]** | 72.18 |
| **WRN**** [31]** | 77.11/79.5 |
| **Highway**** [30]** | 67.76 |
| **FitNet**** [43]** | 64.96 |
| Our Architecture 1 | **73.45** |
| Our Architecture 2 | **74.86** |

** Achieved using several data-augmentation tricks

**Flops and Parameter Comparison:**
<table>
<caption>Flops and Parameter Comparison of Models trained on ImageNet</caption>
<thead>
<tr class="header">
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>MACC</strong></th>
<th style="text-align: center;"><strong>COMP</strong></th>
<th style="text-align: center;"><strong>ADD</strong></th>
<th style="text-align: center;"><strong>DIV</strong></th>
<th style="text-align: center;"><strong>Activations</strong></th>
<th style="text-align: center;"><strong>Params</strong></th>
<th style="text-align: center;"><strong>SIZE(MB)</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">SimpleNet</td>
<td style="text-align: center;">1.9G</td>
<td style="text-align: center;">1.82M</td>
<td style="text-align: center;">1.5M</td>
<td style="text-align: center;">1.5M</td>
<td style="text-align: center;">6.38M</td>
<td style="text-align: center;">6.4M</td>
<td style="text-align: center;">24.4</td>
</tr>
<tr class="even">
<td style="text-align: left;">SqueezeNet</td>
<td style="text-align: center;">861.34M</td>
<td style="text-align: center;">9.67M</td>
<td style="text-align: center;">226K</td>
<td style="text-align: center;">1.51M</td>
<td style="text-align: center;">12.58M</td>
<td style="text-align: center;">1.25M</td>
<td style="text-align: center;">4.7</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Inception v4*</td>
<td style="text-align: center;">12.27G</td>
<td style="text-align: center;">21.87M</td>
<td style="text-align: center;">53.42M</td>
<td style="text-align: center;">15.09M</td>
<td style="text-align: center;">72.56M</td>
<td style="text-align: center;">42.71M</td>
<td style="text-align: center;">163</td>
</tr>
<tr class="even">
<td style="text-align: left;">Inception v3*</td>
<td style="text-align: center;">5.72G</td>
<td style="text-align: center;">16.53M</td>
<td style="text-align: center;">25.94M</td>
<td style="text-align: center;">8.97M</td>
<td style="text-align: center;">41.33M</td>
<td style="text-align: center;">23.83M</td>
<td style="text-align: center;">91</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Incep-Resv2*</td>
<td style="text-align: center;">13.18G</td>
<td style="text-align: center;">31.57M</td>
<td style="text-align: center;">38.81M</td>
<td style="text-align: center;">25.06M</td>
<td style="text-align: center;">117.8M</td>
<td style="text-align: center;">55.97M</td>
<td style="text-align: center;">214</td>
</tr>
<tr class="even">
<td style="text-align: left;">ResNet-152</td>
<td style="text-align: center;">11.3G</td>
<td style="text-align: center;">22.33M</td>
<td style="text-align: center;">35.27M</td>
<td style="text-align: center;">22.03M</td>
<td style="text-align: center;">100.11M</td>
<td style="text-align: center;">60.19M</td>
<td style="text-align: center;">230</td>
</tr>
<tr class="odd">
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: center;">3.87G</td>
<td style="text-align: center;">10.89M</td>
<td style="text-align: center;">16.21M</td>
<td style="text-align: center;">10.59M</td>
<td style="text-align: center;">46.72M</td>
<td style="text-align: center;">25.56M</td>
<td style="text-align: center;">97.70</td>
</tr>
<tr class="even">
<td style="text-align: left;">AlexNet</td>
<td style="text-align: center;">7.27G</td>
<td style="text-align: center;">17.69M</td>
<td style="text-align: center;">4.78M</td>
<td style="text-align: center;">9.55M</td>
<td style="text-align: center;">20.81M</td>
<td style="text-align: center;">60.97M</td>
<td style="text-align: center;">217.00</td>
</tr>
<tr class="odd">
<td style="text-align: left;">GoogleNet</td>
<td style="text-align: center;">16.04G</td>
<td style="text-align: center;">161.07M</td>
<td style="text-align: center;">8.83M</td>
<td style="text-align: center;">16.64M</td>
<td style="text-align: center;">102.19M</td>
<td style="text-align: center;">7M</td>
<td style="text-align: center;">40</td>
</tr>
<tr class="even">
<td style="text-align: left;">NIN</td>
<td style="text-align: center;">11.06G</td>
<td style="text-align: center;">28.93M</td>
<td style="text-align: center;">380K</td>
<td style="text-align: center;">20K</td>
<td style="text-align: center;">38.79M</td>
<td style="text-align: center;">7.6M</td>
<td style="text-align: center;">29</td>
</tr>
<tr class="odd">
<td style="text-align: left;">VGG16</td>
<td style="text-align: center;">154.7G</td>
<td style="text-align: center;">196.85M</td>
<td style="text-align: center;">10K</td>
<td style="text-align: center;">10K</td>
<td style="text-align: center;">288.03M</td>
<td style="text-align: center;">138.36M</td>
<td style="text-align: center;">512.2</td>
</tr>
</tbody>
</table>
    
\*Inception v3, v4 and Inception-ResNetV2 did not have any Caffe model, so we reported their size related information from mxnet2 and tensorflow3 respectively. Inception-ResNet-V2 would take 60 days of training with 2 Titan X to achieve the reported accuracy4

As it can be seen, our architecture both has much fewer number of parameters (with an exception of squeezenet) and also much fewer number of operations.


1# Data-augmentation method used by stochastic depth paper: [https://github.com/Pastromhaug/caffe-stochastic-depth](https://github.com/Pastromhaug/caffe-stochastic-depth).

2# https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md

3# https://github.com/tensorflow/models/tree/master/slim

4# https://github.com/revilokeb/inception\_resnetv2\_caffe
