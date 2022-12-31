
Please note that models are converted from onnx to caffe.
The mean, std and crop ratio used are as follows: 
```python
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
```
Also note that images were not channel swapped during training so you dont need to 
do channel swap. 
You also DO NOT need to rescale the input to [0-255]. 

