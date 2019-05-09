## Pytorch implementation of [*"Squeeze-and-Excitation Networks"*](https://arxiv.org/abs/1709.01507) and [*"Attention Augmented Convolutional Networks"*](https://arxiv.org/abs/1904.09925)

### *Squeeze-and-Excitation Networks*:
![*Squeeze-and-Excitation Networks*](https://github.com/tsmotlp/Attentions-on-Image/blob/master/se.png)
### *Attention Augmented Convolutional Networks*:
![*Attention Augmented Convolutional Networks*](https://github.com/tsmotlp/Attentions-on-Image/blob/master/aa.png)
## Repository Description:
se_block.py---implementation for Squeeze-and-Excitation Network</br>
aa_conv2d.py---implementation for Attention-Augmented Convolutional Network</br>
resnet.py---basic model, including ResNet50, ResNet101, ResNet152</br>
se_resnet.py---Squeeze-and-Excitation Block used in resnet</br>
aa_resnet.py---Attention-Augmented Convolution used in resnet</br>
dataset.py---code for loading train and test data(here using CIFAR10)</br>
opts.py---hyper-parameters options for training and testing</br>
train.py---code for training</br>
test.py---code for testing</br>
main.py---main file, training model 'train_interval' epochs, then testing the model a time</br>
vis_tool.py---code for visualizing loss and accuracy
#### if you have any question and suggestion, please contact:*hyq_tsmotlp@163.com*
