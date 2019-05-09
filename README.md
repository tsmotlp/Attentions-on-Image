## Pytorch implementation of [*"Squeeze-and-Excitation Networks"*](https://arxiv.org/abs/1709.01507) and [*"Attention Augmented Convolutional Networks"*](https://arxiv.org/abs/1904.09925)

### *Squeeze-and-Excitation Networks*:
![*Squeeze-and-Excitation Networks*](https://github.com/tsmotlp/Attentions-on-Image/blob/master/se.png)
### *Attention Augmented Convolutional Networks*:
![*Attention Augmented Convolutional Networks*](https://github.com/tsmotlp/Attentions-on-Image/blob/master/aa.png)
## Repository Description:
1. se_block.py---implementation for Squeeze-and-Excitation Network</br>
2. aa_conv2d.py---implementation for Attention-Augmented Convolutional Network</br>
3. resnet.py---basic model, including ResNet50, ResNet101, ResNet152</br>
4. se_resnet.py---Squeeze-and-Excitation Block used in resnet</br>
5. aa_resnet.py---Attention-Augmented Convolution used in resnet</br>
6. dataset.py---code for loading train and test data(here using CIFAR10)</br>
7. opts.py---hyper-parameters options for training and testing</br>
8. train.py---code for training</br>
9. test.py---code for testing</br>
10. main.py---main file, training model 'train_interval' epochs, then testing the model a time</br>
11. vis_tool.py---code for visualizing loss and accuracy
</br>
if you have any questions and suggestions, please contact: *hyq_tsmotlp@163.com*
