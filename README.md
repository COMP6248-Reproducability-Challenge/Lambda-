## 见解：
- XL
    > - Context 部分使用了Softmax，直接对Context进行”总结“，而不是像之前那样 Q, K, V 的计算 
    > - 看视频说没了Position Embedding这个结构无法发挥作用，猜测是由于patch之间的位置关系被破坏 
    >   - 解决：利用Swin Transformer的patch分割法，尝试下能不能消除Lambda的PE

## TODO:
- Main
    - [ ] [添加 Lambda Layer](https://github.com/lucidrains/lambda-networks)
- SE Attention
    - [X] [SE Attention代码添加](https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py)
- ResNet 改
    - [X] [Torch 官方实现](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)

- 数据集
    - [ ] [COCO数据集的下载](https://blog.csdn.net/m0_37644085/article/details/81948396)

- 实验部分
    - [ ] 消融实验
    - [ ] [Swin 改](https://github.com/microsoft/Swin-Transformer)

## 资源
- [论文](https://openreview.net/forum?id=xTJEN-ggl1b)
- [视频解析](https://www.youtube.com/watch?v=3qxJ2WD8p4w&t=668s)
-  [SE Attention知乎解析](https://zhuanlan.zhihu.com/p/102035721)
-  [SE Attention CSDN](https://blog.csdn.net/Evan123mg/article/details/80058077)
- 其他实现的代码
    - https://github.com/leaderj1001/LambdaNetworks
- 有用的库
  - https://github.com/pprp/SimpleCVReproduction
  - https://github.com/pprp/awesome-attention-mechanism-in-cv
