## 见解：
- XL
    > - Context 部分使用了Softmax，直接对Context进行”总结“，而不是像之前那样 Q, K, V 的计算 
    > - 看视频说没了Position Embedding这个结构无法发挥作用，猜测是由于patch之间的位置关系被破坏 
    >   - 解决：利用Swin Transformer的patch分割法，尝试下能不能消除Lambda的PE

## TODO:

- [ ] [添加 Lambda Layer](https://github.com/lucidrains/lambda-networks)
- [ ] [SE Attention](https://zhuanlan.zhihu.com/p/102035721)
- [ ] [COCO数据集的下载](https://blog.csdn.net/m0_37644085/article/details/81948396)

- [ ] 消融实验

## 资源
- [论文](https://openreview.net/forum?id=xTJEN-ggl1b)
- [视频解析](https://www.youtube.com/watch?v=3qxJ2WD8p4w&t=668s)
- 其他实现的代码
    - https://github.com/leaderj1001/LambdaNetworks
