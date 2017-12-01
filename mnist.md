
## softmax regression

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.

数据分为三组：training data 55000, validation data 5000, test data 10000

x : image 28 x 28 图片
y : label

### 读数据

1. 数据源  http://yann.lecun.com/exdb/mnist/


数据源描述

 将数据由图片变为矩阵，矩阵每行是一个图片的展开为向量，列是所有的待训练的图片 55000 * 784
 标签为 55000 * 10

### 计算模型


y = tf.nn.softmax(tf.matmul(x, W) + b)


### 训练

cross-entropy

stochastic gradient descent

### 评估模型



## 

reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

## 术语

one-hot vector : 一个向量只有一个因素为 1, 其余都都为 0, 比如 3 为 [0, 0, 0, 1, 0, 0]


## QA

这条语句导致 print 语句无法使用，

    from __future__ import print_function

而 python 只是简单地报了如下错误

     File "args.py", line 43
        print 'The filename is : %s .' % args.filename
                                     ^
    SyntaxError: invalid syntax

出错地方与真正出错地方毫无关系, 花了一个小时调试。。。。。

http://yann.lecun.com/exdb/mnist/


## 扩展阅读

http://colah.github.io/posts/2014-10-Visualizing-MNIST/
http://neuralnetworksanddeeplearning.com/chap3.html#softmax
