
## 重要模块

tensorflow/python/training
tensorflow/contrib/slim
tensorflow/contrib/training
tensorflow/contrib/metrics
tensorflow/contrib/losses



## 预备知识

https://www.udacity.com/course/deep-learning--ud730.
http://playground.tensorflow.org/
https://www.coursera.org/learn/neural-networks http://cs231n.stanford.edu/
https://docs.google.com/forms/d/1mUztUlK6_z31BbMW5ihXaYHlhBcbDd94mERe-8XHyoI/viewform

## 步骤

1. Import or generate datasets
2. Transform and normalize data
3. Partition datasets into train, test, and validation sets
4. Set algorithm parameters (hyperparameters)
5. Initialize variables and placeholders
6. De ne the model structure
7. Declare the loss functions
8. Initialize and train the model
9. Evaluate the model
10. Tune hyperparameters
11. Deploy/predict new outcomes

## 困惑新手的几个名词解释

### broadcasting

即在进行矩阵想加时，如果后面是一个数字，这在矩阵中是不允许的，必须是向量，broadcasting
自动帮你做了这件事情，也就是你可以用一个矩阵和一个数字相加

### Variables

参见 variable.md

### Placeholders

### name_scope

### global_step

1. 为 Variables.Variables, ops.Tensor 或 resource_variable_ops.ResourceVariable 的实例
2. 是一个 scalar
3. 基本类型为 int。
4. 保存在 ops.GraphKeys.GLOBAL_STEP 或  graph 的  tensor 中

有什么用 ：TODO

用法

```python
import tensorflow as tf;
global_step = tf.contrib.framework.get_or_create_global_step()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print tf.train.global_step(sess, global_step) # 此前必须初始化
```


创建 global_step 的方法
```
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])
```

global_step 传递给 minimize，每次迭代，global_step 就会加一

## epoch






## Variables 和 Placeholders

Variables are the parameters of the algorithm and TensorFlow keeps track of how to change these to optimize the algorithm.

Placeholders are objects that allow you to feed in data of a speci c type and shape and depend on the results of the
computational graph, such as the expected outcome of a computation.

Placeholders get data from a feed_dict argument in the session.

To put a placeholder in the graph, we must perform at least one operation on the placeholder. We initialize the graph,
declare x to be a placeholder, and de ne y as the identity operation on x, which just returns x.

Placeholders 通过 session.run 的 feed_dict 参数传递数据
Variables 在 session.run 之前是要初始化好的

Placeholders : 要喂给算法的数据
Variables : 待求的数据

## name_scope

常用在函数定义中，如
``` python
def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                 staircase=False, name=None):
  with ops.name_scope(name, "ExponentialDecay",
                 [learning_rate, global_step,
                  decay_steps, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    return math_ops.multiply(learning_rate, math_ops.pow(decay_rate, p), name=name)
```


### Loss Function

* L2 norm loss : tf.square(target - x_vals)
* L1 norm loss : tf.abs(target - x_vals)
* Demming : tf.reduce_mean( tf.truediv( tf.abs(tf.subtract(y_target, tf.add(b, tf.matmul(x_input, A)))), tf.sqrt(tf.add(tf.square(A), 1))))
* l
* Pseudo-Huber loss : tf.mul(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)  //delta1 = 0.25
* Classi cation loss
* Hinge loss : tf.maximum(0., 1. - tf.mul(target, x_vals))  用于 SVM, NN
* Cross-entropy loss : - tf.mul(target, tf.log(x_vals)) - tf.mul((1. - target), tf.log(1. - x_vals))
* Sigmoid cross entropy loss : tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
* Weighted cross entropy loss : tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight) //weight = 0.5
* Softmax cross-entropy loss :  tf.nn.softmax_cross_entropy_with_ logits(unscaled_logits, target_dist)
* Sparse softmax cross-entropy loss : tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits, sparse_target_dist)

Kingma, D., Jimmy, L. Adam: A Method for Stochastic Optimization. ICLR 2015. https://arxiv.org/pdf/1412.6980.pdf
Ruder, S. An Overview of Gradient Descent Optimization Algorithms. 2016. https://arxiv.org/pdf/1609.04747v1.pdf
Zeiler, M. ADADelta: An Adaptive Learning Rate Method. 2012. http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

### MomentumOptimizer : stuck in the  at spot of a saddle.

Sometimes the standard gradient descent algorithm can get stuck or slow down signi cantly.
This can happen when the optimization is stuck in the  at spot of a saddle
To combat this, there is another algorithm that takes into account a momentum term, which adds on
a fraction of the prior step's gradient descent value


### AdagradOptimizer 合理选择步长

Another variant is to vary the optimizer step for each variable in our models. Ideally, we
would like to take larger steps for smaller moving variables and shorter steps for faster changing variables.
We will not go into the mathematics of this approach, but a common implementation of this idea is called the Adagrad algorithm.

### batch, stochastic
Operating on one training example can make for a very erratic learning process, while using a too large batch can be computationally expensive.



### binary predictor

* Logic Regression :  find any separating line that maximizes the distance (probabilistically)
* SVM : minimize the error while maximizing the margin between classes

If the problem has a large number of features compared to training examples, try logistic regression or a linear SVM.

If the number of training examples is larger, or the data is not linearly separable, a SVM with a Gaussian kernel may be used.


## SVM

line kernel

Guassian kernel
