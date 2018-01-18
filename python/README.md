


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

Variables
Placeholders
ops.name_scope

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
