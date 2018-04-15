
## 创建常量的几种方式

### 方式一

>>> a = tf.constant(np.arange(6).reshape(2,3))

### 方式二

>>> a = tf.convert_to_tensor([[1, 0, 3], [3, 0, 0]])

## 创建变量的几种方式

方式一

tf.Variable(a)

方式二

tf.get_variable

>>> a = tf.constant(np.arange(6).reshape(2,3))

### 矩阵操作

#### 矩阵拆合

* tf.split 将一个矩阵按照列进行分割
* tf.stack 将多个矩阵组合成一个矩阵
* tf.unstack 将一个矩阵按照某一维度展开为子矩阵
* tf.concat :
* tf.slice
* tf.tile

tf.stack 与  tf.concat 有什么区别

#### 矩阵求和

* tf.reduce_mean : 将矩阵按照某一维度求和
* tf.reduce_max : 将矩阵按照某一维度求最大值
* tf.minimum
* tf.maximum
* tf.argmax : 计算某一维度的最大值的索引

#### 矩阵逻辑操作

返回 True, False 矩阵

* tf.reduce_any
* tf.logical_and

#### 矩阵比较

返回 True, False 矩阵

* tf.greater
* tf.less
* tf.greater_equal
* tf.equal

#### 矩阵集合



## tf.placeholder

placeholder(dtype, shape=None, name=None)

x = tf.placeholder(tf.float32, shape=(1024, 1024))


### tf.variable_scope  与 tf.get_variable

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v")  # The same as v above.

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v

with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").

with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
```
### tf.constant

constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
tensor = tf.constant(-1.0, shape=[2, 3])

### tf.cast

cast(x, dtype, name=None)

>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> sess = tf.Session()
>>> sess.run(tf.cast(a, tf.int32))
array([ 0,  2,  2,  1, -4], dtype=int32)


### tf.random_normal

random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

与 np.random.normal 的包装

### tf.squeeze

squeeze(input, axis=None, name=None, squeeze_dims=None)

如果 axis 没有指定，就将所有维度只有 1 个元素的维度删除，
如果 axis 指定了， 如果指定的维度只有一个元素就删除该维

>>> v = tf.constant(1.0, shape=[1, 2, 1, 3, 1, 1])
>>> sess.run(tf.shape(tf.squeeze(v)))
array([2, 3], dtype=int32)
>>> sess.run(tf.shape(tf.squeeze(v, [2,4])))
array([1, 2, 3, 1], dtype=int32)
>>> sess.run(tf.shape(tf.squeeze(v,
>>> [0,4])))
array([2, 1, 3, 1], dtype=int32)
>>> sess.run(tf.shape(tf.squeeze(v,
>>> [0,4,5])))
array([2, 1, 3], dtype=int32)

### tf.square

square(x, name=None)

>>> v = tf.constant([1, 2, 3, 4, 5, 6, 7])
>>> vv = tf.square(v)
>>> sess = tf.Session()
>>> sess.run(vv)
array([ 1,  4,  9, 16, 25, 36, 49], dtype=int32)

### tf.reduce_any

reduce_any(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

对 input_tensor 的第 axis 就算逻辑或

    ```python
    # 'x' is [[True,  True]
    #         [False, False]]
    tf.reduce_any(x) ==> True
    tf.reduce_any(x, 0) ==> [True, True]
    tf.reduce_any(x, 1) ==> [True, False]
    ```

### tf.argmax

argmax(input, axis=None, name=None, dimension=None)

>>> sess.run(x)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19]], dtype=int32)
>>> sess.run(tf.argmax(x, 1))
array([3, 3, 3, 3, 3])
>>> sess.run(tf.argmax(x, 0))
array([4, 4, 4, 4])

### tf.reduce_max

reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

>>> a = [[1, 3, 4, 0, 0], [2, 4, 5, 0, 0]]
>>> sess.run(tf.reduce_max(a, axis=0))
array([2, 4, 5, 0, 0], dtype=int32)
>>> sess.run(tf.reduce_max(a, axis=1))
array([4, 5], dtype=int32)


### tf.reduce_mean

reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

>>> import tensorflow as tf
>>  sess = tf.Session()
>>> a = tf.constant(np.arange(10).reshape(2,5))
>>> sess.run(a)
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> sess.run(tf.reduce_mean(a))
4
>>> sess.run(tf.reduce_mean(a, 0))
array([2, 3, 4, 5, 6])
>>> sess.run(tf.reduce_mean(a, 1))
array([2, 7])

>>> a = tf.constant(np.arange(10).reshape(2,5), tf.float32)
>>> sess.run(tf.reduce_mean(a, 1))
array([ 2.,  7.], dtype=float32)
>>> sess.run(tf.reduce_mean(a, 0))
array([ 2.5,  3.5,  4.5,  5.5,  6.5], dtype=float32)
>>> sess.run(a)
array([[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.]], dtype=float32)

### tf.reduce_sum

reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

>>> a = tf.constant(np.arange(6).reshape(2,3))
>>> sess = tf.Session()
>>> sess.run(a)
array([[0, 1, 2],
       [3, 4, 5]])
>>> tf.reduce_mean(a)
<tf.Tensor 'Mean:0' shape=() dtype=int64>
>>> sess.run(tf.reduce_mean(a))
2
>>> sess.run(tf.reduce_mean(a, 0))
array([1, 2, 3])
>>> sess.run(tf.reduce_mean(a, 1))
array([1, 4])

### tf.reshape

>>> a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> sess = tf.Session()
>>> sess.run(tf.reshape(a, [-1,3]))
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]], dtype=int32)
>>> sess.run(tf.reshape(a, [3,3]))
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]], dtype=int32)
>>> b = tf.constant([[[1, 1], [2, 2]],[[3, 3], [4, 4]]])
>>> sess.run(b)
array([[[1, 1],
        [2, 2]],

       [[3, 3],
        [4, 4]]], dtype=int32)
>>> sess.run(tf.reshape(b, [-1, 4])
... )
array([[1, 1, 2, 2],
       [3, 3, 4, 4]], dtype=int32)
>>> sess.run(tf.reshape(b, [-1, 4]))
array([[1, 1, 2, 2],
       [3, 3, 4, 4]], dtype=int32)
>>> sess.run(tf.reshape(b, [-1, 2]))
array([[1, 1],
       [2, 2],
       [3, 3],
       [4, 4]], dtype=int32)
>>> sess.run(tf.reshape(b, [-1, 8]))
array([[1, 1, 2, 2, 3, 3, 4, 4]], dtype=int32)
>>> sess.run(tf.reshape(b, [2, -1]))
array([[1, 1, 2, 2],
       [3, 3, 4, 4]], dtype=int32)

### tf.expand_dims

expand_dims(input, axis=None, name=None, dim=None)

>>> a = tf.constant(np.arange(10).reshape(2,5))
>>> sess.run(tf.expand_dims(a, 0))
array([[[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]]])
>>> sess.run(tf.expand_dims(a, 1))
array([[[0, 1, 2, 3, 4]],

       [[5, 6, 7, 8, 9]]])
>>> sess.run(tf.expand_dims(a, 2))
array([[[0],
        [1],
        [2],
        [3],
        [4]],

       [[5],
        [6],
        [7],
        [8],
        [9]]])
>>> sess.run(tf.shape(tf.expand_dims(a, 0)))
array([1, 2, 5], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(a, 1)))
array([2, 1, 5], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(a, 2)))
array([2, 5, 1], dtype=int32)

### tf.round

round(x, name=None)

>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> sess = tf.Session()
>>> sess.run(tf.round(a))
array([ 1.,  2.,  2.,  2., -4.], dtype=float32)

### tf.equal

equal(x, y, name=None)


### tf.boolean_mask

boolean_mask(tensor, mask, name='boolean_mask')

    ```python
    # 1-D example
    tensor = [0, 1, 2, 3]
    mask = np.array([True, False, True, False])
    boolean_mask(tensor, mask) ==> [0, 2]
    ```

### tf.nn.sigmoid_cross_entropy_with_logits

sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)



### tf.train.GradientDescentOptimizer


### tf.as_type

将 np 的 dtype 转换为  tensorflow 的  type

>>> import numpy as np
>>> a = np.array([1.,2.])
>>> a.dtype
dtype('float64')
>>> tf.as_dtype(a.dtype)
tf.float64


## dtypes.as_dtype

>>> a = tf.convert_to_tensor(1)
>>> a.dtype
tf.int32
>>> from tensorflow.python.framework import dtypes
>>> dtypes.as_dtype(a.dtype)
tf.int32


### tf.expand_dims

>>> v = tf.constant([1, 2, 3, 4, 5, 6, 7])
>>> sess.run(tf.expand_dims(v, 0))
array([[1, 2, 3, 4, 5, 6, 7]], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(v, 0)))
array([1, 7], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(v, 1)))
array([7, 1], dtype=int32)

>>> v = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> sess.run(v)
  array([[1, 2, 3],
         [4, 5, 6]], dtype=int32)
>>> sess.run(tf.expand_dims(v, 0))
  array([[[1, 2, 3],
          [4, 5, 6]]], dtype=int32)
>>> sess.run(tf.expand_dims(v, 1))
  array([[[1, 2, 3]],
        [[4, 5, 6]]], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(v, 1)))
array([2, 1, 3], dtype=int32)
>>> sess.run(tf.shape(tf.expand_dims(v, 0)))
array([1, 2, 3], dtype=int32)


### tf.parallel_stack

parallel_stack(values, name='parallel_stack')

>>> x = tf.constant([1,4])
>>> y = tf.constant([2,5])
>>> z = tf.constant([3,6])
>>> sess.run(tf.parallel_stack([x, y, z]))
    array([[1, 4], [2, 5], [3, 6]], dtype=int32)


### tf.slice

slice(input_, begin, size, name=None)

第一个表示开始坐标，第二个元素表示在各个维度取的数量

>>> x = tf.constant([[11,12,13,14], [21,22,23,24], [31,32,33,4], [41,42,43,44]])
>>> t = tf.convert_to_tensor(x)
>>> sess = tf.Session()
>>> sess.run(tf.slice(t, [1, 1], [3, 2]))
array([[22, 23],
       [32, 33],
       [42, 43]], dtype=int32)

这里 [1, 1] 表示从坐标索引 [1,1] 开始，取 3 行，2 列

>>> z=tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]]])
>>> begin_z=[0,1,1]
>>> size_z=[-1,1,2]
>>> out=tf.slice(z,begin_z,size_z)
>>> print sess.run(out)
[[[ 5  6]]

 [[11 12]]

  [[17 18]]]]

典型场景是读图片数据为一维数组，之后转换为二维

>>> z=tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]]])
>>> sess.run(tf.reshape(tf.slice(a, [0], [6]), [2,3]))
array([[0, 1, 2],
       [3, 4, 5]])

## tf.strided_slice

左闭右开

>>> input = tf.convert_to_tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
>>> sess.run(tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]))
array([[[3, 3, 3]]], dtype=int32)
>>> sess.run(tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]))
array([[[3, 3, 3],
        [4, 4, 4]]], dtype=int32)
>>> sess.run(tf.strided_slice(input, [1, -1, 0], [2, -3, 3], [1, -1, 1]))
array([[[4, 4, 4],
        [3, 3, 3]]], dtype=int32)

### tf.tenspose

transpose(a, perm=None, name='transpose')

>>> a = np.array([[[1, 2, 3], [4, 5, 6]], [[7,8,9], [10,11, 12]]])
>>> sess.run(tf.transpose(a))
array([[[ 1,  7],
        [ 4, 10]],

       [[ 2,  8],
        [ 5, 11]],

       [[ 3,  9],
        [ 6, 12]]])
>>> sess.run(tf.transpose(a, [2, 1, 0]))
array([[[ 1,  7],
        [ 4, 10]],

       [[ 2,  8],
        [ 5, 11]],

       [[ 3,  9],
        [ 6, 12]]])
>>> sess.run(tf.transpose(a, [1, 2, 0]))
array([[[ 1,  7],
        [ 2,  8],
        [ 3,  9]],

       [[ 4, 10],
        [ 5, 11],
        [ 6, 12]]])
>>> sess.run(tf.transpose(a, [1, 0, 2]))
array([[[ 1,  2,  3],
        [ 7,  8,  9]],

       [[ 4,  5,  6],
        [10, 11, 12]]])

### np.vstack

>>> a = np.array([1,2,3,4])
>>> b = np.array([4,5,6,7])
>>> np.vstack((a,b))
array([[1, 2, 3, 4],
       [4, 5, 6, 7]])

>>> a = np.array([[1],[2],[3],[4]])
>>> b = np.array([[4],[5],[6],[7]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [4],
       [4],
       [5],
       [6],
       [7]])

### tf.split

split(value, num_or_size_splits, axis=0, num=None, name='split')

num_or_size_splits 支持数字和列表，如果是数字，就将 value 的第  axis 维
均分成 num_or_size_splits 部分；如果是列表，就将 value 的第  axis 依次
分成数组指定的每个元素的部分，下面的例子最能说明问题

    ```python
    # 'value' is a tensor with shape [5, 30]
    # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
    split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
    tf.shape(split0) ==> [5, 4]
    tf.shape(split1) ==> [5, 15]
    tf.shape(split2) ==> [5, 11]
    # Split 'value' into 3 tensors along dimension 1
    split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
    tf.shape(split0) ==> [5, 10]
    ```

### tf.concat

concat(values, axis, name='concat')

axis = 0: 列的数量不变，行叠加
axis = 1: 行的数量不变，行叠加

>>> x = [[1, 2, 3], [4, 5, 6]]
>>> y = [[3, 2, 1], [6, 5, 4]]
>>> sess.run(tf.concat([x, y], 1))
array([[1, 2, 3, 3, 2, 1],
       [4, 5, 6, 6, 5, 4]], dtype=int32)
>>> sess.run(tf.concat([x, y], 0))
array([[1, 2, 3],
       [4, 5, 6],
       [3, 2, 1],
       [6, 5, 4]], dtype=int32)

### tf.stack

stack(values, axis=0, name='stack')

>>> a = [1, 2]
>>> b = [3, 4]
>>> c = [5, 6]
>>> tf.stack([a, b, c])
<tf.Tensor 'stack:0' shape=(3, 2) dtype=int32>
>>> sess.run(tf.stack([a, b, c]))
array([[1, 2],
       [3, 4],
       [5, 6]], dtype=int32)
>>> sess.run(tf.stack([a, b, c], axis = 1))
array([[1, 3, 5],
       [2, 4, 6]], dtype=int32)

### tf.unstack

unstack(value, num=None, axis=0, name='unstack')

>>> a = [1, 2, 3]
>>> x, y, z = tf.unstack(a)
>>> sess.run([x, y, z])
[1, 2, 3]

>>> a = [[1, 3, 4, 0, 0], [2, 4, 5, 0, 0]]
>>> sess.run(tf.unstack(a))
[array([1, 3, 4, 0, 0], dtype=int32), array([2, 4, 5, 0, 0], dtype=int32)]

### 条件判断

* tf.greater

返回 bool 矩阵或向量

###

### tf.train.string_input_producer

string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None, cancel_op=None)
创建一个 FIFO 队列

## tf.nn.conv2d

conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

## tf.nn.relu

relu(features, name=None)

## tf.nn.max_pool

max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)

## tf.nn.dropout

dropout(x, keep_prob, noise_shape=None, seed=None, name=None)

```python
#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np


def main():

    with tf.Graph().as_default():
        import numpy as np
        input_x = np.ones([3, 4])
        print input_x

        keep_prob_list = [0.1, 0.5, 0.8, 1.0]
        drop = []
        for i, keep_prob in enumerate(keep_prob_list):
            drop.append(tf.nn.dropout(x=input_x, keep_prob=keep_prob))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i, drop_i in enumerate(drop):
                _drop_i = sess.run(drop_i)
                print '\n----' + str(keep_prob_list[i]) + '------\n'
                print _drop_i

if __name__ == "__main__":
    main()
```
多次运行，结果非常有意思


## tf.contrib.layers.full

>>> input_x = np.ones([3, 4])
>>> fn = tf.contrib.layers.fully_connected(inputs=input_x, num_outputs=1)
>>> sess.run(tf.global_variables_initializer())
>>> print(input_x)
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
>>> print(sess.run(fn))
[[ 2.07639697]
 [ 2.07639697]
 [ 2.07639697]]

#### tf.nn.softmax

>>> input_x = np.ones([3, 4])
>>> fn = tf.contrib.layers.fully_connected(inputs=input_x, num_outputs=1)
>>> sess.run(tf.global_variables_initializer())
>>> print(sess.run(tf.nn.softmax(input_x)))
[[ 0.25  0.25  0.25  0.25]
 [ 0.25  0.25  0.25  0.25]
 [ 0.25  0.25  0.25  0.25]]

## tf.nn.bias_add

bias_add(value, bias, data_format=None, name=None)

用 tf.nn.bias_add 替代 tf.add 用于增加 bias，原因是  tf.nn.bias_add 自带
broadcasting.

## tf

value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value)
with tf.Session():
    x = tf.get_variable('x', shape=[2, 4], initializer=init)
    x.initializer.run()
    print(x.eval())
    x = tf.get_variable('x', shape=[3, 4], initializer=init)
    x.initializer.run()
    print(x.eval())
    x = tf.get_variable('x', shape=[2, 3], initializer=init)
    x.initializer.run()
    print(x.eval())

## tf.add_n

add_n(inputs, name=None) :  将 inputs 中每个元素的对应元素相加

## tf.cumprod  tf.cumsum

>>> x =[ [1, 2, 3],[4,5,6]]
>>> sess.run(tf.cumprod(tf.convert_to_tensor(x), axis=1))
array([[  1,   2,   6],
       [  4,  20, 120]], dtype=int32)
>>> sess.run(tf.cumprod(tf.convert_to_tensor(x), axis=0))
array([[ 1,  2,  3],
       [ 4, 10, 18]], dtype=int32)
>>> sess.run(tf.cumsum(tf.convert_to_tensor(x)))
array([[1, 2, 3],
       [5, 7, 9]], dtype=int32)
>>> sess.run(tf.cumsum(tf.convert_to_tensor(x), axis=1))
array([[ 1,  3,  6],
       [ 4,  9, 15]], dtype=int32)


## tf.exponential_decay

exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

## tf.control_dependencies

control_dependencies(control_inputs)

只有 control_inputs 中的 Operation 或 Tensor
执行或计算之后，才能继续执行当前上下文的内容

### tf.cumsum

cumsum(x, axis=0, exclusive=False, reverse=False, name=None)


>>> b = [4, 5, 6]
>>> sess.run(tf.cumsum(b))
array([ 4,  9, 15], dtype=int32)

>>> a = tf.reshape(tf.range(15), (3,5))
>>> sess.run(a)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]], dtype=int32)
>>> sess.run(tf.cumsum(a, 1))
array([[ 0,  1,  3,  6, 10],
       [ 5, 11, 18, 26, 35],
       [10, 21, 33, 46, 60]], dtype=int32)
>>> sess.run(tf.cumsum(a))
array([[ 0,  1,  2,  3,  4],
       [ 5,  7,  9, 11, 13],
       [15, 18, 21, 24, 27]], dtype=int32)

tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]

## tf.reduce_sum

reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):

>>> import tensorflow as tf
>>> a = tf.convert_to_tensor([[1, 2, 3], [3, 2, 1]])
>>> sess.run(a)
array([[1, 2, 3],
       [3, 2, 1]], dtype=int32)
>>> sess.run(tf.reduce_sum(2))
2
>>> sess.run(tf.reduce_sum(a))
12
>>> sess.run(tf.reduce_sum(a, 0))
array([4, 4, 4], dtype=int32)
>>> sess.run(tf.reduce_sum(a, 0, keep_dims=True))
array([[4, 4, 4]], dtype=int32)
>>> sess.run(tf.reduce_sum(a, 1, keep_dims=True))
array([[6],
       [6]], dtype=int32)
>>> sess.run(tf.reduce_sum(a, [0,1], keep_dims=True))
array([[12]], dtype=int32)

## tf.count_nonzero

```python
>>> import tensorflow as tf
>>> a = tf.convert_to_tensor([[1, 0, 3], [3, 0, 0]])
>>> sess.run(tf.count_nonzero(a))
3
>>> sess.run(tf.count_nonzero(a, 1))
array([2, 1])
>>> sess.run(tf.count_nonzero(a, 0))
array([2, 0, 1])
>>> sess.run(tf.count_nonzero(a, 0, keep_dims=True))
array([[2, 0, 1]])
>>> sess.run(tf.count_nonzero(a, [0, 1], keep_dims=True))
array([[3]])
```

## tf.reduce_mean

## ops.convert_to_tensor

convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None)

将一个Python Object, 转为  Tensor，可以为 Tensor, numpy arrays, Python List
Python scalars

## tf.nn.zero_fraction

```python
>>> a = tf.convert_to_tensor([1, 0, 0, 0, 1])
>>> sess = tf.Session()
>>> sess.run(tf.nn.zero_fraction(a))
0.60000002
```

## tf.pad

pad(tensor, paddings, mode='CONSTANT', name=None)

paddings[i] 的第 i 个元素表示在第 n 为的前面和后面分别增加几行 0
```python
>>> a = tf.convert_to_tensor([[1,2,3], [4, 5, 6]])
>>> sess
<tensorflow.python.client.session.Session object at 0x10dbe46d0>
>>> sess.run(tf.pad(a, [[2, 1], [2, 1]])
... )
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 3, 0],
       [0, 0, 4, 5, 6, 0],
       [0, 0, 0, 0, 0, 0]], dtype=int32)
>>> sess.run(tf.pad(a, [[2, 1], [1, 2]]))
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 1, 2, 3, 0, 0],
       [0, 4, 5, 6, 0, 0],
       [0, 0, 0, 0, 0, 0]], dtype=int32)
```
## tf.group

group(inputs, **kwargs)

1. 对  inputs 根据 device 排序
2. 该语句执行完成，inputs 的每个变量都已经完成

## SparseTensor

稀疏矩阵

SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

等于

```python
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
```

## tf.size

size(input, name=None, out_type=tf.int32)

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
``k`

## tf.rank

rank(input, name=None)

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

## tf.range

range(start, limit=None, delta=1, dtype=None, name='range')

```python
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

# 'start' is 3
# 'limit' is 1
# 'delta' is -0.5
tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]

# 'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
```
### tf.ones_like

ones_like(tensor, dtype=None, name=None, optimize=True)

将  tensor 元素都变为 1

```python
>>> sess.run(tf.ones_like(tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]])))
array([[1, 1, 1],
       [1, 1, 1]], dtype=int32)

>>> a = tf.convert_to_tensor([[False, True, False],[True, True, True]])
>>> sess.run(tf.ones_like(a))
array([[ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> a = tf.convert_to_tensor([[0, 2, 4],[-1, 2, 1]])
>>> sess.run(tf.ones_like(a))
array([[1, 1, 1],
       [1, 1, 1]], dtype=int32)
```
### tf.metrics.mean

mean(values, weights=None, metrics_collections=None, updates_collections=None, name=None)

if weights is None:  return tf.reduce_sum(values) / tf.size(values)
if weights is not None,  return tf.reduce_sum(values * weights) / tf.reduce_mean(weights)

计算均值
```python
import tensorflow as tf

feed_values = ((0, 1), (-4.2, 9.1), (6.5, 0), (-3.2, 4.0))
values = tf.placeholder(dtype=tf.float32)
weights = [[1,1], [1, 0], [0, 1], [0, 0]]

mean, update_op = tf.metrics.mean(
    values, weights,
    metrics_collections=[my_collection_name],
    updates_collections = my_collection_update)

with tf.Session() as sess:
  sess.run(tf.local_variables_initializer())
  print(sess.run(update_op, feed_dict={values: feed_values}))
  print("my_collection_name", sess.run(tf.get_collection(my_collection_name)))
  print("my_collection_update", sess.run(tf.get_collection(my_collection_update)))
  print(sess.run(mean))
  print(sess.run(tf.constant(1)))
```

### tf.metrics.mean_tensor

mean_tensor(values, weights=None, metrics_collections=None, updates_collections=None, name=None)

计算均值，但是与 values 的 shape 是相同的
```python
import tensorflow as tf

def _enqueue_vector(sess, queue, values, shape=None):
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(
      queue.enqueue(tf.constant(
          values, dtype=dtype, shape=shape)))

with tf.Session() as sess:
  # Create the queue that populates the values.
  values_queue = tf.FIFOQueue(
      4, dtypes=tf.float32, shapes=(1, 2))
  _enqueue_vector(sess, values_queue, [0, 1])
  _enqueue_vector(sess, values_queue, [-4.2, 9.1])
  _enqueue_vector(sess, values_queue, [6.5, 0])
  _enqueue_vector(sess, values_queue, [-3.2, 4.0])
  values = values_queue.dequeue()

  # Create the queue that populates the weighted labels.
  weights_queue = tf.FIFOQueue(
      4, dtypes=tf.float32, shapes=(1, 2))
  _enqueue_vector(sess, weights_queue, [1, 1])
  _enqueue_vector(sess, weights_queue, [1, 0])
  _enqueue_vector(sess, weights_queue, [0, 1])
  _enqueue_vector(sess, weights_queue, [0, 0])
  weights = weights_queue.dequeue()

  mean, update_op = tf.metrics.mean_tensor(values, weights)

  sess.run(tf.local_variables_initializer())
  sess.run(tf.global_variables_initializer())
  for _ in range(4):
    print(sess.run(update_op))
```
结果
[[ 0.  1.]]
[[-2.0999999  1.       ]]
[[-2.0999999  0.5      ]]
[[-2.0999999  0.5      ]]

```python
>>> a, update_a = tf.metrics.mean_tensor(b)
>>> sess.run(tf.local_variables_initializer())
>>> sess.run(update_a)
array([ 2.,  3.,  0.,  0.], dtype=float32)
>>> sess.run(a)
array([ 2.,  3.,  0.,  0.], dtype=float32)
```


### tf.random_uniform

random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

>>> sess.run(tf.random_uniform((10, 3), maxval=3, dtype=tf.int64, seed=1))
array([[2, 1, 0],
       [2, 2, 1],
       [2, 1, 1],
       [1, 1, 1],
       [1, 2, 2],
       [2, 2, 2],
       [2, 2, 2],
       [1, 1, 2],
       [1, 2, 1],
       [0, 0, 1]])


### tf.metrics.accuracy

accuracy(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None)

return tf.metrics.mean(tf.equal(labels, predictions), weights)

```python
import tensorflow as tf
predictions = tf.random_uniform(
    (10, 3), maxval=3, dtype=tf.int64, seed=1)
labels = tf.random_uniform(
    (10, 3), maxval=3, dtype=tf.int64, seed=2)
accuracy, update_op = tf.metrics.accuracy(predictions, labels)

with tf.Session() as sess:
  sess.run(tf.local_variables_initializer())

  # Run several updates.
  for _ in range(10):
    print("update_op", sess.run(update_op))

  # Then verify idempotency.
  for _ in range(10):
    print("accuracy", sess.run(accuracy))
```

### tf.metrics.mean_absolute_error

mean_absolute_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None)

```python
>>> a = [1, 2, 3, 4]
>>> b = [2, 3, 4, 5]
>>> op, update_op = tf.metrics.mean_absolute_error(a, b)
>>> sess.run(tf.local_variables_initializer())   # 不可少
>>> sess.run(update_op)
1.0
>>> sess.run(op)
1.0
```


### tf.metrics.mean_squared_error

mean_squared_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None)
```python
>>> op, update_op = tf.metrics.mean_squared_error(a, b)
>>> sess.run(tf.local_variables_initializer())  # 不可少
>>> sess.run(op)
0.0
>>> sess.run(update_op)
1.0
>>> sess.run(op)
1.0
```

### tf.metrics.percentage_below

percentage_below(values, threshold, weights=None, metrics_collections=None, updates_collections=None, name=None)



### tf.metrics.true_positives

true_positives(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None)

labels 和  predictions 都为 True 的数量
```python
>>> a = [1, 0, 1, 0]
>>> b = [1, 1, 1, 1]
>>> op, update_op = tf.metrics.true_positives(a, b)
>>> sess.run(tf.local_variables_initializer())
>>> sess.run(op)
0.0
>>> sess.run(update_op)
2.0
>>> sess.run(op)
2.0
>>> w = [0, 0, 1, 0]
>>> op, update_op = tf.metrics.true_positives(a, b, w)
>>> sess.run(tf.local_variables_initializer())
>>> sess.run(update_op)
1.0
>>> sess.run(op)
1.0
```
同理
tf.metrics.false_positives : label 为 False, predictions 为  True 的数量
tf.metrics.false_negatives: label 为  True, predictions 为  False


### tf.div tf.truediv

truediv(x, y, name=None)
div(x, y, name=None)

```python
>>> a = [1, 2, 3, 4]
>>> b = [2, 3, 4, 5]
>>> sess.run(tf.div(b,a))
array([2, 1, 1, 1], dtype=int32)
>>> sess.run(tf.truediv(b,a))
array([ 2.        ,  1.5       ,  1.33333333,  1.25      ])
```

### tf.confusion_matrix

confusion_matrix(labels, predictions, num_classes=None, dtype=tf.int32, name=None, weights=None)

num_classes 如果为  None, 为  labels 中的最大值加 1 (labels 以 0 开始的分类问题)

行：为 labels
列: 为 predictions

labels 与  predictions 的元素个数必须相同，且为一维向量
```python
tf.contrib.metrics.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
    [[0 0 0 0 0]
     [0 0 1 0 0]
     [0 0 1 0 0]
     [0 0 0 0 0]
     [0 0 0 0 1]]
    ```

### tf.tile

tile(input, multiples, name=None)

将 input 的维度 i 重复 multiples[i] 倍

multiples 必须为整数，且元素个数必须与 input 的维度相同
```python
>>> c
[[[1, 2], [3, 4]], [[4, 5], [5, 6]]]
>>> sess.run(tf.tile(c, [2, 2, 3]))
array([[[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]],

       [[4, 5, 4, 5, 4, 5],
        [5, 6, 5, 6, 5, 6],
        [4, 5, 4, 5, 4, 5],
        [5, 6, 5, 6, 5, 6]],

       [[1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]],

       [[4, 5, 4, 5, 4, 5],
        [5, 6, 5, 6, 5, 6],
        [4, 5, 4, 5, 4, 5],
        [5, 6, 5, 6, 5, 6]]], dtype=int32)
```

### tf.logical_and tf.logical_not

### tf.Assert

Assert(condition, data, summarize=None, name=None)

如果 condition 为 False 会抛异常，并答应 data

>>> a = [ 1., 2., 3., 4., 5., 6. ]
>>> sess.run(tf.Assert(tf.less_equal(tf.reduce_max(a), 10.), [a])
>>> sess.run(tf.Assert(tf.less_equal(tf.reduce_max(a), 1.), [a]))

### tf.cond

cond(*args, **kwargs)

```python
x = tf.constant(2)
y = tf.constant(5)
def f1(): return tf.multiply(x, 17)
def f2(): return tf.add(y, 23)
r = tf.cond(tf.less(x, y), f1, f2)
```

### tf.where

where(condition, x=None, y=None, name=None)

返回 x 中匹配条件的索引

### tf.squared_difference

squared_difference(x, y, name=None)

 返回  (x-y).^2

>>> a = [1, 2, 4]
>>> b = [4, 5, 6]
>>> tf.squared_difference(a, b)
<tf.Tensor 'SquaredDifference:0' shape=(3,) dtype=int32>
>>> sess.run(tf.squared_difference(a, b))
array([9, 9, 4], dtype=int32)


### tf.gather

gather(params, indices, validate_indices=None, name=None)

剪切  params 的 indices 维度

>>> a = [[1, 2, 3], [4, 5, 6]]
>>> sess.run(tf.gather(a, 1))
array([4, 5, 6], dtype=int32)
>>> sess.run(tf.gather(a, 0))
array([1, 2, 3], dtype=int32)
>>> sess.run(tf.gather(a, [0, 1]))
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)

>>> b = [[[1, 2, 3], [4, 5, 6]], [[1,1,1], [2,2,2]]]
>>> sess.run(tf.gather(a, [[0], [1]]))
array([[[1, 2, 3]],

       [[4, 5, 6]]], dtype=int32)

>>> print tf.gather(a, [[0], [1]]).get_shape()
(2, 1, 3)
>>> print tf.gather(a, [[0], [1]]).shape
(2, 1, 3)


### tf.reduce_prod

reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

>>> c = [[[1, 2, 3]], [[4, 5, 6]]]
>>> sess.run(tf.reduce_prod(c))
720
>>> sess.run(tf.reduce_prod(c, 0))
array([[ 4, 10, 18]], dtype=int32)
>>> sess.run(tf.reduce_prod(c, 1))
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)


### tf.squared_difference

squared_difference(x, y, name=None)

return (x-y)^2


### tf.one_hot

one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)

>>> a = [0, 1, 2, 2, 1, 0]
>>> sess.run(tf.one_hot(a, 3, 1, 0, 0))
array([[1, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0]], dtype=int32)
>>> sess.run(tf.one_hot(a, 3, 1, 0, -1))
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]], dtype=int32)
>>> a = [[0, 1, 2], [2, 1, 0]]
>>> sess.run(tf.one_hot(a, 3, 1, 0, -1))
array([[[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],

       [[0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]]], dtype=int32)
>>> sess.run(tf.one_hot(a, 3, 1, 0, 0))
array([[[1, 0, 0],
        [0, 0, 1]],

       [[0, 1, 0],
        [0, 1, 0]],

       [[0, 0, 1],
        [1, 0, 0]]], dtype=int32)

#### tf.setdiff1d

setdiff1d(x, y, index_dtype=tf.int32, name=None)

返回在 x 不在 y 的元素及其在 x 中的索引

>>> x = [1, 2, 3, 4, 5, 6]
>>> y = [1, 3, 5]
>>> sess.run(tf.setdiff1d(x, y))
ListDiff(out=array([2, 4, 6], dtype=int32), idx=array([1, 3, 5], dtype=int32))


#### tf.dynamic_stitch


dynamic_stitch(indices, data, name=None)

其中 indices 保存的是 合并后的索引，data 与是 indices 索引相同的对应的值

    ```python
        # Apply function (increments x_i) on elements for which a certain condition
        # apply (x_i != -1 in this example).
        x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
        condition_mask=tf.not_equal(x,tf.constant(-1.))
        partitioned_data = tf.dynamic_partition(
            x, tf.cast(condition_mask, tf.int32) , 2)
        partitioned_data[1] = partitioned_data[1] + 1.0
        condition_indices = tf.dynamic_partition(
            tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
        x = tf.dynamic_stitch(condition_indices, partitioned_data)
        # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
        # unchanged.
    ```

    ```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
    ```
合并之后的 merge[6] = [61,62], merge[4] = [41, 42] merge[1] = [11,12]





### tf.test.is_gpu_available()

如果只会 gpu 返回  True,
如果不支持 gpu, 返回  False

### tf.clip_by_value

clip_by_value(t, clip_value_min, clip_value_max, name=None)

将 t 的范围限定在 [clip_value_min, clip_value_max] 之间，
小于 clip_value_min 的值取  clip_value_min，大于 clip_value_max
的值取 clip_value_max

### tf.map_fn

map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)


>>> elems = np.array([1, 2, 3, 4, 5, 6])
>>> square = lambda x: x * x
>>> sess.run(tf.map_fn(square, elems))
array([ 1,  4,  9, 16, 25, 36])

>>> elems = np.array([1, 2, 3, 4, 5, 6])
>>> alternate = lambda x: (-x, x)
>>> sess.run(tf.map_fn(alternate, elems, dtype=(tf.int64, tf.int64)))
(array([-1, -2, -3, -4, -5, -6]), array([1, 2, 3, 4, 5, 6]))

#### 打印内存变量

```python

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for name, value in zip(tf.global_variables(), sess.run(tf.global_variables())):
        print name, ': \n', value, '\n'
```

比如

```python
print 'my/BatchNorm/beta:0', (sess.run('my/BatchNorm/beta:0'))
```
#### tf.py_func

inputs = 1.
my_func = lambda x : np.sinh(x)
y = tf.py_func(my_func, [inputs], tf.float32)


# TODO

tf.Coordinator
tf.train.Saver
tf.parse_single_example
tf.image.resize_image_with_crop_or_pad
tf.random_crop


## tips

reshape(-1) : 将一个多维矩阵转为一维向量



附录

```
>>> a = tf.reshape(tf.range(5), (5,1))
>>> b = tf.reshape(tf.range(5, 10), (5,1))
>>> c = tf.cast(tf.ones((5, 1)), tf.int32)
>>> d = tf.cast(tf.zeros((5, 1)), tf.int32)
>>> sess.run(tf.concat([a, b, c, d], 1))
array([[0, 5, 1, 0],
       [1, 6, 1, 0],
       [2, 7, 1, 0],
       [3, 8, 1, 0],
       [4, 9, 1, 0]], dtype=int32)
>>> sess.run(tf.concat([a, b, c, d], 0))
array([[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9],
       [1],
       [1],
       [1],
       [1],
       [1],
       [0],
       [0],
       [0],
       [0],
       [0]], dtype=int32)
>>> sess.run(tf.stack([a, b, c, d], 0))
array([[[0],
        [1],
        [2],
        [3],
        [4]],

       [[5],
        [6],
        [7],
        [8],
        [9]],

       [[1],
        [1],
        [1],
        [1],
        [1]],

       [[0],
        [0],
        [0],
        [0],
        [0]]], dtype=int32)
>>> sess.run(tf.stack([a, b, c, d], 1))
array([[[0],
        [5],
        [1],
        [0]],

       [[1],
        [6],
        [1],
        [0]],

       [[2],
        [7],
        [1],
        [0]],

       [[3],
        [8],
        [1],
        [0]],

       [[4],
        [9],
        [1],
        [0]]], dtype=int32)
>>> sess.run(tf.stack([a, b, c, d]))
array([[[0],
        [1],
        [2],
        [3],
        [4]],

       [[5],
        [6],
        [7],
        [8],
        [9]],

       [[1],
        [1],
        [1],
        [1],
        [1]],

       [[0],
        [0],
        [0],
        [0],
        [0]]], dtype=int32)
```
