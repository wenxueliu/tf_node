


broadcasting 如何理解?

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

### tf.random_normal

random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

与 np.random.normal 的包装

### tf.squeeze

squeeze(input, axis=None, name=None, squeeze_dims=None)

>>> v = tf.constant(1.0, shape=[1, 2, 1, 3, 1, 1])
>>> sess.run(tf.shape(tf.squeeze(v)))
array([2, 3], dtype=int32)
>>> sess.run(tf.shape(tf.squeeze(v,
>>> [2,4])))
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

### tf.cast

cast(x, dtype, name=None)

>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> sess = tf.Session()
>>> sess.run(tf.cast(a, tf.int32))
array([ 0,  2,  2,  1, -4], dtype=int32)

### tf.equal

equal(x, y, name=None)

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

可以从两方面理解



>>> x = tf.constant([[11,12,13,14], [21,22,23,24], [31,32,33,4], [41,42,43,44]])
>>> t = tf.convert_to_tensor(x)
>>> sess = tf.Session()
>>> sess.run(tf.slice(t, [1, 1], [3, 2]))
array([[22, 23],
       [32, 33],
       [42, 43]], dtype=int32)

>>> z=tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]],
>>> [[13,14,15],[16,17,18]]])
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

## tf.train.string_input_producer

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

## TODO

tf.Coordinator
tf.train.Saver
tf.parse_single_example
tf.image.resize_image_with_crop_or_pad
tf.random_crop



