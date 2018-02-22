


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

### tf.greater

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

将 input[i] 重复  multiples[i] 倍

multiples  必须为整数，且元素个数必须与 input 的维度相同

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

剪切  params 的  indices 维度

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


### tf.one_host

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


### tf.concat

concat(values, axis, name='concat')

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

# TODO

tf.Coordinator
tf.train.Saver
tf.parse_single_example
tf.image.resize_image_with_crop_or_pad
tf.random_crop



