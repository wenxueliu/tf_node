


broadcasting 如何理解?

### tf.placeholder

placeholder(dtype, shape=None, name=None)

x = tf.placeholder(tf.float32, shape=(1024, 1024))

### tf.Variable


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

创建一个 FIFO 队列

##

## TODO

tf.Coordinator
tf.train.Saver
tf.parse_single_example

