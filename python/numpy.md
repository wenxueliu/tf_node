
文档

axis
空: 所有元素
0: 每列
1: 每行

为了避免广播导致潜在的问题，在创建矩阵的时候，应要指定维度，避免 (n,)
这种维度导致的潜在 bug，并通过 assert 来确认维度。


用 np.random.rand(10,1) 而不用 np.random.rand(10)

快速构建一个矩阵

np.arange(121 * 3).reshape([3,11,11])
np.ones(121 * 3).reshape([3,11,11])

## 初始化

### np.full

生成一个矩阵，并赋值默认值

full(shape, fill_value, dtype=None, order='C')

>>> np.full((2, 2), np.inf)
array([[ inf,  inf],
        [ inf,  inf]])
>>> np.full((2, 2), 10)
array([[10, 10],
       [10, 10]])

### np.ones

ones(shape, dtype=None, order='C')

生成全 1

>>> np.ones(5)
array([ 1.,  1.,  1.,  1.,  1.])
>>> np.ones(5, dtype=np.int)
array([1, 1, 1, 1, 1])
>>> np.ones((2,3), dtype=np.int)
array([[1, 1, 1],
       [1, 1, 1]])
>>> np.ones((2,3), dtype=np.int, order='C')
array([[1, 1, 1],
       [1, 1, 1]])
>>> np.ones((2,3), dtype=np.int, order='F')
array([[1, 1, 1],
       [1, 1, 1]])

### np.empty_like

创建一个与已有某个矩阵一样 shape 的矩阵，并不初始化

empty_like(a, dtype=None, order='K', subok=True)

* ones_like
* zeros_like

>>> a = ([1,2,3], [4,5,6])
>>> b = np.empty_like(a)
>>> b
array([[5764607523034234880, 5764607523034234880,          4473356328],
       [         4472177928, 5764607523034234880, 5764607523034234880]])

>>> b = np.ones_like(a)
>>> b
array([[1, 1, 1],
       [1, 1, 1]])
>>> b = np.zeros_like(a)
>>> b
array([[0, 0, 0],
       [0, 0, 0]])


>>> x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
>>> v = np.array([1, 0, 1])
>>> y = np.empty_like(x)
>>> for i in range(len(x)):
...   y[i,:] = x[i,:] + v
...
>>> y
array([[ 2,  2,  4],
       [ 5,  5,  7],
       [ 8,  8, 10],
       [11, 11, 13]])

### np.arange

arange([start,] stop[, step,], dtype=None)

生成指定范围的，左闭右开，经常与 reshape 用于快速创建矩阵

When using a non-integer step, such as 0.1, the results will often not be consistent.  It is better to use ``linspace`` for these cases.

>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(1.0, 3.0)
array([ 1.,  2.])
>>> np.arange(1.0, 3.0, 0.5)
array([ 1. ,  1.5,  2. ,  2.5])
>>> np.linspace(1.0, 3.0, 4)
array([ 1.        ,  1.66666667,  2.33333333,  3.        ])
>>> np.linspace(1.0, 4.0, 4)
array([ 1.,  2.,  3.,  4.])

np.arange(121 * 3).reshape([3,11,11])
np.ones(121 * 3).reshape([3,11,11])

### 类型转换

astype

### 行列切换

>>> b = np.arange(12).reshape([2,2,3])
>>> b
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> swap = (0,2,1)
>>> b.transpose(swap)
array([[[ 0,  3],
        [ 1,  4],
        [ 2,  5]],

       [[ 6,  9],
        [ 7, 10],
        [ 8, 11]]])


## 矩阵操作

### broadcast

>>> x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
>>> v = np.array([1, 0, 1])
>>> y=x+v
>>> y
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]

Functions that support broadcasting are known as universal functions.


### np.tile

tile(A, reps)

将 A 重复

>>> np.tile(v, [4, 1])
array([[1, 0, 1],
       [1, 0, 1],
       [1, 0, 1],
       [1, 0, 1]])

>>> np.repeat(v, 4, axis=0).reshape(3,4)
array([[1, 1, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1]])
>>> np.repeat(v, 4, axis=0).reshape(3,4).T
array([[1, 0, 1],
       [1, 0, 1],
       [1, 0, 1],
       [1, 0, 1]])

### np.repeat

repeat(a, repeats, axis=None)

重复某个矩阵以某个维度重复 repeats

>>> x = np.array([[1,2],[3,4]])
>>> np.repeat(x, 2)
array([1, 1, 2, 2, 3, 3, 4, 4])
>>> np.repeat(x, 2, axis=0)
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, 2, axis=1)
array([[1, 1, 2, 2],
       [3, 3, 4, 4]])
>>> np.repeat(x, [1,2 ], axis=1)
array([[1, 2, 2],
       [3, 4, 4]])
>>> np.repeat(x, [1,2 ], axis=0)
array([[1, 2],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [1,3], axis=0)
array([[1, 2],
       [3, 4],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [0,2], axis=0)
array([[3, 4],
       [3, 4]])
>>> np.repeat(x, [0,3], axis=0)
array([[3, 4],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [2,3], axis=0)
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [2,2], axis=0)
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [3,2], axis=0)
array([[1, 2],
       [1, 2],
       [1, 2],
       [3, 4],
       [3, 4]])
>>> np.repeat(x, [3,2], axis=1)
array([[1, 1, 1, 2, 2],
       [3, 3, 3, 4, 4]])
>>> np.repeat(x, [3,3], axis=1)
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])

### np.linspace

生成指定范围的一维数组，类似于 python 中的 range, 没有 step 而是 num，注意
endpoint 对结果的影响

linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

>>> import nump as np
>>> np.linspace(1,10,10)
array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
>>> np.linspace(1,10,10, retstep = True)
(array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.]), 1.0)
>>> np.linspace(1,10,10, endpoint=False, retstep = True)
(array([ 1. ,  1.9,  2.8,  3.7,  4.6,  5.5,  6.4,  7.3,  8.2,  9.1]), 0.90000000000000002)

### np.random.normal

高斯分布

normal(loc=0.0, scale=1.0, size=None)

>>> np.random.normal(0, 1, [2,2])
array([[ 0.02640879,  1.61693267],
       [-0.17924483, -0.53904041]])
>>> np.random.normal(0, 1, 10)
array([ 0.76485244,  1.1470795 , -0.52399549,  0.44193735, -2.47316263,
       -0.97565456,  0.83487939, -0.56358797,  0.28430778, -0.49732677])
>>> np.random.normal(0, 1, [1, 2, 3])
array([[[ 0.24176915, -0.07871348, -0.47244455],
        [ 1.4323179 , -1.00614932, -0.35050296]]])


### np.matrix

a = np.matrix('1 2; 3 4')
np.matrix([[1, 2], [3, 4]])

元素相等

x = np.matrix(np.arange(12).reshape((3,4))); x
y = x[0]; y
(x == y)
(x == y).all(0)
(x == y).all(1)

(x == y).any(0)
(x == y).any(1)

注意与 np.any, np.all 的关系

最大最小索引

x = np.matrix(np.arange(12).reshape((3,4))); x
x.argmax()
x.argmax(1)

x.argmin()
x.argmin(1)

注意与 np.argmax, np.argmin 的关系

展开

m = np.matrix([[1,2], [3,4]])
m.flatten()
m.flatten('F')

转为 ndarray

x = np.matrix(np.arange(12).reshape((3,4))); x
x.getA()
x.getA1()

x = np.matrix(np.arange(12).reshape((3,4)))
z = x - 1j*x; z
z.getH()  //(complex) conjugate transpose

逆矩阵

m = np.matrix('[1, 2; 3, 4]'); m
m.getI()
m.getI() * m

矩阵转置

m = np.matrix('[1, 2; 3, 4]')

最大

x = np.matrix(np.arange(12).reshape((3,4))); x
x.max()
x.max(0)
x.min()
x.min(0)
x.mean()
x.mean(0)
x.prod()
x.prod(0)
x.prod(1)
x.ptp()
x.ptp(0)
x.ptp(1)
x.ravel()
x.ravel('C')
x.ravel('F')
x.ravel('A')
x.ravel('K')
x.squeeze() //TODO
x.std()
x.std(0)
x.std(1)
x.sum()
x.sum(0)
x.sum(1)
x.tolist() //转为 list
x.var()
x.var(0)
x.var(1)
x.astype(np.float)
x.choose([1, 3])

A = np.array([1, 256, 8755], dtype=np.int16)
map(hex, A)
A.byteswap(True)
map(hex, A)



reshape
resize

### np.reshape

a = np.array([[1,2,3], [4,5,6]])
np.reshape(a, 6)
np.reshape(a, 6, order='F')
a = np.arange(6).reshape((3, 2))
np.reshape(a, (2, 3))
np.reshape(np.ravel(a), (2, 3))
np.reshape(a, (2, 3), order='F')
np.reshape(np.ravel(a, order='F'), (2, 3), order='F')

#### np.ravel

>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> print(x.ravel())
[1 2 3 4 5 6]

#### np.clip

clip(a, a_min, a_max, out=None)

将 a 中每个元素限制在 a_min, a_max，小于  a_min 的设置为  a_min, 大于  a_max
的设置为 a_max

>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, 2, 6)
array([2, 2, 2, 3, 4, 5, 6, 6, 6, 6])
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, 2, 6, a)
array([2, 2, 2, 3, 4, 5, 6, 6, 6, 6])
>>> a
array([2, 2, 2, 3, 4, 5, 6, 6, 6, 6])

#### np.roll

roll(a, shift, axis=None)

>>> x = np.arange(10)
>>> np.roll(x, 2)
array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

>>> x2 = np.reshape(x, (2,5))
>>> x2
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> np.roll(x2, 1)
array([[9, 0, 1, 2, 3],
       [4, 5, 6, 7, 8]])
>>> np.roll(x2, 1, axis=0)
array([[5, 6, 7, 8, 9],
       [0, 1, 2, 3, 4]])
>>> np.roll(x2, 1, axis=1)
array([[4, 0, 1, 2, 3],
       [9, 5, 6, 7, 8]])

### np.transpose

转置


### np.column_stack

>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.column_stack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])

#### np.hstack

hstack(tup) 水平扩展

>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.hstack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])

### np.random.permutation

permutation(...)

生成随机序列

>>> np.random.permutation(np.arange(10))
array([2, 0, 1, 3, 4, 8, 5, 9, 7, 6])
>>> np.random.permutation(10)
array([7, 5, 6, 9, 3, 8, 1, 4, 0, 2])

### np.random.choice

choice(a, size=None, replace=True, p=None)

从 a 中随机采样生成 shape 为 size 的矩阵或向量

>>> np.random.choice(5, 3)
array([1, 4, 4])
>>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
array([0, 0, 2])
>>> np.random.choice(np.arange(10), 4)
array([5, 9, 9, 6])
>>> np.random.choice(np.arange(10), (2,5))
array([[9, 6, 0, 7, 7],
       [4, 0, 4, 0, 0]])
>>> np.random.choice(np.arange(10), (2,2))
array([[2, 0],
       [6, 6]])
>>> np.random.choice(10, (2,2))
array([[9, 8],
       [7, 6]])

>>> b = set(np.random.choice(len(a), 5, replace=False))
>>> b
set([0, 9, 2, 4, 5])
>>> b = set(np.random.choice(len(a), 7, replace=False))
>>> b
set([2, 3, 4, 5, 6, 7, 8])
>>> b = set(np.random.choice(len(a), 7))
>>> b
set([8, 9, 5, 6, 7])
>>> b = set(np.random.choice(len(a), 5))
>>> b
set([0, 6, 7])

### np.c_


>>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
array([[1, 2, 3, 0, 0, 4, 5, 6]])

## 矩阵操作

### 矩阵切片(slice)

指定范围与不指定范围的区别

>>> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> row_r1 = a[1, :]
>>> row_r2 = a[1:2, :]
>>> print(row_r1, row_r1.shape)
(array([5, 6, 7, 8]), (4,))
>>> print(row_r2, row_r2.shape)
(array([[5, 6, 7, 8]]), (1, 4))

索引为数组

>>> a = np.array([[1,2], [3, 4], [5, 6]])
>>> print(a[[0, 1, 2], [0, 1, 0]])
[1 4 5]
>>> print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
[1 4 5]

>>> a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
>>> b = np.array([0, 2, 0, 1])
>>> print(a[np.arange(4), b])
[ 1  6  7 11]
>>> a[np.arange(4), b] += 10
>>> print(a)
[[11  2  3]
 [ 4  5 16]
 [17  8  9]
 [10 21 12]]

### 矩阵条件判断

>>> a = np.array([[1,2], [3, 4], [5, 6]])
>>> bool_idx = (a > 2)
>>> print(bool_idx)
[[False False]
 [ True  True]
 [ True  True]]
 >>> print(a[bool_idx])
[3 4 5 6]
>>> print(a[a > 2])
[3 4 5 6]

### Elementwise product vs Inner product

>>> x = np.array([[1,2],[3,4]])
>>> y = np.array([[5,6],[7,8]])
>>> print(x.dot(y))
[[19 22]
 [43 50]]
>>> print(np.multiply(x,y))
[[ 5 12]
 [21 32]]
>>> print(np.dot(x,y))
[[19 22]
 [43 50]]
>>> print(x*y)
[[ 5 12]
 [21 32]]

### np.np.meshgrid

meshgrid(*xi, **kwargs)

x, y = msehgrid(a, b)
其中  x  为 a 的每个元素用 b' 的数组表示，
其中  y  为 b 的每个元素用 a 的数组表示，


>>> b
[1, 2, 3]
>>> a
[1, 2]
>>> x, y = np.meshgrid(a, b)
>>> x
array([[1, 2],
       [1, 2],
       [1, 2]])
>>> y
array([[1, 1],
       [2, 2],
       [3, 3]])
>>> x, y = np.meshgrid(b, a)
>>> x
array([[1, 2, 3],
       [1, 2, 3]])
>>> y
array([[1, 1, 1],
       [2, 2, 2]])

>>> b = [[1, 2, 3], [4, 5, 6]]
>>> np.meshgrid(a, b)
[array([[0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1]]), array([[1, 1],
       [2, 2],
       [3, 3],
       [4, 4],
       [5, 5],
       [6, 6]])]



>>> nx, ny = (3, 2)
>>> x = np.linspace(0, 1, nx)
>>> y = np.linspace(0, 1, ny)
>>> xv, yv = np.meshgrid(x, y)
>>> x
array([ 0. ,  0.5,  1. ])
>>> y
array([ 0.,  1.])
>>> xv
array([[ 0. ,  0.5,  1. ],
       [ 0. ,  0.5,  1. ]])
>>> yv
array([[ 0.,  0.,  0.],
       [ 1.,  1.,  1.]])

### nan_to_num

nan_to_num(x, copy=True)

Replace nan with zero and inf with finite numbers

>>> np.set_printoptions(precision=8)
>>> np.set_printoptions(precision=2)
>>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
>>> np.nan_to_num(x)
array([  1.80e+308,  -1.80e+308,   0.00e+000,  -1.28e+002,   1.28e+002])

### np.squeeze

squeeze(a, axis=None)

删除矩阵中元素只有一个的维度

>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x)
array([0, 1, 2])
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=0).shape
(3, 1)
>>> np.squeeze(x, axis=0)
array([[0],
       [1],
       [2]])
>>> np.squeeze([ i for i in range(10) ])
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.squeeze(np.squeeze([ i for i in range(10) ]))
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


#### np.nditer
