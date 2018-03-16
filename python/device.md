
在实际运行的时候，同一个 Operation   既支持 CPU  也支持 GPU  运行，GPU
优先级比 CPU 优先级高，即如果你的机器有 GPU，就在 GPU 上运行，如果
没有 GPU 才在 CPU 上运行。 这个完全就 tensorflow 自动帮你实现。

你有多个 GPU，如何将多个操作在多个 GPU 上合理分配，
就需要自己手动设置哪些操作在 GPU 上运行，哪些操作在其他 GPU 上
运行，当然，有时候，即使你有 GPU，你也希望它在 CPU 上执行（这
主要用于测试）。



### 查询可用的设备(GPU/CPU)列表
```python
>>> from tensorflow.python.client import device_lib as _device_lib
>>> [x for x in local_device_protos if x.device_type == 'CPU']
>>> [x for x in local_device_protos if x.device_type == 'GPU']
>>> [x.name for x in local_device_protos if x.device_type == 'GPU']
>>> [x.name for x in local_device_protos if x.device_type == 'CPU']
```
```python
def is_gpu_available(cuda_only=True):
  """
  code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
  Returns whether TensorFlow can access a GPU.
  Args:
    cuda_only: limit the search to CUDA gpus.
  Returns:
    True iff a gpu device of the requested kind is available.
  """
  from tensorflow.python.client import device_lib as _device_lib

  if cuda_only:
    return any((x.device_type == 'GPU')
               for x in _device_lib.list_local_devices())
  else:
    return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
               for x in _device_lib.list_local_devices())
```

tensorflow对GPU设备的编码

执行：

CUDA_VISIBLE_DEVICES=1,2  python test_util_tf.py

    1

输出为：

/gpu:0
/gpu:1

    1
    2

可以看出， 无论CUDA可见的设备是哪几个， tensorflow都会对它们从0开始重新编码。

### 哪些操作既支持 GPU 也支持 CPU 运行

### 让指定 Operation 在指定设备上运行

如果已经知道如何获取当前机器的设备信息的时候，在运行的时候，就可用自己
指定哪些 Operation 在哪些设备上运行

```python
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
结果

```
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
```

有一个问题是，有时候，你指定某个操作在某个 CPU 或 GPU 上执行，但是
当你的程序在另外一台设备运行的时候，由于该设备没有对应的 GPU 或 CPU，
正常情况下会报错，解决办法就是

```python
config.allow_soft_placement = True
```

### 如何合理分配各个操作在设备上运行

TODO

### 查看运行时，operation 和  tensor 分配到哪些设备

```python
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

### GPU 内存控制
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  // gpu 的内存随着需要逐渐增加
config.gpu_options.per_process_gpu_memory_fraction = 0.4 //只允许使用最多 40% 的 GPU 内存
session = tf.Session(config=config, ...)
```

### 参考

https://www.tensorflow.org/programmers_guide/using_gpu
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
