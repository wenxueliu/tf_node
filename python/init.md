

## 参考 paper

factor=1.0 mode='FAN_AVG' uniform=True
[Understanding the difficulty of training deep feedforward neural networks](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

factor=2.0 mode='FAN_IN' uniform=False

[Delving Deep into Rectifiers]( http://arxiv.org/pdf/1502.01852v1.pdf)

[Convolutional Architecture for Fast Feature Embedding](http://arxiv.org/abs/1408.5093) : factor=1.0 mode='FAN_IN' uniform=True

##


Initializer
    Zeros
    Ones
    Constant
    RandomUniform :  实际为 random_ops.random_uniform
    RandomNormal : 实际为 random_ops.random_normal
    TruncatedNormal :   对应 random_ops.truncated_normal
    UniformUnitScaling :
    VarianceScaling
    Orthogonal



zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
uniform_unit_scaling_initializer = UniformUnitScaling
variance_scaling_initializer = VarianceScaling
orthogonal_initializer = Orthogonal


glorot_uniform_initializer : http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
glorot_normal_initializer : http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf


variance_scaling_initializer : 有三篇 paper 支持，是 random_ops.truncated_normal
和 random_ops.random_uniform 的封装，主要体现在参数上的差别。

## 例子

```python
value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value)
x = tf.get_variable('x', shape=[2, 4], initializer=init)
```
