
需要明白

1. 哪些可以用于 training, 哪些可以用于  evalution
1. 回归时用哪些
2. 多分类的时候用哪些
3. 二分类的时候用哪些


def \_num_present(losses, weights, per_batch=False)

1. broadcasting weights  使得与 losses  的  shape 一致
2. 返回 weights 不为 0 的元素个数

def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. 将  losses 和  weights 都转为 float32 类型，并相乘得到 value
2. 如果 reduction 为  Reduction.None, res = value
   如果 reduction 为  Reduction.MEAN, res = value / tf.reduce_sum(weights)
   如果 reduction 为  Reduction.SUM_BY_NONZERO_WEIGHTS, res = value / \_num_present(weights)
3. 将 res 转为输入 loss.dtype 类型
4. 将 res 加入 loss_collection


def absolute_difference(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

等于 compute_weighted_loss(tf.abs(predictions  - labels), weights, loss_collection, reduction)

def cosine_distance(
    labels, predictions, dim=None, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. value = 1 - tf.reduce_sum(labels * predictions, dim, keep_dims=True)
2. 返回 compute_weighted_loss(value, weights, loss_collection, reduction)

这个好像有问题，与 costine distance 的定义不符合

def hinge_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. value = tf.nn.relu(1 - (2 * labels - 1) * logits)
2. 返回 compute_weighted_loss(value, weights, loss_collection, reduction)

def huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)


1. abs_error = tf.abs(predictions - labels)
2. quadratic = tf.minimum(abs_error, delta)
3. value = 0.5 * quadratic**2 + (abs_error - quadratic) * delta
4. 返回 compute_weighted_loss(value, weights, loss_collection, reduction)

参考 https://en.wikipedia.org/wiki/Huber_loss

def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None,
             loss_collection=ops.GraphKeys.LOSSES,
             reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. value = labels * tf.log(predictions + epsilon) - (1 - labels) * tf.log( 1 - predictions + epsilon)
2. 返回 compute_weighted_loss(value, weights, loss_collection, reduction)

def mean_pairwise_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES)

1. diff = predictions - labels
2. reduction_indices = tf.range(1, tf.rank(diff))
3. num_present = \_num_present(diffs, weights, per_batch=True)
4. value = tf.reduce_sum(2.0 * (tf.reduce_sum(diff ** 2) / num_present - tf.reduce_sum(diff) / num_present) * weights)
5. if tf.reduce_sum(num_present) > 0 return value, else return tf.zeros_like(value)

比如
labels=[a, b, c] predictions=[x, y, z]
loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

这里文档应该是有问题的，和代码描述的不一致

def mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. value = (predictions - labels).^2
2. 返回 compute_weighted_loss(value, weights, loss_collection, reduction)

def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. if label_smoothing > 0 multi_class_labels = multi_class_labels * (1 - label_smoothing) + 0.5 * label_smoothing
2. loss = tf.nn.sigmoid_cross_entropy_with_logits(multi_class_labels, logits)
3. 返回 compute_weighted_loss(loss, weights, loss_collection, reduction)

def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

1. num_classes = tf.shape(onehot_labels)[1]  # labels shape 为 [batch_size, num_classes]
2. if label_smoothing > 0 onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
3. loss = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
4. compute_weighted_loss(loss, weights, scope, loss_collection, reduction)

def sparse_softmax_cross_entropy(
    labels, logits, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
返回 compute_weighted_loss(loss, weights, loss_collection, reduction)

def log_poisson_loss(targets, log_input, compute_full_loss=False, name=None)


```
value = exp(log_input) - targets * log_input
if compute_full_loss is Fase or zeros < targets < ones return value
else return value + targets * log(targets) - targets + 0.5 * log(2 * pi * targets)
```

```
    c = log_input,  z = targets
        -log(exp(-x) * (x^z) / z!)
      = -log(exp(-x) * (x^z)) + log(z!)
      ~ -log(exp(-x)) - log(x^z) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
          [ Note the second term is the Stirling's Approximation for log(z!).
            It is invariant to x and does not affect optimization, though
            important for correct relative loss comparisons. It is only
            computed when compute_full_loss == True. ]
      = x - z * log(x) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
      = exp(c) - z * c [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
```

def sigmoid_cross_entropy_with_logits(\_sentinel=None, labels=None, logits=None, name=None)

if logits > zeros : return logits - labels * logits + tf.log(1 + tf.exp(-logits))
else return  - labels * logits + tf.log( 1 + tf.exp(logits))

```
  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))
```

def weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None)




一般的公式定义为

  targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))

加入 pos_weight 之后，变为

  targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))

推导过程

其中 `x = logits` `z = targets` `q = pos_weight`

```
  qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
= (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
```

其中，

当 x 接近无穷正数的时候，后面部分接近于 0
当 x 接近无穷负数的时候，后面部分接近于 无穷大

为了确保稳定性和溢出，修改为

```
(1 - z) * x + (1 + (q - 1) * z) * (log(1 + exp(-abs(x))) + max(-x, 0))
```

def relu_layer(x, weights, biases, name=None

return tf.relu(x * weights + biases)

def l2_normalize(x, dim, epsilon=1e-12, name=None)

return x / tf.sqrt(tf.maximum(tf.reduce_sum(x**2, dim), epsilon))

def zero_fraction(value, name=None)

value 中 0 的百分比

return tf.reduce_mean(tf.cast(tf.equal(value, zeros), tf.float32)

def depthwise_conv2d(input,
                     filter,
                     strides,
                     padding,
                     rate=None,
                     name=None,
                     data_format=None)

TODO

def separable_conv2d(input,
                     depthwise_filter,
                     pointwise_filter,
                     strides,
                     padding,
                     rate=None,
                     name=None,
                     data_format=None)

TODO

def sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None)

https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

1. count *= v for v in axes
2. ms = tf.reduce_sum(x - shift)
3. vs = tf.reduce_sum((x - shift)^2)
return count, ms, vs

def normalize_moments(counts, mean_ss, variance_ss, shift, name=None)

return mean_ss /counts + shift, variance_ss - (mean_ss /counts)^2 + shift

def moments(x, axes, shift=None, name=None, keep_dims=False)

mean = tf.reduce_mean(x, axes)
variance = tf.reduce_mean((y - tf.stop_gradient(mean))^2, axes)
return mean, variance

def weighted_moments(x, axes, frequency_weights, name=None, keep_dims=False)

weighted_mean = tf.reduce_mean(x * frequency_weights, axes) / tf.reduce_sum(frequency_weights, axes)
weighted_variance = tf.reduce_sum(frequency_weights * (x - weighted_mean)^2) / tf.reduce_sum(frequency_weights, axes)
return weighted_mean, weighted_variance

def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)

inv = tf.sqrt(variance + variance_epsilon) * scale
return x * inv + (offset - mean * inv)

参考 http://arxiv.org/abs/1502.03167

def fused_batch_norm(x, scale, offset, mean=None, variance=None,
    epsilon=0.001, data_format="NHWC", is_training=True, name=None)

TODO gen_nn_ops._fused_batch_norm

参考 http://arxiv.org/abs/1502.03167

def batch_norm_with_global_normalization(t, m, v, beta, gamma, variance_epsilon, scale_after_normalization, name=None)

等于  batch_normalization

def \_sum_rows(x)

对 x  的每行求和

def \_compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None)

TODO

def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss")

TODO



参考 [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf)
   [Candidate Sampling Algorithms Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

def sampled_softmax_loss(weights,
                      biases,
                      labels,
                      inputs,
                      num_sampled,
                      num_classes,
                      num_true=1,
                      sampled_values=None,
                      remove_accidental_hits=True,
                      partition_strategy="mod",
                      name="sampled_softmax_loss")

TODO
