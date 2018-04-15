

## Loss

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss
 * WeightedIOULocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * BootstrappedSigmoidClassificationLoss

### WeightedL2LocalizationLoss

Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2


compute_loss(self, prediction_tensor, target_tensor, weights)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]

### WeightedSmoothL1LocalizationLoss

```
if |x| < 1 0.5*x^2
else |x| - 0.5
其中  x = |predict - target|
```

compute_loss(self, prediction_tensor, target_tensor, weights)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]

```
abs_diff = tf.abs(prediction_tensor - target_tensor)
abs_diff_lt_1 = tf.less(abs_diff, 1)
tf.reduce_sum(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.52) * weights
```

### WeightedIOULocalizationLoss


compute_loss(self, prediction_tensor, target_tensor, weights)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]

reduce_sum((1 - IoU(predict - target)) * (weights))

### WeightedSigmoidClassificationLoss

compute_loss(self, prediction_tensor, target_tensor, weights, class_indices=None)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]

tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor) * weights


### SigmoidFocalClassificationLoss

https://arxiv.org/pdf/1708.02002.pdf

compute_loss(self, prediction_tensor, target_tensor, weights, class_indices=None)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]
* class_indices : 

per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
prediction_probabilities = tf.sigmoid(prediction_tensor)
p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
modulating_factor = 1.0
if self._gamma:
  modulating_factor = tf.pow(1.0 - p_t, self._gamma)
alpha_weight_factor = 1.0
if self._alpha is not None:
  alpha_weight_factor = (target_tensor * self._alpha +
                         (1 - target_tensor) * (1 - self._alpha))
focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                            per_entry_cross_ent)


### WeightedSoftmaxClassificationLoss


compute_loss(self, prediction_tensor, target_tensor, weights)

* prediction_tensor : [batch_size, num_anchors, code_size]
* target_tensor : [batch_size, num_anchors, code_size]
* weights : [batch_size, num_anchors]

```
    prediction_tensor = tf.divide(prediction_tensor, self._logit_scale, name='scale_logit')
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
```

### BootstrappedSigmoidClassificationLoss

Training Deep Neural Networks On Noisy Labels with Bootstrapping by Reed et al.

TODO


### HardExampleMiner

* Training Region-based Object Detectors with Online Hard Example Mining (CVPR 2016) by Srivastava et al
* SSD: Single Shot MultiBox Detector (ECCV 2016) by Liu et al


self._num_hard_examples = num_hard_examples
self._iou_threshold = iou_threshold
self._loss_type = loss_type
self._cls_loss_weight = cls_loss_weight
self._loc_loss_weight = loc_loss_weight
self._max_negatives_per_positive = max_negatives_per_positive
self._min_negatives_per_image = min_negatives_per_image
self._max_negatives_per_positive
self._num_positives_list = None
self._num_negatives_list = None

def \__call__(self, location_losses, cls_losses, decoded_boxlist_list, match_list=None):

所有窗口: decoded_boxlist_list[ind]
scores : image_loss = cls_losses[ind] * self._cls_loss_weight + location_losses[ind] * self._loc_loss_weight
max_output_size : self._num_hard_examples 或 decoded_boxlist_list[ind].num_boxes()
iou : self._iou_threshold

基于以上参数计算 nms，计算完之后，selected_indices  保留了所有希望的 box，但是由于正负样本不均衡，
一般是正样本少，负样本多，因此，根据正负样本比例，从负样本中选出一部分，最后正负样本组成整个样本。

整体的实现非常简单，就是在  nms 选择的基础之上，对负样本的数量进行了控制

def \_subsample_selection_to_desired_neg_pos_ratio(self,
                                                  indices,
                                                  match,
                                                  max_negatives_per_positive,
                                                  min_negatives_per_image=0)

indices : 全部索引
match : 保存了正负样本的全部索引
max_negatives_per_positive : 每个正样本对应几个负样本
min_negatives_per_image : 每图片最少的负样本

1. 从 match 中获取正负样本的索引
2. 根据正样本的数量计算出最大负样本的数量
3. 根据正负样本从 indices 选出采样的索引，正样本数量，负样本数量

最大负样本以 max_negatives_per_positive * num_positives 和 min_negatives_per_image 的最大值为准

