

### BoxList

对 [y_min, x_min, y_max, x_max] 的封装，

注
1. 注意顺序
2. height = y_max - y_min, width = x_max - x_min

其中 self.data['boxes'] 保存了所有 box

1. 修改和获取 box
2. 支持 box 输出其他格式如 [y_center, x_center, height, width], [x_min, y_min, x_max, y_max]，
3. self.data 支持增加自定义 key

#### 提供的操作

* area(boxlist, scope=None): 计算 boxlist 中每个 box 的面积, (ymax - ymin) * (xmax - xmin)
* height_width(boxlist, scope=None): 计算 boxlist 中每个 box 的面积, (ymax - ymin) * (xmax - xmin)
* scale(boxlist, y_scale, x_scale, scope=None): 将 boxlist 的 y, x 分别伸缩指定比例
* gather(boxlist, indices, fields=None, scope=None) : 获取 boxlist 指定的索引组成的子集
* clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None) : 取 boxlist 与 window 的交集，默认会去掉交集为 0 的 box
* prune_outside_window(boxlist, window, scope=None) :  将 boxlist 有任意部分在 window 外的元素删除，包含与 window 完全重合的元素
* prune_completely_outside_window(boxlist, window, scope=None) : 将 boxlist  所有角都在 window 外的元素删除，不包含与 window 相同的元素
* intersection(boxlist1, boxlist2, scope=None) : 计算 boxlist1 的每个 box 与 boxlist2 的每个 box 对应的交叉面积，返回 [N, M]
* matched_intersection(boxlist1, boxlist2, scope=None) : 计算  boxlist1 和 boxlist2  对应元素的交集的面积
* iou(boxlist1, boxlist2, scope=None) : 计算 boxlist1 和 boxlist2 对应元素的 IoU
* matched_iou(boxlist1, boxlist2, scope=None) : 计算 boxlist1 和 boxlist2 对应元素的 IoU
* ioa(boxlist1, boxlist2, scope=None) : 计算 boxlist1 和 boxlist2 对应元素的 IoA
* prune_non_overlapping_boxes(boxlist1, boxlist2, min_overlap=0.0, scope=None) : 保留 boxlist1 中 boxlist1 和 boxlist2 的 ioa 大于  min_overlap 的元素
* prune_small_boxes(boxlist, min_side, scope=None) : 将 boxlist 中元素 e, 如果 e 长小于 min_side 的长或 e 的宽 小于 min_side 的宽删除
* change_coordinate_frame(boxlist, window, scope=None) : boxlist 中元素 e - window 之后, 高度除以 window.height,  宽度除以 window.width
* sq_dist(boxlist1, boxlist2, scope=None) :  计算 boxlist1  的每个  box 与 boxlist2  每个 box 的平方差，返回 [N, M] 矩阵
* boolean_mask(boxlist, indicator, fields=None, scope=None) : 获取 indicator  中对应为 Ture 的索引在 boxlist 中的元素
* concatenate(boxlists, fields=None, scope=None) : 将 boxlists 中的每个  boxlist 连起来
* sort_by_field(boxlist, field, order=SortOrder.descend, scope=None) : 对 boxlist 的 field 进行排序
* visualize_boxes_in_image(image, boxlist, normalized=False, scope=None) : 将 boxlist 在  image 中画出来
* filter_field_value_equals(boxlist, field, value, scope=None) : 从 boxlist 中过滤 field 的值中与 value 相同
* filter_greater_than(boxlist, thresh, scope=None) : 从 boxlist 的 score 中找到比 thresh 大的元素
* non_max_suppression(boxlist, thresh, max_output_size, scope=None) : 从  boxlist 中找到 thresh 的面积
* to_normalized_coordinates(boxlist, height, width, check_range=True, scope=None) : boxlist 中高度除以 height, 宽度除以 width
* to_absolute_coordinates(boxlist, height, width, check_range=True, maximum_normalized_coordinate=1.01, scope=None) : boxlist 中高度除以 height, 宽度除以 width
* box_voting(selected_boxes, pool_boxes, iou_thresh=0.5) : 
* pad_or_clip_box_list(boxlist, num_boxes, scope=None) :  将 boxlist 中元素设置为 num_boxes, 少了添加 0，多了截取
* refine_boxes_multi_class(pool_boxes, num_classes, nms_iou_thresh, nms_max_detections, voting_iou_thresh=0.5):TODO


refine_boxes(pool_boxes, nms_iou_thresh, nms_max_detections, voting_iou_thresh=0.5)

1. 对 pool_boxes  以 nms_iou_thresh 为阈值计算 nms
2. TODO

box_voting(selected_boxes, pool_boxes, iou_thresh=0.5)

TODO

IoU : a, b交叉面积/ (a, b面积和 - a, b交叉面积)
IoA : a, b交叉面积/ b的面积


### boxCoder

BoxCoder 抽象类，具体包含如下四种实现

* FasterRcnnBoxCoder
* KeypointBoxCoder
* MeanStddevBoxCoder
* SquareBoxCoder

#### FasterRcnnBoxCoder

self._scale_factors

encode(self, boxes, anchors)

计算 boxes 相对于 anchors 之间的偏移，并用 anchors 归一化

decode(self, rel_codes, anchors)

将归一化的相对 ancher 偏移的 rel_codes 变为实际的值


### box predictor

#### ConvolutionalBoxPredictor

* self.num_classes : 分类数量
* self._conv_hyperparams = conv_hyperparams
* self._min_depth : 控制 1x1 卷积核的个数最小值
* self._max_depth : 控制 1x1 卷积核的个数最大值
* self._num_layers_before_predictor = feature map 到 predictor 经过 1x1 卷积的层数
* self._use_dropout : 是否用 dropout
* self._kernel_size : 最后一个卷积核大小
* self._box_code_size : box 的大小
* self._dropout_keep_prob : dropout 参数
* self._apply_sigmoid_to_scores : 分类的时候是否经过 sigmoid
* self._class_prediction_bias_init = class_prediction_bias_init

predict(self, image_features, num_predictions_per_location)

image_features : [batch_size, height, width, channel]
num_predictions_per_location : 每个位置 anchor 数量

1. 经过 self._num_layers_before_predictor 个 1x1 卷积，数量为 image_features

box 位置预测

2. 经过 self._kernel_size * self._kernel_size 的卷积，数量为 num_predictions_per_location * self._box_code_size
3. 将 2 的输出维度修改为 （batch_size, height * width * num_predictions_per_location, 1, self._box_code_size)

box 分类预测

2. 经过 dropout 层，参数为  self._dropout_keep_prob
3. 经过 self._kernel_size * self._kernel_size 的卷积，数量为 num_predictions_per_location * self.num_classes
4. 经过 sigmoid, 该步可选.
5. 将 3 的输出维度修改为 （batch_size, height * width * num_predictions_per_location, self.num_classes + 1)

返回字典 {"box_coding": box_coding, 'class_predictions_with_background': class_predictions_with_background}

#### MaskRCNNBoxPredictor

* self.num_classes
* self._fc_hyperparams = fc_hyperparams
* self._use_dropout = use_dropout
* self._box_code_size = box_code_size
* self._dropout_keep_prob = dropout_keep_prob
* self._conv_hyperparams = conv_hyperparams
* self._predict_instance_masks = predict_instance_masks
* self._mask_height = mask_height
* self._mask_width = mask_width
* self._mask_prediction_conv_depth = mask_prediction_conv_depth
* self._predict_keypoints = predict_keypoints


predict(self, image_features, num_predictions_per_location)

1. 将 image_features 展开为一维
2. 通过 dropout 层，参数为  self._dropout_keep_prob

box 位置预测

3.1 经过全连接层，数量 self._num_classes * self._box_code_size
3.2 维度变为 [-1, 1, self._num_classes, self._box_code_size]

box 分类预测

3.1 经过全连接层，数量 self._num_classes + 1
3.2 [-1, 1, self._num_classes + 1]

mask 预测

1. 对图片进行二分插值
2. 经过 2x2 的卷积，数量 self._mask_prediction_conv_depth
3. 进行维度转换

返回字典 {'box_coding': box_coding,
          'class_predictions_with_background': class_predictions_with_background,
          'mask_predictions': instance_masks}

#### RfcnBoxPredictor

TODO

predict(self, image_features, num_predictions_per_location, proposal_boxes)

*** self._conv_hyperparams = conv_hyperparams
*** self._num_spatial_bins = num_spatial_bins
*** self._depth = depth
*** self._crop_size = crop_size
*** self._box_code_size = box_code_size


box 位置预测

1. 经过 1x1 的卷积，个数为 self._num_spatial_bins[0] * self._num_spatial_bins[1] * self.num_classes * self._box_code_size
2. position_sensitive_crop_regions
