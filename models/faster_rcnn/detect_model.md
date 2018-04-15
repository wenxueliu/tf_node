
## DetectionModel


### FasterRCNNMetaArch

*** self._is_training = is_training
*** self._image_resizer_fn : 对输入图像进行 resize
*** self._feature_extractor : 计算特征图的函数，FasterRCNNFeatureExtractor
*** self._first_stage_only : True|False 是否只计算 RPN 网络
*** self._proposal_target_assigner
*** self._detector_target_assigner
*** self._box_coder
*** self._proposal_target_assigner.box_coder
*** self._first_stage_anchor_generator : grid_anchor_generator.GridAnchorGenerator

*** self._first_stage_box_predictor_arg_scope
*** self._first_stage_atrous_rate
*** self._first_stage_box_predictor_kernel_size : 默认 3
*** self._first_stage_box_predictor_depth : 默认 512

*** self._first_stage_minibatch_size :  每个张图片取标签的数量，默认值 256
*** self._first_stage_sampler  : 样本采集函数, 包括 BalancedPositiveNegativeSampler
*** self._first_stage_box_predictor :  实际为 ConvolutionalBoxPredictor

*** self._first_stage_nms_score_threshold : box_coding 会去掉 score 小于该值的 box，建议值为 0
*** self._first_stage_nms_iou_threshold : box_coding 计算 nms 时的 iou 阈值
*** self._first_stage_max_proposals : 计算 nms 最后剩余的最多的 box 的数量

*** self._first_stage_localization_loss : losses.WeightedSmoothL1LocalizationLoss
*** self._first_stage_objectness_loss : losses.WeightedSoftmaxClassificationLoss
*** self._first_stage_loc_loss_weight : 默认 1.0， 定位的权重
* self._first_stage_obj_loss_weight : 默认 1.0， 判断物体有无的 权重

* self._initial_crop_size : 14
* self._maxpool_kernel_size : 2
* self._maxpool_stride : 2k
* self._mask_rcnn_box_predictor
* self._second_stage_batch_size : 默认 64，必须小于第一阶段的  proposal_boxes 数量
* self._second_stage_sampler
* self._second_stage_nms_fn
* self._second_stage_score_conversion_fn

*** self._second_stage_localization_loss : losses.WeightedSmoothL1LocalizationLoss
*** self._second_stage_classification_loss : second_stage_classification_loss
*** self._second_stage_mask_loss : losses.WeightedSigmoidClassificationLoss
*** self._second_stage_loc_loss_weight : 1.0
*** self._second_stage_cls_loss_weight : 1.0
*** self._second_stage_mask_loss_weight : 1.0

*** self._hard_example_miner
*** self._parallel_iterations

* first_stage_positive_balance_fraction  : 默认 0.5
* first_stage_nms_score_threshold : 默认 0.0
* first_stage_nms_iou_threshold : 默认 0.7
* first_stage_max_proposals  : 默认 300
* second_stage_balance_fraction : 默认 0.25



#### 关于采样

第一阶段采样

1. 采样 BalancedPositiveNegativeSampler 算法采样
2. 通过 first_stage_positive_balance_fraction 控制正负比例

第二阶段采样

1. 采样 BalancedPositiveNegativeSampler 算法采样
1. 通过 second_stage_balance_fraction 控制正负比例
2. second_stage_batch_size 控制一次处理的XX数量

#### 关于 proposal

first_stage_max_proposals 控制  proposal 的最大数量

#### 关于 NMS

第一阶段

first_stage_nms_score_threshold
first_stage_nms_iou_threshold

第二阶段

second_stage_non_max_suppression_fn


## Faster_RCNN 算法流程

整个算法分三个阶段，两过程

训练

1. 预处理
2. 第一阶段 : 训练 RPN 网络 proposal
3. 第二阶段 : 根据标签(groundtruth_boxlist) 训练图片分类和定位网络

测试

1. 预处理
2. 第一阶段 : RPN 网络生成 proposal
3. 第二阶段 : 根据生产的 proposal_boxes 预测图片分类和物体定位


### 读数据

每次从源数据中批量读 batch_size 的源数据，包括图片以及标签

输入图片: input_image [batch_size, height, width]

标签 : 数量与一次读取的图片数量一致

groundtruth_boxes_list :  每个元素的维度为 [num_boxes, 4] 其中 4 为 [ymin, xmin, ymax, xmax]，而且都小于 [0,1]
groundtruth_classes_list : 每个元素的维度为 [num_boxes, num_classes]
groundtruth_masks_list : 每个元素的维度为 [num_boxes, height_in, width_in]
groundtruth_keypoints_list : 每个元素的维度为 [num_boxes, num_keypoints, 2]

对于  groundtruth 在实际用的时候，需要做预处理，参见 \_format_groundtruth_data


### 训练

预测

box_coding : [N, 4] 其中 4 依次为 [x_center, y_center, w, h]
anchor : [ymin, xmin, ymax, xmax]

groundtruth_boxes_list :  每个元素的维度为 [num_boxes, 4] 其中 4 为 [ymin, xmin, ymax, xmax]，而且都小于 [0,1]
groundtruth_classes_list : 每个元素的维度为 [num_boxes, num_classes]
groundtruth_masks_list : 每个元素的维度为 [num_boxes, height_in, width_in]
groundtruth_keypoints_list : 每个元素的维度为 [num_boxes, num_keypoints, 2]

注：实际在用之前都需要做处理的，参考 \_format_groundtruth_data

CNN 特征提取网络：FasterRCNN 特有的术语

#### 预处理

1. BILINEAR 进行保持比例的 resize, 一般最长 1200，最短 600
2. 不同的 CNN 特征提取网络需要不同的预处理. 如 inception_v2 如 resnet

经过预处理之后，图片变为固定大小，如 [batch_size, 224, 224, 3]

#### 第一阶段

3. 将 preprocessed_inputs  经过 CNN 提取网络得到特征图，支持 inception_v2, resnet_v1, nasnet, inception_resnet_v2  得到 14x14x256 的图片
4. 根据特征提取图大小生成 anchor, 生成算法参照 anchor.md
5. 经过一个 3x3 的 conv2d 卷积, 数量为 512， 得到 rpn_box_predictor_features，目的是得到固定数量的维度

注: 生成 anchors 的个数为 14x14x9，其中 9 为 scale 和 aspect_ratios 不同组合，在
paper 中为 scales=(0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0) 两两共 9 种组合

box 位置预测

6. rpn_box_predictor_features 经过 1 x 1 的卷积，数量为 9 * 4
7. 将的输出 box_coding 维度修改为 (batch_size, height*width*9, 1, 4)

box 分类预测

8. 经过  1 x 1  的卷积，数量为 9 * 2 得到 objectness_predictions_with_background
9. 将输出 objectness_predictions_with_background 维度修改为 (batch_size, height*width*9, 2) 其中最后一维的第一个列为背景，第二维为非背景

如果当前是训练阶段

11. 将 anchors 中四个角都在 input_image 内的 anchors 保留，其余都删除，同时将被删除元素在 box_coding 和 objectness_predictions_with_background  中对应的元素也删除

如果不是训练阶段

11. 将 anchors 中每个 anchor 修改为与 input_image 的交集，去掉与输入图片没有交集的 anchor


如果只运行第一阶段，求 Loss

1. 将 groundtruth_boxeslist 每个 box 转为绝对坐标，即 横轴乘以 image_shape.width, 纵轴乘以 image_shape.height
2. 将 groundtruth_classes_list 每个 box 第一列前增加一列，为背景列。 维度为 [num_boxes, 1 + num_classes ]
3. 用 NEAREST_NEIGHBOR 算法 将 groundtruth_masks_list 每个元素 resize 为 input_image 的尺寸，并对齐
4. 计算 anchors 与 groundtruth_boxes 的 IoU 矩阵 match_quality_matrix，其中行为 groundtruth_boxes,  列为 anchors
5. 计算 match_quality_matrix 每列的最大值在行的索引，保存在 match，并且将 IoU 与阈值比较，小于 0.3 设置为 -1, 0.3 到 0.7  之间设置为 -2
3. 从 anchors 和 groundtruth_boxes 找到大于阈值的 groundtruth_boxe 和 anchor 并编码得到 reg_box，行长度不够部分补 0
4. 从 groundtruth_labels 找到大于阈值的 groundtruth_labels 的 reg_cls,  行长度不够的部分补 0
5. 初始化回归和分类的权重 cls_weights, reg_weights
6. 按照 BalancedPositiveNegativeSampler 方法，从 cls_targets 中采样 256 个样本，正负比例为 0.5
7. 计算分类和位置的 loss 之后，归一化返回

这里需要注意 loss 计算的 predictor 和 target 的数量问题:

对于 predictor 生成了 heigh*width*9 元素
对于 target 生成了 heigh*width*9 个，anchors 和 groundtruth_boxlist 计算 IoU 交集，实际上真正目标标签的可能很少， 为了解决这个问题，如果元素不够，就用哪个 0 来填充，确保 target 也有 heigh*width*9 个元素
 在实际真正计算的时候，以 target 的数量为主


#### 第二阶段

12. 对 anchors 和 box_coding 解码，得到候选框 proposal_boxes  维度为 [batch_size, num_anchors, 4]  TODO 编码和解码意义，即怎么做
13. objectness_predictions_with_background 通过 softmax，取出没有背景的部分
14. 将 proposal_boxes 按照每个类别计算 nms, 之后排序，取前面 self._first_stage_max_proposals 个 box， 返回剩余的候选框，候选框的分数，
以及候选框的数量

如果是训练阶段

15. 停止梯度计算
12. 对 groundtruth 做处理，
12.1 将 groundtruth["boxes"] 的 x 部分乘以 width, y 部分乘以 heigh
12.2 将 groundtruth["classes"] 列索引 0 增加一列 0 表示不是背景
12.3 将 groundtruth["masks"] TODO
13. 从 proposal_boxes 取出与 groundtruth_boxlist 最匹配的 self._second_stage_batch_size 个元素，
其中取出的每个元素满足如下条件
13.1 与某个 groundtruth_boxlist IoU 比其他的所有 groundtruth_boxlist 都大，而且大于阈值 0.7
13.2 采样 proposal_boxes 的顺序是乱序的
13.3 其中正样本比例为 second_stage_balance_fraction
14. 对采样得到的 proposal_boxes 与 input_image 做归一化，得到  proposal_boxes, proposal_scores, num_proposals

注： groundtruth 的初始化是在 provide_groundtruth 实现的


13. 计算 loss


resized_inputs = self._image_resizer_fn(inputs)
preprocessed_inputs = self._feature_extractor.preprocess(resized_inputs)
self.predict(preprocessed_inputs)
    rpn_features_to_crop = self._feature_extractor.extract_proposal_features(preprocessed_inputs)
        feature_map_shape = tf.shape(rpn_features_to_crop)
        anchors = self._first_stage_anchor_generator.generate([(feature_map_shape[1], feature_map_shape[2])])
        slim.cov2d
    self._predict_rpn_proposals(rpn_box_predictor_features)
        self._first_stage_box_predictor.predict
    if is_training self._remove_invalid_anchors_and_predictions
    else box_list_ops.clip_to_window
    self._predict_second_stage
        self._postprocess_rpn
            self._batch_decode_boxes


self._format_groundtruth_data : 计算 loss



### 测试









predict(self, preprocessed_inputs)

preprocessed_inputs : 预处理之后的图片, [batch, height, width, channels]

1. 提取 RPN 特征
2. 经过 self._first_stage_box_predictor_kernel_size 的卷积核，数量为 self._first_stage_box_predictor_depth






def \_batch_decode_boxes(self, box_encodings, anchor_boxes)

参数
* box_encoding : [batch_size, num_anchors, num_classes, self._box_coder.code_size]
* anchor_boxes : [batch_size, num_anchors, 4]
返回
* decoded_boxes : [batch_size, num_anchors, num_classes, 4]

1. 将 box_encoding 维度变为 [batch_size * num_anchors * num_classes, self._box_coder.code_size]
2. 将 anchors 维度变为 [batch_size * num_anchors * num_classes, 4]
3. 将变化后的 box_encoding 和 anchors 解码，之后维度调整为 [combined_shape[0], combined_shape[1], num_classes, 4]

def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   change_coordinate_frame=False,
                                   masks=None,
                                   additional_fields=None,
                                   scope=None):

boxes : [k, q, 4]
* k 为 box 数量;
* 4 保存的是 [ymin, xmin, ymax, xmax] 这个四维向量;
* q 如果为 1，即所有分类都用同一 [k, 4] 的 boxlist，如果不为 1，表示每行一个分类对应一个 [k, 4] 的 boxlist
这里关键是看算法，有的算法会按照分类生成 box, 有的算法会将所有分类用同一 boxlist，比如  faster rcnn q = 1

scores : [k, num_classes, 1]
* k 为 box 数量
* num_classes 表示分类的分数, 一列，表示 k 个 box 在该分类中的分数。

score_thresh : 每个分类中 score 小于该值的 box 会从对应分类的  boxlist 中删除
iou_thresh : nms 的 IoU 阈值
max_size_per_class :  最后 boxlist 每个分类最多的 box 数量
max_total_size : 最后 boxlist 中的 box 数量
clip_window : [ymin, xmin, ymax, xmax]，一般为输入图像大小
change_coordinate_frame : 是否将 boxlist 中的 box 修改相对 clip_window 的坐标
masks : [k, q, mask_height, mask_width]
additional_fields : 增加到 boxlist 中的其他 field


将 boxes 和 scores 按照类别得到对应分类的 box_list，对每类 box_list
1. 去掉 scores 小于 score_thresh 的元素
2. 将每个 box_list 中的 box 和 clip_window 计算交集，裁剪为其交集
3. 将剩余 box 计算 nms，最多选 max_size_per_class
4. 对剩余的 box，增加 classes 属性，设置其所属分类
将各个分类的 box_list 合并，按照 score 排序，获取前 max_total_size 返回之

返回的 boxlist 包含 "classes", "scores", "masks" 字段，可以根据需要获取

详细版本
1. 将 boxes 按照每个分类分开， 每个分类一个 [k,4] 的 box 矩阵，总共得到 q 个
2. 如果 mask 不为空，将 mask 按照每个分类分开， 得到 q 个 [k ,mask_height, mask_width] 的矩阵
3. 定义 selected_boxes_list 保存最后选中的每个分类的 box 列表
4. 遍历每个分类的 box 矩阵 per_class_boxes
4.1 从 scores 取对应分类的 score 加入 per_class_boxes["scores"]
4.2 从 masks 取对应分类的 mask 加入 per_class_boxes["masks"]
4.3 遍历 additional_fields 加入 per_class_boxes
4.4 将 score 小于 score_thresh 的 box  从 per_class_boxes 去掉
4.5 如果 clip_window 不为空，将 per_class_boxes 的每个 box 设置为与 clip_window 的交集
4.6 如果 change_coordinate_frame 为 True, 将 per_class_boxes 变为与 clip_window 的相对坐标，并归一化
4.7 对 per_class_boxes 计算 nms，最多保留 per_class_boxes 个 box
4.8 创建 [k, 1]  的矩阵，其中列保存的是 scores 中的列索引， 并将矩阵加入 per_class_boxes["classes"]
4.9 将 per_class_boxes 加入 selected_boxes_list
5. 对 selected_boxes_list  每个分类的 boxlist  连接起来，并根据 score 进行排序，取前 max_total_size
6. 返回 selected_boxes_list


def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         change_coordinate_frame=False,
                                         num_valid_boxes=None,
                                         masks=None,
                                         additional_fields=None,
                                         scope=None,
                                         parallel_iterations=32)

将 boxes 和 scores 按照类别得到对应分类的 box_list，对每类 box_list
1. 去掉 scores 小于 score_thresh 的元素
2. 将每个 box_list 中的 box 和 clip_window 计算交集，裁剪为其交集
3. 将剩余 box 计算 nms，最多选 max_size_per_class
4. 对剩余的 box，增加 classes 属性，设置其所属分类
将各个分类的 box_list 合并，按照 score 排序，获取前 max_total_size 返回之，如果 box_list 不够  max_total_size，补零。


def \_sample_box_classifier_minibatch(self,
                                     proposal_boxlist,
                                     groundtruth_boxlist,
                                     groundtruth_classes_with_background)

1. 计算 proposal_boxlist 与 groundtruth_boxlist 的 IoU 矩阵 match_quality_matrix，其中横轴为 groundtruth_boxlist, 纵轴为 proposal_boxes
2. 计算 match_quality_matrix 每列的最大值，并且将 IoU 与阈值比较，小于阈值的设置为 -1 或 -2
3. 从 anchors 和  groundtruth_boxes 找到对应的 大于阈值的 groundtruth_boxe 和 anchor 并编码得到 reg_box，不匹配的部分补 0
4. 从 groundtruth_labels 找到大于阈值的 groundtruth_labels 的 reg_cls, 不匹配的部分补 0
5. 初始化回归和分类的权重 cls_weights, reg_weights，得到 reg_box, reg_weights, reg_cls, cls_weights
6. 对分类标签进行 BalancedPositiveNegativeSampler 采样 TODO
7. 从 proposal_boxlist 中取出样本

def \_unpad_proposals_and_sample_box_classifier_batch(
    self,
    proposal_boxes,
    proposal_scores,
    num_proposals,
    groundtruth_boxlists,
    groundtruth_classes_with_background_list)

参数
* proposal_boxes [batch_size, num_proposals, 4]
* proposal_scores [batch_size, num_proposals]
* num_proposals : [batch_size] 每张图片的 proposal_box 个数
* groundtruth_boxlists :  每张图片的 groundtruth 的绝对坐标
* groundtruth_classes_with_background_list : [num_boxes, num_classes + 1]
其中  batch_size 表示图片数量，num_proposals 每张图片的 proposal_boxes 个数

返回
* proposal_boxes
* proposal_scores
* num_proposals

概述

从 proposal_boxes 取出与 groundtruth_boxlist IoU 最匹配的
self._second_stage_batch_size 个元素，其中取出的每个元素
满足如下条件

1. 与某个 groundtruth_boxlist IoU 比其他的所有 groundtruth_boxlist 都大，而且大于阈值 0.7
2. 采样 proposal_boxes 的顺序是乱序的
3. 其中正样本比例为 second_stage_balance_fraction

详细实现

依次遍历 proposal_boxes, proposal_scores 的每个元素
1. 从 proposal_boxes[i], proposal_scores[i] 中取前 num_proposals[i] 个元素构成 single_image_boxlist
2. 计算 single_image_boxlist 与 groundtruth_boxlist[i] 的 IoU 矩阵 match_quality_matrix
3. 计算 match_quality_matrix 每列的最大值，并且将 IoU 与阈值比较，小于阈值的设置为 -1 或 -2
4. 计算 single_image_boxlist 与 groundtruth_boxes, groundtruth_labels  编码，并进行处理，将 single_image_boxlist 与其匹配的放在前，不匹配的放在后面
5. 得到回归，分类矩阵，初始化回归和分类的权重
6. 采样
TODO
将个图片的采样组合起来


def \_postprocess_rpn(self, rpn_box_encodings_batch,
                     rpn_objectness_predictions_with_background_batch,
                     anchors, image_shape)

参数
* rpn_box_encodings_batch : [batch_size, num_anchors, self._box_coder.code_size]
* rpn_objectness_predictions_with_background_batch : [batch_size, num_anchors, 2]
* anchors : [num_anchors, 4]
* image_shape :

返回
* proposal_boxes : [batch_size, max_num_proposals, 4]
* proposal_scores : [batch_size, max_num_proposals]
* num_proposals : [batch]

1. 将 anchors  复制 batch_size 份得到 batch_anchor_boxes
2. 将 batch_anchor_boxes 与 rpn_box_encodings_batch 解码得到 proposal_boxes
3. 将 rpn_objectness_predictions_with_background_batch 通过  softmax  并去掉背景得到 rpn_objectness_softmax_without_background
4. 将 proposal_boxes 和 rpn_objectness_softmax_without_background 按照类别得到对应分类的 box_list，对每类 box_list
4.1. 去掉 scores 小于 self._first_stage_nms_score_threshold 的元素
4.2. 将每个 box_list 中的 box 和 input_image 计算交集，裁剪为其交集
4.3. 将剩余 box 计算 nms，iou 阈值为 self._first_stage_nms_iou_threshold 最多选 self._first_stage_max_proposals
4.4. 对剩余的 box，增加 classes 属性，设置其所属分类
5. 将各个分类的 box_list 合并，按照 score 排序，获取前 self._first_stage_max_proposals  为新的 proposal，如果 box_list 不够 max_total_size，补零。
6. 如果是训练阶段
6.1 停止梯度计算
6.2 从 proposal_boxes 取出与 groundtruth_boxlists IoU 最匹配的 self._second_stage_batch_size 个元素
7.  将采样 proposal_boxes 用输入图片进行归一化

def \_predict_second_stage(self, rpn_box_encodings,
                          rpn_objectness_predictions_with_background,
                          rpn_features_to_crop,
                          anchors,
                          image_shape)

参数
* rpn_box_encodings : [batch_size, num_valid_anchors, self._box_coder.code_size]
* rpn_objectness_predictions_with_background : [batch_size, num_valid_anchors, 2]
* rpn_features_to_crop : feature map
* anchors : [num_anchors, self._box_coder.code_size] 输入图片的 anchor
* image_shape : [batch_size, height, width] 输入图片的尺寸

返回 prediction_dict 包含
* refined_box_encodings : [total_num_proposals, num_classes, 4] total_num_proposals=batch_size*self._max_num_proposals
* class_predictions_with_background : [total_num_proposals, num_classes + 1]
* num_proposals : [batch_size]
* proposal_boxes : [batch_size, self.max_num_proposals, 4]
* proposal_boxes_normalized : [batch_size, self.max_num_proposals, 4]
* box_classifier_features : 
* mask_predictions : [total_num_padded_proposals, num_classes, mask_height, mask_width]

1. 参考 \_postprocess_rpn
2. 将 proposal_boxes_normalized 的维度由 [batch_size, num_proposals, 4] 变为 [batch_size * num_proposals, 4]
3. 用 tf.image.crop_and_resize 对 features_to_crop TODO
4. 通过大小为 self._maxpool_kernel_size ，步长为 self._maxpool_stride 的 pool 卷积
5. 通过 inception_v2/resnet_v1  分类提取网络，得到 box_classifier_features
6. 对 box_classifier_features 进行 mask 预测
6.1 将 image_features 展开为一维
6.2 通过 dropout 层，参数为  self._dropout_keep_prob

box 位置预测

6.3 经过全连接层，数量 self._num_classes * self._box_code_size
6.4 维度变为 [-1, 1, self._num_classes, self._box_code_size]

box 分类预测

6.5 经过全连接层，数量 self._num_classes + 1
6.6 [-1, 1, self._num_classes + 1]

mask 预测

6.7. 对图片进行二分插值
6.8. 经过 2x2 的卷积，数量 self._mask_prediction_conv_depth
6.9. 进行维度转换

得到 {'box_coding': box_coding,
          'class_predictions_with_background': class_predictions_with_background,
          'mask_predictions': instance_masks}

构造 prediction_dict 并返回



def \_compute_second_stage_input_feature_maps(self, features_to_crop,
                                             proposal_boxes_normalized)

1. 将 proposal_boxes_normalized 的维度由 [batch_size, num_proposals, 4] 变为 [batch_size * num_proposals, 4]
2. 用 tf.image.crop_and_resize 对 features_to_crop TODO
3. 通过大小为 self._maxpool_kernel_size ，步长为 self._maxpool_stride 的 pool 卷积


def predict(self, preprocessed_inputs)

1. 将 preprocessed_inputs  经过 CNN 提取网络得到特征图，支持 inception_v2, resnet_v1, nasnet, inception_resnet_v2  得到 14x14x256 的图片
2. 根据特征提取图大小生成 anchor, 生成算法参照 anchor.md
3. 经过一个 self._first_stage_box_predictor_kernel_size 的 conv2d 卷积, 数量为 self._first_stage_box_predictor_depth，目的是得到固定数量的维度
box 位置预测
4. rpn_box_predictor_features 经过 1 x 1 的卷积，数量为 num_predictions_per_location * 4
5. 将的输出 box_coding 维度修改为 (batch_size, height * width * num_predictions_per_location, 4)
box 分类预测
6. 经过  1 x 1  的卷积，数量为 num_predictions_per_location * 2
7. 将输出 objectness_predictions_with_background 维度修改为 (batch_size, height*width*num_predictions_per_location, 2) 其中最后一维的第一个列为背景，第二维为非背景
8. 将 anchors 中四个角都在 clip_window 内的 anchors 保留，其余都删除，并记录保留 anchors 的索引
9. 对于 box_encoding 和 objectness_predictions_with_background 中，如果对应的 anchor 之前被删了，
那么也删除该元素，返回剩余的 box_encoding, objectness_predictions_with_background

注: 生成 anchors 的个数为 14x14x9，其中 9 为 scale 和 aspect_ratios 不同组合，在
paper 中为 scales=(0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0) 两两共 9 种组合



def \_extract_rpn_feature_maps(self, preprocessed_inputs)

参数
* provide_groundtruth : 输入图片

返回
* rpn_box_predictor_features : rpn_features_to_crop 经过 conv2d 的卷积之后的特征图, 主要用于生成 proposal
* rpn_features_to_crop : 只经过 CNN 提取网络得到特征图，在第二阶段基于此进行图片预测和物体定位
* anchors : anchor 列表
* image_shape : 图片维度

1. 将 preprocessed_inputs  经过 CNN 提取网络得到特征图，支持 inception_v2, resnet_v1, nasnet, inception_resnet_v2  得到 14x14x256 的图片
2. 根据特征提取图大小生成 anchor, 生成算法参照 anchor.md
3. 经过一个 self._first_stage_box_predictor_kernel_size 的 conv2d 卷积, 数量为 self._first_stage_box_predictor_depth，目的是得到固定数量的维度

注: 生成 anchors 的个数为 14x14x9，其中 9 为 scale 和 aspect_ratios 不同组合，在
paper 中为 scales=(0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0) 两两共 9 种组合


def \_predict_rpn_proposals(self, rpn_box_predictor_features)

参数
* rpn_box_predictor_features : [batch, height, width, depth] CNN 网络提取层经过 conv2d 的卷积之后的特征图, 主要用于生成 proposal

返回
* box_encodings : [batch_size, num_anchors, self._box_coder.code_size] proposal 位置预测结果
* objectness_predictions_with_background : [batch_size, num_anchors, 2] proposal 分类预测结果

box 位置预测

1. rpn_box_predictor_features 经过 1 x 1 的卷积，数量为 num_predictions_per_location * 4
2. 将的输出 box_coding 维度修改为 （batch_size, height*width*num_predictions_per_location, 4)

box 分类预测

3. 经过  1 x 1  的卷积，数量为 num_predictions_per_location * 2
4. 将输出 objectness_predictions_with_background 维度修改为 (batch_size, height*width*num_predictions_per_location, 2) 其中最后一维的第一个列为背景，第二维为非背景

def \_remove_invalid_anchors_and_predictions(self, box_encodings,
    objectness_predictions_with_background, anchors_boxlist, clip_window):

参数
* box_encodings : [batch_size, num_anchors, self._box_coder.code_size] proposal 位置预测结果
* objectness_predictions_with_background : [batch_size, num_anchors, 2] proposal 分类预测结果
* anchors_boxlist : anchor 列表，绝对坐标，即原点为中心
* clip_window : [ymin, xmin, ymax, xmax] 输入图片大小

返回
* box_encodings : [batch_size, num_valid_anchors, self._box_coder.code_size] proposal 位置预测结果
* objectness_predictions_with_background : [batch_size, num_valid_anchors, 2] proposal 分类预测结果
* anchors :  有效的 anchor，数量 bnum_valid_anchors

1. 将 anchors 中四个角都在 clip_window 内的 anchors 保留，其余都删除，并记录保留 anchors 的索引
2. 对于 box_encoding 和 objectness_predictions_with_background 中，如果对应的 anchor 之前被删了，
那么也删除该元素，返回剩余的 box_encoding, objectness_predictions_with_background



def \_postprocess_box_classifier(self,
                                 refined_box_encodings,
                                 class_predictions_with_background,
                                 proposal_boxes,
                                 num_proposals,
                                 image_shape,
                                 mask_predictions=None):

参数
* refined_box_encodings : [total_num_proposals, num_classes, box_coder.code_size] 每个 proposal 在每个类别的坐标
* class_predictions_with_background : [total_num_padded_proposals, num_classes + 1]
* proposal_boxes : [batch_size, self.max_num_proposals, 4]
* num_proposals : [batch]
* image_shape :
* mask_predictions : [total_num_padded_proposals, num_classes, mask_height, mask_width]

输出
* detection_boxes: [batch, max_detection, 4]
* detection_scores: [batch, max_detections]
* detection_classes: [batch, max_detections]
* num_detections: [batch]
* detection_masks: [batch, max_detections, mask_height, mask_width]



def \_format_groundtruth_data(self, image_shape)

1. 将 groundtruth_boxeslist 每个 box 转为绝对坐标，即 横轴乘以 image_shape.width, 纵轴乘以 image_shape.height
2. 将 groundtruth_classes_list 每个 box 第一列前增加一列，为背景列。 维度为 [num_boxes, 1 + num_classes ]
3. 用 NEAREST_NEIGHBOR 算法 将 groundtruth_masks_list 每个元素 resize 为 image_shape 的尺寸，并对齐

输入
* groundtruth_boxes_list :  每个元素的维度为 [num_boxes, 4] 其中 4 为 [ymin, xmin, ymax, xmax]，而且都小于 [0,1]
* groundtruth_classes_list : 每个元素的维度为 [num_boxes, num_classes]
* groundtruth_masks_list : 每个元素的维度为 [num_boxes, height_in, width_in]
* groundtruth_keypoints_list : 每个元素的维度为 [num_boxes, num_keypoints, 2]

输出
* groundtruth_boxes_list :  每个元素的维度为 [num_boxes, 4] 其中 4 为 [ymin*image_shape.height, xmin*image_shape.width, ymax*image_shape.height, xmax*image_shape.width]
* groundtruth_classes_list : 每个元素的维度为 [num_boxes, 1 + num_classes]
* groundtruth_masks_list : 每个元素的维度为 [num_boxes, image_shape.height, image_shape.width]
* groundtruth_keypoints_list : 每个元素的维度为 [num_boxes, num_keypoints, 2]

def \_loss_rpn(self,
           rpn_box_encodings,
           rpn_objectness_predictions_with_background,
           anchors,
           groundtruth_boxlists,
           groundtruth_classes_with_background_list)

参数
* rpn_box_encodings : [batch_size, num_anchors, 4]
* rpn_objectness_predictions_with_background : [batch_size, num_anchors, 4]
* anchors : 数量为 batch_size 个，每个元素为 [num_anchors, 4]
* groundtruth_boxlists : 数量为 batch_size 个 每个元素为 [num_boxes, 4]
* groundtruth_classes_with_background_list : 数量为 batch_size 个 每个元素为 [num_boxes, 1 + num_classes]

1. 计算 anchors 与 groundtruth_boxes 的 IoU 矩阵 match_quality_matrix，其中行为 groundtruth_boxes,  列为 anchors
2. 计算 match_quality_matrix 每列的最大值在行的索引，保存在 match，并且将 IoU 与阈值比较，小于不匹配阈值的设置为 -1 或 -2
3. 从 anchors 和 groundtruth_boxes 找到大于阈值的 groundtruth_boxe 和 anchor 并编码得到 reg_box，行长度不够部分补 0
4. 从 groundtruth_labels 找到大于阈值的 groundtruth_labels 的 reg_cls,  行长度不够的部分补 0
5. 初始化回归和分类的权重 cls_weights, reg_weights
6. 按照 BalancedPositiveNegativeSampler 方法，从 cls_targets 中采样，正负比例为 first_stage_positive_balance_fraction
7. 计算分类和位置的 loss 之后，归一化返回


def \_padded_batched_proposals_indicator(self, num_proposals, max_num_proposals)

参数
* num_proposals : [batch_size]  每个元素记录每张图片的 proposal_boxes 的个数
* max_num_proposals : 所有图片，最大的 proposal_boxes 的个数

返回
* [batch_size, max_num_proposals] 矩阵，参考下面例子

1. 生成 [batch_size, max_num_proposals] 的矩阵 m1，其中每行的每个元素都是该图片当前的 proposal_boxes 的数量
2. 生成 [batch_size, max_num_proposals] 的矩阵 m2，其中每行都为 [0, 1, ...  max_num_proposals]
3. 计算 m1 中大于 m2 中对应索引的索引

```python
>>> x = sess.run(tf.tile(tf.expand_dims(tf.range(6), 0), [4, 1]))
>>> x
array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]], dtype=int32)

>>> y = sess.run(tf.tile(tf.expand_dims([3, 4, 1, 5], 1), [1, 6]))
>>> y
array([[3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4],
       [1, 1, 1, 1, 1, 1],
       [5, 5, 5, 5, 5, 5]], dtype=int32)

>>> sess.run(tf.greater(y, x))
array([[ True,  True,  True, False, False, False],
       [ True,  True,  True,  True, False, False],
       [ True, False, False, False, False, False],
       [ True,  True,  True,  True,  True, False]], dtype=bool)
```

def \_loss_box_classifier(self,
                   refined_box_encodings,
                   class_predictions_with_background,
                   proposal_boxes,
                   num_proposals,
                   groundtruth_boxlists,
                   groundtruth_classes_with_background_list,
                   image_shape,
                   prediction_masks=None,
                   groundtruth_masks_list=None)

refined_box_encodings : [total_num_proposals, num_classes, box_coder.code_size] 每个 proposal 在每个类别的坐标
class_predictions_with_background : [total_num_proposals, num_classes + 1]
proposal_boxes :
num_proposals :
groundtruth_boxlists:
groundtruth_classes_with_background_list:
image_shape:
prediction_masks:
groundtruth_masks_list:

proposal_boxes 与 groundtruth_boxeslist, groundtruth_classes_with_background 生成预测分类和位置标签
refined_box_encodings 与 batch_reg_targets 计算定位 loss
class_predictions_with_background 与 batch_cls_targets_with_background 计算分类 loss

1. 根据 proposal_boxes 和 groundtruth_boxlist, groundtruth_classes_with_background_list 计算 batch_cls_targets_with_background, batch_cls_weights, batch_reg_targets, batch_reg_weights
2. 对 batch_cls_targets_with_background 计算每行的最大值的索引，生成索引数组
3. 根据索引数组生成 one_hot 矩阵，其中为 1 的列，表明其所属分类 [batch_size * self.max_num_proposals, num_classes + 1]
4. refined_box_encodings 增加背景行得到 refined_box_encodings_with_background
5. 从 refined_box_encodings_with_background 找到每个 proposal_boxes 所属分类的坐标，至此就得到了每个 proposal 预测的分类和坐标
6. refined_box_encodings 与 batch_reg_targets 计算定位 loss
7. class_predictions_with_background 与 batch_cls_targets_with_background 计算分类 loss
8. 对分类和位置 loss 计算只取每张图片 num_proposal 个数的元素
9. 对剩余的位置和分类 loss 采用 hard miming
10. 分类和定位  loss 分别乘以自己的权重矩阵得到分类和权重 loss
11 如果 prediction_masks 和 groundtruth_masks_list 都不为空，

def \_unpad_proposals_and_apply_hard_mining(self,
                                           proposal_boxlists,
                                           second_stage_loc_losses,
                                           second_stage_cls_losses,
                                           num_proposals)


