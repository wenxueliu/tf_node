
## RegionSimilarityCalculator

RegionSimilarityCalculator
    IouSimilarity : 计算两个 boxlist 对应 IoU 的值
    NegSqDistSimilarity : 计算两个 boxlist 对应元素的 box 的平方差，返回 [N, M] 矩阵
    IoaSimilarity : 计算两个 boxlist 对应元素的 ioa ，返回 [N, M] 矩阵

compare(self, boxlist1, boxlist2, scope=None)

boxlist1 : [N]
boxlist2 : [M]

返回 [N, M] 的矩阵，每个元素为对应 box 的相似度

## Matcher

Matcher
    ArgMaxMatcher

相似矩阵根据阈值将其划分为三类，match, unmatch, ignore

### ArgMaxMatcher

self._matched_threshold
self._unmatched_threshold
self._force_match_for_each_row
self._negatives_lower_than_unmatched :  如果为 True, 表示负值越小越不匹配；否则表示负值越小越匹配；

match(self, similarity_matrix)

similarity_matrix : [N, M]
return : [M]

如果 self._matched_threshold 为空，self._force_match_for_each_row 为 False，返回 similarity_matrix 每列的最大值

如果 self._matched_threshold 不为空，self._negatives_lower_than_unmatched 为 True, self._force_match_for_each_row 为 False

返回 similarity_matrix 每列的最大值 matches，
其中 matches 中的元素 x
    x 小于 self._unmatched_threshold 的为 -1
    x 大于 self._unmatched_threshold，小于 self._matched_threshold 的为 -2

如果 self._matched_threshold 不为空，self._negatives_lower_than_unmatched 为 False, self._force_match_for_each_row 为 False

返回 similarity_matrix 每列的最大值 matches，
其中 matches 中的元素 x

    x 小于 self._unmatched_threshold 的为 -2
    x 大于 self._unmatched_threshold，小于 self._matched_threshold 的为 -1

如果 self._matched_threshold 不为空，self._force_match_for_each_row 为True，
TODO

如果 self._matched_threshold 不为空，self._force_match_for_each_row 为True，
TODO

如果 self._matched_threshold 为空，self._force_match_for_each_row 为 True，

TODO

如果 self._matched_threshold 为空，self._force_match_for_each_row 为 False

TODO

``` python
>>> similarity_matrix
array([[ 8,  5,  6,  1],
       [ 8, 13, 13,  3],
       [ 7, 11,  3,  4],
       [ 5, 14, 13, 16],
       [13,  0,  6,  3],
       [ 4, 13,  2,  4]], dtype=int32)
>>> forced_matches_ids = tf.cast(tf.argmax(similarity_matrix, 1), tf.int32)
>>> sess.run(forced_matches_ids)
array([0, 1, 1, 3, 0, 1], dtype=int32)
>>> row_range = tf.range(tf.shape(similarity_matrix)[0])
>>> col_range = tf.range(tf.shape(similarity_matrix)[1])
>>> sess.run(row_range)
array([0, 1, 2, 3, 4, 5], dtype=int32)
>>> sess.run(col_range)
array([0, 1, 2, 3], dtype=int32)
>>> matches = tf.argmax(similarity_matrix, 0)
>>> sess.run(matches)
array([4, 3, 1, 3])
>>> forced_matches_values = tf.cast(row_range, matches.dtype)
>>> keep_matches_ids, _ = tf.setdiff1d(col_range, forced_matches_ids)
>>> sess.run(forced_matches_values)
array([0, 1, 2, 3, 4, 5])
>>> sess.run(keep_matches_ids)
array([2], dtype=int32)
>>> keep_matches_values = tf.gather(matches, keep_matches_ids)
>>> sess.run(keep_matches_values)
array([1])
>>> matches = tf.dynamic_stitch([forced_matches_ids,keep_matches_ids], [forced_matches_values, keep_matches_values])
>>> sess.run(matches)
array([4, 5, 1, 3])
>>> [forced_matches_ids,keep_matches_ids]
[<tf.Tensor 'Cast_33:0' shape=(6,) dtype=int32>, <tf.Tensor 'ListDiff_4:0' shape=(?,) dtype=int32>]
>>> sess.run([forced_matches_ids,keep_matches_ids])
[array([0, 1, 1, 3, 0, 1], dtype=int32), array([2], dtype=int32)]
>>> sess.run([forced_matches_values, keep_matches_values])
[array([0, 1, 2, 3, 4, 5]), array([1])]
```

### match

对 Matcher 的 match() 返回值的封装

self._match_results : [N] 记录的是 anchors 与  groundtruth_boxes 的 IoU 大于阈值，在 groundtruth_boxes 中的索引。-1 表示不匹配，-2 表示忽略，其他表示匹配

matched_column_indices(self) :  返回 self._match_results 中值大于 -1 的元素在 self._match_results 中索引
matched_column_indicator(self) :  返回 self._match_results 中值大于等于 0 元素设置为 True
num_matched_columns(self) :  返回 self._match_results 中值大于 -1 的元素的个数
unmatched_column_indices(self) :  返回 self._match_results 中值等于 -1 的元素在 self._match_results 中索引
unmatched_column_indicator(self) : self._match_results 中值等于 -1 元素设置为 True，并返回
num_unmatched_columns(self) : 返回 self._match_results 中等于 -1 的元素的个数
ignored_column_indices(self) : 返回 self._match_results 中值等于 -2 的元素在 self._match_results 中索引
ignored_column_indicator(self) :  self._match_results 中值等于 -2 元素设置为 True, 并返回
num_ignored_columns(self) : 返回 self._match_results 中等于 -2 的元素的个数
unmatched_or_ignored_column_indices(self) : 返回 self._match_results 中值等于 -2 或 -1 的元素在 self._match_results 中索引
matched_row_indices(self) : 返回 self._match_results 中值大于 -1 的元素

## TargetAssigner

self._similarity_calc : 计算相似的方法，FasterRCNN 为 IoU
self._matcher : 两个 box 匹配方法，FasterRCNN 为 argmax
self._box_coder : 对 box 的编码方法
self._positive_class_weight : FasterRCNN 为 1.0
self._negative_class_weight : FasterRCNN 为 1.0
self._unmatched_cls_target

def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None, \**params)

参数
* anchors : box_list.BoxList   在 faster rcnn 中为 proposal box
* groundtruth_boxes : box_list.BoxList
* groundtruth_labels :  M

返回
cls_targets : [num_anchors, d_1, d_2 ... d_k] groundtruth_labels 与 match 匹配的部分
cls_weights : [num_anchors] groundtruth_labels 与  match 匹配的部分 设置为 1.0，其余为 0
reg_targets : [num_anchors, box_code_dimension] groundtruth_boxes 与 anchors IoU 大于阈值的部分编码之后
reg_weights : [num_anchors] groundtruth_boxes 大于阈值的部分，为 1.0，其余为 0

cls_targets 为每个 anchor 所属分类，d_1 为背景，其他维度为具体分类

1. 计算 anchors 与 groundtruth_boxes 的 IoU 矩阵 match_quality_matrix，其中行为 groundtruth_boxes,  列为 anchors
2. 计算 match_quality_matrix 每列的最大值在每列的索引保存为 match，并且将 IoU 与阈值比较，小于阈值的设置为 -1 或 -2
3. 从 anchors 和  groundtruth_boxes 找到大于阈值的 groundtruth_boxe 和 anchor 并编码得到 reg_box，不够的部分补 0
4. 从 groundtruth_labels 找到大于阈值的 groundtruth_labels 的 reg_cls, 不够的部分补 0
5. 初始化回归和分类的权重 cls_weights, reg_weights，返回

注：
1. 这里 IoU 阈值在 FasterRCNN detection 阶段为 0.5，在  FasterRCNN proposal 阶段为 0.3, 0.7
2. 这里是针对一张图片的

def \_create_regression_targets(self, anchors, groundtruth_boxes, match)

参数
* anchors : [num_anchors]
* groundtruth_boxes: [M, 4]
* match : [num_anchors] anchors 与  groundtruth_boxes 的匹配数组，其中匹配项其值不为 0，不匹配为 -1 或 -2

返回
* reg_targets : [num_anchors, box_code_dimension]

根据 match 从 anchors 和 groundtruth_boxes 中找到匹配的元素，进行编码，返回。为了保持输入维度，后面补零。
1. 从 anchors 中找到与 match 对应的 matched_anchors
2. 从 groundtruth_boxes 中找到与 match 对应的 matched_gt_boxes
3. 计算 matched_gt_boxes 相对于 matched_anchors  的偏移，并归一化
4. 将 3 后面加上 0 矩阵，使其仍然保持输入 match 的元素个数一致，返回

def \_create_classification_targets(self, groundtruth_labels, match)

参数
* groundtruth_labels : [num_gt_boxes, 1 + num_classes]
* match : [num_anchors] anchors 与  groundtruth_boxes 的匹配数组，其中匹配项其值不为 0，不匹配为 -1 或 -2

返回值
* cls_targets : [num_anchors] 总的 labels 数，其中前面是 groundtruth_labels 与 match 匹配的，后面为不匹配的，用 0 补齐，使之行为 num_anchors

1. 从 groundtruth_labels 找到与 match 对应的 matched_cls_targets
2. 将不匹配的设置为 1，放在返回矩阵的后面

def \_create_regression_weights(self, match)

将 match 中元素大于0 的设置为 1.0, 其余设置为 0

def \_create_classification_weights(self,
                                    match,
                                    positive_class_weight=1.0,
                                    negative_class_weight=1.0)

将 match 中匹配的权重设置为  positive_class_weight,  不匹配的设置为
negative_class_weight，返回修改后的矩阵

def create_target_assigner(reference, stage=None,
                           positive_class_weight=1.0,
                           negative_class_weight=1.0,
                           unmatched_cls_target=None)

```python
  if reference == 'Multibox' and stage == 'proposal':
    similarity_calc = sim_calc.NegSqDistSimilarity()
    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    box_coder = mean_stddev_box_coder.MeanStddevBoxCoder()

  elif reference == 'FasterRCNN' and stage == 'proposal':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                           unmatched_threshold=0.3,
                                           force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])
  elif reference == 'FasterRCNN' and stage == 'detection':
    similarity_calc = sim_calc.IouSimilarity()
    # Uses all proposals with IOU < 0.5 as candidate negatives.
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           negatives_lower_than_unmatched=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=[10.0, 10.0, 5.0, 5.0])

  elif reference == 'FastRCNN':
    similarity_calc = sim_calc.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                           unmatched_threshold=0.1,
                                           force_match_for_each_row=False,
                                           negatives_lower_than_unmatched=False)
  box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
```
