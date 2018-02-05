
TODO :  有很多操作还没有完全理解


python/ops/metrics
contrib/

streaming_recall = recall
streaming_precision = precision
streaming_true_positives = true_positives
streaming_true_negatives = true_negatives
streaming_false_positives  = false_positives
streaming_false_negatives = false_negatives
streaming_mean = mean
streaming_mean_tensor = mean_tensor
streaming_accuracy = accuracy
\_streaming_confusion_matrix_at_thresholds
streaming_true_positives_at_thresholds
streaming_false_negatives_at_thresholds
streaming_false_positives_at_thresholds
streaming_true_negatives_at_thresholds
streaming_curve_points TODO
streaming_auc = auc
streaming_specificity_at_sensitivity = specificity_at_sensitivity
streaming_sensitivity_at_specificity = sensitivity_at_specificity
streaming_precision_at_thresholds = metrics.precision_at_thresholds
streaming_recall_at_thresholds = metrics.recall_at_thresholds
streaming_sparse_recall_at_k = metrics.recall_at_k
streaming_sparse_precision_at_k = metrics.sparse_precision_at_k
streaming_sparse_precision_at_top_k = metrics_impl._sparse_precision_at_top_k
sparse_recall_at_top_k = metrics_impl._sparse_recall_at_top_k
streaming_sparse_average_precision_at_k = metrics.sparse_average_precision_at_k
streaming_sparse_average_precision_at_top_k = metrics_impl._streaming_sparse_average_precision_at_top_k
streaming_mean_absolute_error = metrics.mean_absolute_error
streaming_mean_relative_error = metrics.mean_relative_error
streaming_mean_squared_error = metrics.mean_squared_error
streaming_root_mean_squared_error = metrics.root_mean_squared_error
streaming_covariance = TODO
streaming_pearson_correlation = TODO
streaming_mean_cosine_distance = TODO
streaming_percentage_less = metrics.percentage_below
streaming_mean_iou = metrics.mean_iou
streaming_concat =  TODO


def aggregate_metrics(*value_update_tuples)

    value_ops, update_ops = zip(*value_update_tuples)
    return list(value_ops), list(update_ops)

def aggregate_metric_map(names_to_tuples)

    metric_names = names_to_tuples.keys()
    value_ops, update_ops = zip(*names_to_tuples.values())
    return dict(zip(metric_names, value_ops)), dict(zip(metric_names, update_ops))


def \_next_array_size(required_size, growth_factor=1.5)

返回 tf.ceil(growth_factor ** tf.ceil(required_size / growth_factor))





def accuracy(predictions, labels, weights=None, name=None)

如果  weights 为 None，返回 tf.reduce_mean(tf.cast(tf.equals(predictions, labels), tf.float32))
如果  weights 不为 None，返回 a = tf.cast(tf.equals(predictions, labels), tf.float32) b= a*weights  返回 b / (weights * tf.ones_like(b))


def \_safe_div(numerator, denominator, name)

    0 if `denominator` <= 0, else `numerator` / `denominator`

def \_create_local(name, shape, collections=None, validate_shape=True, dtype=dtypes.float32)

    创建 local 变量

def remove_squeezable_dimensions(labels, predictions, expected_rank_diff=0, name=None)

如果  predictions.ndims - labels.ndims = expected_rank_diff + 1, 将  predictions 最后一维去掉
如果  predictions.ndims - labels.ndims = expected_rank_diff - 1, 将  labels 最后一维去掉


def true_positives(labels, predictions, weights=None,
                metrics_collections=None, updates_collections=None, name=None)

计算 labels 和  predictions 都为 True 的数量
1. is_true_positive = tf.logical_and(tf.equals(labels, True), tf.equals(predictions, True))
2. 返回 \_count_condition(is_true_positive, weights, metrics_collections, updates_collections, name)

TODO : 还没有完全理解

def false_positives(labels, predictions, weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):

计算 labels 为 False, predictions 都为 True 的数量
1. is_false_positive = tf.logical_and(tf.equals(labels, False), tf.equals(predictions, True))
2. 返回 \_count_condition(is_false_positive, weights, metrics_collections, updates_collections, name)

TODO : 还没有完全理解


def false_negatives(labels, predictions, weights=None,
                    metrics_collections=None, updates_collections=None,
                    name=None)

计算 labels 为 True, predictions 都为 False 的数量
1. is_false_negatives = tf.logical_and(tf.equals(labels, True), tf.equals(predictions, False))
2. 返回 \_count_condition(is_false_negatives, weights, metrics_collections, updates_collections, name)


def mean(values, weights=None, metrics_collections=None,
         updates_collections=None, name=None)

1. 定义两个变量 total, count
2. update_op = total + tf.reduce_mean(values) / count + tf.size(values)
3. mean_t = total / count
4. 返回  mean_t, update_op


def \_confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=None, includes=None)

根据  includes 的元素计算  tp, tn, fp, fn

其中
tp : labels 和 predictions > thresholds 中都为 True 的数量
fn : labels  为 True, predictions > thresholds 中为 False 的数量
tn : labels 和 predictions > thresholds 中都为 False 的数量
fp : labels  为 False, predictions > thresholds 中为 True 的数量

def auc(labels, predictions, weights=None, num_thresholds=200,
        metrics_collections=None, updates_collections=None,
        curve='ROC', name=None)

TODO

def mean_absolute_error(labels, predictions, weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None)

1. values = tf.abs(labels - predictions)
2. tf.mean(values, weights)



def mean_cosine_distance(labels, predictions, dim, weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None)
TODO
返回 tf.mean(tf.reduce_sum(labels * predictions, dim), weights)

def mean_per_class_accuracy(labels,
                            predictions,
                            num_classes,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None)

计算平均每个分类命中的比例
cm = tf.confusion_matrix(labels, predictions) # 计算命中矩阵
per_row_sum = tf.reduce_sum(cm, 1)   # 每个分类的实例数量
hit = tf.diag_part(cm)               # 正确预测的个数
per_class_ac = hit / per_row_sum     # 各个分类命中的比例
tf.reduce_mean(per_class_ac)         # 平均每个分离命中的比例

def mean_iou(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None)

计算 per-step mean Intersection-Over-Union (mIOU)

cm = tf.confusion_matrix(labels, predictions) # 计算命中矩阵
per_row_sum = tf.reduce_sum(cm, 1)    # 每个分类的实例数量
per_colum_sum = tf.reduce_sum(cm, 0)  # 每个分类的实例数量
hit = tf.diag_part(cm)                # 正确预测的个数
d = per_colum_sum + per_row_sum - hit
tf.reduce_mean(hit / d)

def mean_relative_error(labels, predictions, normalizer, weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None)

TODO

def mean_squared_error(labels, predictions, weights=None,
                       metrics_collections=None,
                       updates_collections=None,
                       name=None)

1. values = tf.square(labels - predictions)
2. tf.mean(values, weights)

def mean_tensor(values, weights=None, metrics_collections=None,
                updates_collections=None, name=None)


从 values 依次取出值，取平均值，其中维度与  values 的维度相同。

返回  values * weights / tf.ones_like(values) if weights not None
返回  values / tf.ones_like(values) if weights is None

需要注意的是，这里  values 需要结合队列才能体现出效果。

def percentage_below(values, threshold, weights=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None)

返回 tf.metrics.mean(tf.less(values, threshold), weights)
返回 tf.metrics.mean(tf.less(values, threshold))

def \_count_condition(values, weights=None, metrics_collections=None, updates_collections=None)

计算 values 中为 True 的元素个数

其中, values 必须为  bool 类型
weights 为 0，1 来控制对应 values 的值保持原来，还是为 0

1. 创建一个变量 count
2. update_op = tf.assign_add(count, tf.reduce_sum(tf.to_float(values)* weights))
3. 将 count 加入  metrics_collections, update_op 加入 updates_collections
4. 返回 count, update_op

def precision(labels, predictions, weights=None,
              metrics_collections=None, updates_collections=None,
              name=None)

tp, tp_u = true_positives(labels, predictions, weights)
fp, fp_u = false_positives(labels, predictions, weights)
p = tf.where(tp.greater(tp+fp, 0), tp/(fp + tp), 0, name)
p_u = tf.where(tp.greater(tp_u+fp_u, 0), tp_u/(fp_u + tp_u), 0, name)
返回 p，p_u

def precision_at_thresholds(labels, predictions, thresholds,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None, name=None)

TODO

def recall(labels, predictions, weights=None,
           metrics_collections=None, updates_collections=None,
           name=None)

tp, tp_u = true_positives(labels, predictions, weights)
fn, fn_u = false_negatives(labels, predictions, weights)
p = tf.where(tp.greater(tp+fn, 0), tp/(fn + tp), 0, name)
p_u = tf.where(tp.greater(tp_u+fn_u, 0), tp_u/(fn_u + tp_u), 0, name)
返回 p，p_u


def \_select_class_id(ids, selected_id)

def \_maybe_select_class_id(labels, predictions_idx, selected_id=None)

def \_sparse_true_positive_at_k(labels,
                               predictions_idx,
                               class_id=None,
                               weights=None,
                               name=None)

def \_streaming_sparse_true_positive_at_k(labels,
                                         predictions_idx,
                                         k=None,
                                         class_id=None,
                                         weights=None)

def \_sparse_false_negative_at_k(labels,
                                predictions_idx,
                                class_id=None,
                                weights=None)

def \_streaming_sparse_false_negative_at_k(labels,
                                          predictions_idx,
                                          k,
                                          class_id=None,
                                          weights=None,
                                          name=None)

def recall_at_k(labels,
                predictions,
                k,
                class_id=None,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None)

def \_sparse_recall_at_top_k(labels,
                            predictions_idx,
                            k=None,
                            class_id=None,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):

def recall_at_thresholds(labels, predictions, thresholds,
                         weights=None, metrics_collections=None,
                         updates_collections=None, name=None)

def root_mean_squared_error(labels, predictions, weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None)
def sensitivity_at_specificity(
    labels, predictions, specificity, weights=None, num_thresholds=200,
    metrics_collections=None, updates_collections=None, name=None)

def \_expand_and_tile(tensor, multiple, dim=0, name=None)

def \_num_relevant(labels, k)

def \_sparse_average_precision_at_top_k(labels, predictions_idx)

def \_streaming_sparse_average_precision_at_top_k(labels,
                                                 predictions_idx,
                                                 weights=None,
                                                 metrics_collections=None,
                                                 updates_collections=None,
                                                 name=None)jkkkkk

def sparse_average_precision_at_k(labels,
                                  predictions,
                                  k,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None)

def \_sparse_false_positive_at_k(labels,
                                predictions_idx,
                                class_id=None,
                                weights=None)

def \_streaming_sparse_false_positive_at_k(labels,
                                          predictions_idx,
                                          k=None,
                                          class_id=None,
                                          weights=None,
                                          name=None)

def \_sparse_precision_at_top_k(labels,
                               predictions_idx,
                               k=None,
                               class_id=None,
                               weights=None,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):

def sparse_precision_at_k(labels,
                          predictions,
                          k,
                          class_id=None,
                          weights=None,
                          metrics_collections=None,
                          updates_collections=None,
                          name=None):

def specificity_at_sensitivity(
    labels, predictions, sensitivity, weights=None, num_thresholds=200,
    metrics_collections=None, updates_collections=None, name=None):
