
将 session_manager, input, model, saver, summary 综合到一起，使得用户只需要实现
model 和  input 接口开始训练，而且通过了灵活自定义方式。

## estimator

### RunConfig

主要包括一些参数配置项，已经对参数项的校验。 可以通过 replace 来设置属性，避免大量垃圾代码和类的膨胀

实现较为简单，分析略去

\_validate_properties 没有校验某些属性

### Estimator

子类需要实现

self._call_input_fn
self._call_model_fn

方法来控制模型

self._config : 配置项，参考 RunConfig
self._model_dir : 保存 checkpoints, event
self._params : 保存超参数
self._device_fn :
self._model_fn : 模型函数

model_fn 函数的约束

1. 参数 features labels, mode, params, config, 其中  features, labels 来自 input_fn, params, config 来自 Estimator  的构造函数
2. 返回类 EstimatorSpec 实例 estimator_spec
3. mode = train 必须包含 train_op, loss
4. mode = eval  必须包含 loss
5. mode = prediction 必须包含 predictions


def \__init__(self, model_fn, model_dir=None, config=None, params=None)

model_fn : 模型函数
model_dir : 保存运行数据的文件夹
参数初始化及校验
params : model_fn 参数
config : model_fn 参数

def train(self, input_fn, hooks=None, steps=None, max_steps=None)

1. 参数校验
2. 检查 max_steps 与 checkpoints 文件中的 global_step. 如果 max_steps 小于 global_step，返回

传递 hook 到训练中有两种方法:
1. 通过参数 hooks
2. 通过 model_fn 函数对象的  training_hook

def \_train_model(self, input_fn, hooks)

1. 调用 input_fn 获取 features, labels
2. 调用 self._model_fn 返回 estimator_spec
3. 增加 hook, 参考下面默认回调
4. 执行，具体参考下面执行分析

创建 training.MonitoredTrainingSession，调用  sess.run 开始训练，返回找到的 op
返回 estimator_spec.loss

#### 默认集成的回调

1. StopAtStepHook : 检查  step 是否达到 max_steps
2. LoggingTensorHook : 每迭代 100 步日志记录 loss, global_step
3. NanTensorHook : 记录 loss 到 summary
4. Saver :  记录 checkpoint 包括 global_step, grap, graph_meta，只在  chief 节点记录
5. StepCounterHook : 记录 step 到 summary
6. SummarySaverHook :  记录 scaffold.summary_op, global_step 到 summary
7. CheckpointSaverHook : 记录 global_step, grap, graph_meta 到 checkpoint_dir；此类 hook 只有一个

其中 StepCounterHook, SummarySaverHook 只在主节点进行

自定义 Saver :  创建 Saver 将其加入 ops.GraphKeys.SAVERS 或 estimator_spec.scaffold.saver

#### 执行分析

如下面伪代码

```python
  [ hook.begin() for hook in hooks ]
  sess = tf.Session()
  [ hook.after_create_session() for hook in hooks]
  while not stop is requested:
    [ hooks.before_run() for hook in hooks ]
    try:
      results = sess.run(merged_fetches, feed_dict=merged_feeds)
    except (errors.OutOfRangeError, StopIteration):
      break
    [ hooks.after_run() for hook in hooks ]
  [ hooks.end() for hook in hooks ]
  sess.close()
```


    features, labels = self._get_features_and_labels_from_input_fn(input_fn, mode)
      self._call_input_fn(input_fn, mode)
        input_fn(kwargs)
    self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN)
      self._model_fn(features, kwargs)
    training.MonitoredTrainingSession()


def \_evaluate_model(self,
                      input_fn,
                      hooks=None,
                      checkpoint_path=None,
                      name='')

1. 调用 input_fn 获取 features, labels
2. 调用 self._model_fn 返回 estimator_spec
3. 设置 estimator_spec.eval_metric_ops
*  loss: estimator_spec.loss (默认)
*  estimator_spec.eval_metric_ops
4. 初始化 hook : hooks + estimator_spec.evaluation_hooks
5. 解析 estimator_spec.eval_metric_ops 找到 updata_op 和 final_op
6. 获取 eval_step( 0 或 ops.GraphKeys.EVAL_STEP)
7. 将 eval_step + 1 加入 eval_ops
8. 将 final_op 加入 hooks
9. 执行 eval_op 操作，feed_dict 喂数据给 eval_ops，具体的执行多少次，需要在 hooks 中增加 evaluation.StopAfterNEvalsHook 来指定
10. 返回 final_op 中的值
11. 将 final_op 的结果转化为 summary_proto 写入 summary


    features, labels = self._get_features_and_labels_from_input_fn(input_fn, mode)
      self._call_input_fn(input_fn, mode)
        input_fn(kwargs)
    self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN)
      self._model_fn(features, kwargs)
    evaluation._evaluate_once

summary 的包括:

estimator_spec.loss

saver 包括


def \_extract_metric_update_ops(eval_dict)

解析字典 eval_dict 每个元素，返回  updata_op, value_op

eval_dict 格式
{
  name : [value_op, update_op]
  name : [value_op, update_op]
  ...
}

其中  updata_op 是 sess.run 执行的部分

def train(self, input_fn, hooks=None, steps=None, max_steps=None)

1. 参数校验
2. 调用  self._train_model


def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)

1. 参数校验
2. 通过 \_StopAfterNEvalsHook 设定执行步骤数
3. 调用  self._evaluate_model

def predict(self,
            input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None)

1. 调用 input_fn 获取 features, labels
2. 调用 self._model_fn 得到 estimator_spec
3. 从 estimator_spec.prediction 得到提取 predict_keys 中的值
4. sess.run 运行 prediction

def export_savedmodel(
    self, export_dir_base, serving_input_receiver_fn,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None)


1. 调用 serving_input_receiver_fn 得到 serving_input_receiver
2. 以  serving_input_receiver_fn  为 input, 调用 self._model_fn 得到 estimator_spec
3. 遍历 estimator_spec.export_outputs，将其转为字典之后返回，其中 value 转为 signature_def
4. 从 checkpoint_path 恢复 graph
5. 构造 SavedModel 保存在  export_dir_base/time.time() 文件夹
6. 如果 assets_extra 不为空， 遍历  assets_extra 将 export_dir_base/assets.extra/key 拷贝到  value



### EstimatorSpec

collections.namedtuple('EstimatorSpec', [
    'predictions', 'loss', 'train_op', 'eval_metric_ops',
    'export_outputs', 'training_chief_hooks', 'training_hooks',
    'scaffold', 'evaluation_hooks'
])):

对 train, evaluation, prediction 的封装

* loss : 必须是一个 scalar 的 Tensor
* train_op :  Operation 或 Tensor
* predictions : Tensor 或  dict( value 为 Tensor，此外 Tensor 必须在 default_graph 中)
* eval_metric_ops : 必须是 dict(格式{ string: (value, update)} )， 其中 value, update 必须为 Operation 或 Tensor， 其中 value 是最后执行， 只在 evaluation时起作用。
* export_outputs : 必须是 dict( 格式 {string: value}，value 必须为 ExportOutput 实例)
* training_chief_hooks : 必须是 session_run_hook.SessionRunHook 实例，只在 mode 为 train 时有用
* training_hooks : 必须是 session_run_hook.SessionRunHook 实例
* evaluation_hooks : 必须是 session_run_hook.SessionRunHook 实例
* scaffold :  必须是 monitored_session.Scaffold 实例

其中

* mode = train 必须包含 train_op, loss, training_chief_hooks 只在此时有用
* mode = eval  必须包含 loss
* mode = prediction 必须包含 predictions

### _FeedingQueueRunner


def \__init__(self, queue=None, enqueue_ops=None, close_op=None,
             cancel_op=None, feed_fns=None,
             queue_closed_exception_types=None)

self._enqueue_ops = enqueue_ops
self._feed_fns = feed_fns


def \_run(self, sess, enqueue_op, feed_fn, coord=None)

循环运行 sess.run(enqueue_ops, feed_dict = feed_fn)，直到遇到异常

def create_threads(self, sess, coord=None, daemon=False, start=False)

创建线程，遍历 self._enqueue_ops, self._feed_fns 每个对创建一个线程，在 sess
下运行


def \_get_integer_indices_for_next_batch(
    batch_indices_start, batch_size, epoch_end, array_length,
    current_epoch, total_epochs)

将  batch_indices_start, batch_indices_start+batch_size 以 array_length
划分， 返回划分后的分组索引，和分组数 + current_epoch

```
s = 1
e = 15
a = 8
batch_indices = [j % a for j in range(s, e)] # [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6]
```


### _ArrayFeedFn

def __init__(self, placeholders, array, batch_size,
        random_start=False, seed=None, num_epochs=None)

从 array 中从 0 或一个随机索引开始，以 batch_size 为一组，
返回取到的组数，每组的数据一样

如果多次调用，下次从上一次组的的最后索引之后开始继续取

比如 [ a,  b,  c,  d,  e,  f,  g,  h,  i,  j, k, l, m, n, o, p]

以 batch_size 为 3

第一次调用，只取 [0, 1, 2]，返回 { placeholders[0]: [0, 1, 2], placeholders[1]: [a, b, d] }
第二次调用，只取 [3, 4, 5]，返回 { placeholders[0]: [4, 5, 6], placeholders[1]: [e, f, g] }
以此类推
每 5 次是一轮，总共取 num_epochs 轮
之后，继续取就会抛异常

\_OrderedDictNumpyFeedFn \_PandasFeedFn \_GeneratorFeedFn 原理与 \_ArrayFeedFn 类似


def \_enqueue_data(data,
                  capacity,
                  shuffle=False,
                  min_after_dequeue=None,
                  num_threads=1,
                  seed=None,
                  name="enqueue_input",
                  enqueue_size=1,
                  num_epochs=None

产生的实际轮数是  num_threads * num_epochs

shuffle  为  True 与  num_threads > 1 一起设置，此时 num_epochs 不要设置

1. 启动 num_threads * enqueue_size 个线程，从 data 中读取 enqueue_size 个元素，启动 enqueue_size 个线程，将读到的数据写入队列，总共执行 num_epochs 轮
2. 统计队列使用情况到  summary


### export

ServingInputReceiver

def \__new__(cls, features, receiver_tensors)

features : 推荐为 dict， 其中 key 为 string，value 为 tensor
receiver_tensors : 推荐为  dict， 其中 key 为 string，value 为 tensor

def build_parsing_serving_input_receiver_fn(feature_spec, default_batch_size=None)

将  feature_spec 转为  ServingInputReceiver

def build_raw_serving_input_receiver_fn(features, default_batch_size=None)

将  features 转为 ServingInputReceiver

def build_all_signature_defs(receiver_tensors, export_outputs)

将  export_outputs 转为 signature_def


```python
    features = {
        "feature0": constant_op.constant([0]),
        u"feature1": constant_op.constant([1]),
        "feature2": sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
    }
    receiver_tensors = {
        "example0": array_ops.placeholder(dtypes.string, name="example0"),
        u"example1": array_ops.placeholder(dtypes.string, name="example1"),
    }
    export.ServingInputReceiver(features, receiver_tensors)


    feature = constant_op.constant(5)
    receiver_tensor = array_ops.placeholder(dtypes.string)
    input_receiver = export.ServingInputReceiver(feature, receiver_tensor)

        receiver_tensor = constant_op.constant(["11"])
    output_1 = constant_op.constant([1.])
    output_2 = constant_op.constant(["2"])
    output_3 = constant_op.constant(["3"])
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            export_output.RegressionOutput(value=output_1),
        "head-2": export_output.ClassificationOutput(classes=output_2),
        "head-3": export_output.PredictOutput(outputs={
            "some_output_3": output_3
        }),
    }

    signature_defs = export.build_all_signature_defs(
        receiver_tensor, export_outputs)
```

### ExportOutput


class ClassificationOutput(ExportOutput)

def \__init__(self, scores=None, classes=None)

1. 校验 scores, classes
2. self._scores = scores , self._classes = classes

def as_signature_def(self, receiver_tensors)

receiver_tensors.values 为 input, self._scores, self._classes 为 output，构造 signature_def

class RegressionOutput(ExportOutput)

def \__init__(self, value)

1. 校验 value
2. self._value = value

def as_signature_def(self, receiver_tensors)

receiver_tensors.values 为  input, self._value 为 output，构造 signature_def

class PredictOutput(ExportOutput)

def \__init__(self, outputs)

1. 校验 outputs
2. self._outputs = outputs

def as_signature_def(self, receiver_tensors)

receiver_tensors.values 为  input, self._outputs 为 output，构造 signature_def


例子
```python
    input_tensors = {
        "input-1":
            array_ops.placeholder(
                dtypes.string, 1, name="input-tensor-1")
    }
    classes = array_ops.placeholder(dtypes.string, 1, name="output-tensor-1")

    export_output = export_output_lib.ClassificationOutput(classes=classes)
    actual_signature_def = export_output.as_signature_def(input_tensors)
```
