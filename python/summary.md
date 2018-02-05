

1. 默认所有的元素都加入了变量集合 ops.GraphKeys.SUMMARIES
2. 可以通过  op.get_collection(ops.GraphKeys.SUMMARIES) 获取所有变量
3. 所有的实现都是基于 Cpp, 具体实现参考每个方法的提示
4. 如果你自定义了 collections 参数，那么在 merge_all 的时候，就不能使用默认参数

## SummaryWriter

初始化创建一个队列，创建一个 EventsWriter，开启一个 \_EventLoggerThread 线程，
从队列中取数据，用 EventsWriter 写事件到 ${logdir}/events，每隔 flush_secs(默认 120)
秒，将写入文件的内容落盘

写的文件为 
该线程一直在运行，当调用  close 的时候，线程仍然在执行，但是写事件的时候，并
不会将其写入文件，此时写事件是失效的。

注：与 EventsWriter 各有好处，看场景。


def add_summary(self, summary, global_step=None)

将 summary 加入队列，等待被写入文件

def add_session_log(self, session_log, global_step=None)

将 session_log 加入队列，等待被写入文件

def add_event(self, event)

event 加入队列

def add_graph(self, graph, global_step=None, graph_def=None)

将 graph 加入队列，等待被写入文件

def add_run_metadata(self, run_metadata, tag, global_step=None)

将 run_metadata 加入队列，等待被写入文件

def flush(self)

将数据落盘

def close(self)

将数据落盘之后，关闭文件

### _EventLoggerThread

线程，主要工作的就是从队列中取出  event 写入文件(文件名为${logdir}/events)
每隔 self._flush_secs 秒， 刷新 evnt 到磁盘

### SummaryWriterCache

保存 logdir

## FileWriter

继承自 SummaryToEventTransformer

将事件与写事件分开，更加解耦

写事件采用 EventFileWriter

### EventFileWriter

初始化创建一个队列，创建一个 EventsWriter，开启一个 \_EventLoggerThread 线程，
从队列中取数据，用 EventsWriter 写入磁盘，每隔 flush_secs(默认 120) 秒，将
写入文件的内容落盘。当调用  close 的时候，会增加一个空 event 到队列。 这样，
当线程从队列中取到该事件时，就会终止运行。

### FileWriterCache

保存 logdir

## 其他

def summary_iterator(path)

```
  for r in tf_record.tf_record_iterator(path)
    yield event_pb2.Event.FromString(r)
```

例子

```python
for e in summary_iterator(path):
  print(e)
```

def tensor_summary(name, tensor, summary_description=None,
                   collections=None, summary_metadata=None,
                   family=None, display_name=None)

1. 调用 SummaryScalarOp 将 tag = family/name values = tensor 序列化
2. 如果 collections 不为空，将 value 加入 collections，否则加入 ops.GraphKeys.SUMMARIES

返回 gen_logging_ops._tensor_summary_v2(tensor=tensor, tag=tag, name=scope,
        serialized_summary_metadata=serialized_summary_metadata)

参考 core/kernels/summary_op.cc

def image(name, tensor, max_outputs=3, collections=None, family=None)

return  \_op_def_lib.apply_op("ImageSummary", tag=tag, tensor=tensor,
                                max_images=max_images, bad_color=bad_color,
                                name=name)
参考 core/kernels/summary_image_op.cc

def histogram(name, values, collections=None, family=None)

return \_op_def_lib.apply_op("HistogramSummary", tag=tag, values=values,
                                name=name)

参考 core/kernels/summary_op.cc


def audio(name, tensor, sample_rate, max_outputs=3, collections=None, family=None)

return \_op_def_lib.apply_op("AudioSummaryV2", tag=tag, tensor=tensor,
                                sample_rate=sample_rate,
                                max_outputs=max_outputs, name=name)

参考 core/kernels/summary_audio_op.cc

def merge(inputs, collections=None, name=None)

return \_op_def_lib.apply_op("MergeSummary", inputs=inputs, name=name)

参考 core/kernels/summary_op.cc

def merge_all(key=_ops.GraphKeys.SUMMARIES)

获取  key 对应的所有元素 summary_ops，调用 merge(summary_ops)

def get_summary_description(node_def)

获取 node_def.attr['description']




```
\_InitOpDefLibrary.op_list_ascii = """op {
  name: "Assert"
  input_arg {
    name: "condition"
    type: DT_BOOL
  }
  input_arg {
    name: "data"
    type_list_attr: "T"
  }
  attr {
    name: "T"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
  is_stateful: true
}
op {
  name: "AudioSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type: DT_FLOAT
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "sample_rate"
    type: "float"
  }
  attr {
    name: "max_outputs"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
  deprecation {
    version: 15
    explanation: "Use AudioSummaryV2."
  }
}
op {
  name: "AudioSummaryV2"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type: DT_FLOAT
  }
  input_arg {
    name: "sample_rate"
    type: DT_FLOAT
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "max_outputs"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "HistogramSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "ImageSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "max_images"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_FLOAT
        type: DT_HALF
      }
    }
  }
  attr {
    name: "bad_color"
    type: "tensor"
    default_value {
      tensor {
        dtype: DT_UINT8
        tensor_shape {
          dim {
            size: 4
          }
        }
        int_val: 255
        int_val: 0
        int_val: 0
        int_val: 255
      }
    }
  }
}
op {
  name: "MergeSummary"
  input_arg {
    name: "inputs"
    type: DT_STRING
    number_attr: "N"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "Print"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "data"
    type_list_attr: "U"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "U"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "message"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "first_n"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
  is_stateful: true
}
op {
  name: "ScalarSummary"
  input_arg {
    name: "tags"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "TensorSummary"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "description"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "labels"
    type: "list(string)"
    default_value {
      list {
      }
    }
  }
  attr {
    name: "display_name"
    type: "string"
    default_value {
      s: ""
    }
  }
}
"""
\_op_def_lib = \_InitOpDefLibrary()
```
