


def string_input_producer(string_tensor,
                          num_epochs=None,
                          shuffle=True,
                          seed=None,
                          capacity=32,
                          shared_name=None,
                          name=None,
                          cancel_op=None):

    调用 input_producer

def input_producer(input_tensor,
                   element_shape=None,
                   num_epochs=None,
                   shuffle=True,
                   seed=None,
                   capacity=32,
                   shared_name=None,
                   summary_name=None,
                   name=None,

   1. 创建一个 FIFOQueue (FIFOQueue 具体由  C++ 代码实现)
   2. 将创建的队列加入 ops.GraphKeys.QUEUE_RUNNERS


## Reader

tensorflow/python/ops/io_ops.py

ReaderBase
    WholeFileReader
    TextLineReader
    FixedLengthRecordReader
    TFRecordReader
    LMDBReader
    IdentityReader

* read(self, queue, name=None)
* read_up_to(self, queue, num_records, name=None)
* num_records_produced(self, name=None)
* num_work_units_completed(self, name=None)
* serialize_state(self, name=None)
* restore_state(self, state, name=None)

对 gen_io_ops 的简单封装，具体实现参考 cpp  部分的 io 章节。

创建一个读处理器，读的时候将数据读到队列里面
