

class QueueRunner(object)

def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, queue_closed_exception_types=None,
               queue_runner_def=None, import_scope=None):


两种创建队列的方式
1. 定义 queue, enqueue_ops, cancel_op, queue_closed_exception_types
2. 定义 queue_runner_def

两种只能选其一，如果都定义，会抛异常


def create_threads(self, sess, coord=None, daemon=False, start=False):

每个 enqueue_ops 会创建一个线程去执行, 如果 coord 不为空，会创建一个 coord 线程，支持 daemon 模式


def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS)

 将 qr 加入 collection(即 ops.GraphKeys.QUEUE_RUNNERS)

def start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection=ops.GraphKeys.QUEUE_RUNNERS):

遍历 collection 中的元素，并调用 create_threads 创建并开始线程

ops.register_proto_function(ops.GraphKeys.QUEUE_RUNNERS,
                            proto_type=queue_runner_pb2.QueueRunnerDef,
                            to_proto=QueueRunner.to_proto,
                            from_proto=QueueRunner.from_proto)

将

`register.Registry self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: stack[2]}`

其中 name 为 ops.GraphKeys.QUEUE_RUNNERS, candidate 为 (proto_type, to_proto, from_proto)
