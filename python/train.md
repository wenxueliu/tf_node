

tensorflow/python/traing/input.py
contrib/training/python/training/training.py


def batch(tensors, batch_size, keep_input, num_threads=1, capacity=32,
           enqueue_many=False, shapes=None, dynamic_pad=False,
           allow_smaller_final_batch=False, shared_name=None,
           name=None):

与 shuffle_batch 不一样的地方在于用的是 FIFOQueue(dynamic_pad=False) 或
PaddingFIFOQueue(dynamic_pad = True)

def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                   keep_input, num_threads=1, seed=None, enqueue_many=False,
                   shapes=None, allow_smaller_final_batch=False,
                   shared_name=None, name=None):

capacity : batch_size 的整数倍，如果 allow_smaller_final_batch 为 True, 可以加小于 batch_size 的 tensor
allow_smaller_final_batch : 为 True, 允许最后一组元素个数小于 batch_size
num_threads : 默认值放一个 queue_ops 到 RandomShuffleQueue，num_threads 线程，一次方  thread 个操作
batch_size : 每次从队列中取出的元素个数

1. tensors 中所有元素类型必须相同
2. 创建一个 RandomShuffleQueue 队列，
遍历 tensors 的每一个元素，将 RandomShuffleQueue 的 enqueue 或参数指定的 enqueue_many  加入 ops.GraphKeys.QUEUE_RUNNERS 等待运行
3. 从队列中取出 batch_size 个的元素

TODO : 这里还有表述不太准确的地方

def batch_join(tensors_list, batch_size, keep_input, capacity=32,
             enqueue_many=False, shapes=None, dynamic_pad=False,
             allow_smaller_final_batch=False, shared_name=None, name=None)

def shuffle_batch_join(tensors_list, batch_size, capacity,
               min_after_dequeue, keep_input, seed=None,
               enqueue_many=False, shapes=None,
               allow_smaller_final_batch=False, shared_name=None,
               name=None)

batch_join 和 batch 的入队列方式不一样，
batch_join 是每个元素一个会有一个单独的线程处理，
batch 整个 tensors 分成 num_threads 个线程处理。
具体可以参考 enqueue 和  enquque_join 的区别


### python/training/training.py

def create_train_op(total_loss, optimizer,
        global_step=_USE_GLOBAL_STEP,
        update_ops=None,
        variables_to_train=None,
        transform_grads_fn=None,
        summarize_gradients=False,
        gate_gradients=tf_optimizer.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        check_numerics=True)

1. 初始化 global_step
2. 如果  update_ops 不为空，运行 update_ops
3. 创建梯度下降
4. 运行 transform_grads_fn
5. 计算梯度下降
6. 将 grad 写入 summary
返回所有 train_op

