


```python

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.training import server_lib

server_config = config_pb2.ConfigProto(device_count={'CPU': 2})
server1 = server_lib.Server.create_local_server(config=server_config)
server2 = server_lib.Server.create_local_server(config=server_config)

cluster_def = cluster_pb2.ClusterDef()
job = cluster_def.job.add()
job.name = 'worker'
job.tasks[0] = server1.target[len('grpc://'):]
job.tasks[1] = server2.target[len('grpc://'):]
config = config_pb2.ConfigProto(cluster_def=cluster_def)

with ops.Graph().as_default() as g:
  with ops.device('/job:worker/task:1/cpu:1'):
    input1 = constant_op.constant(17, dtypes.float32)
    feed1 = array_ops.placeholder(dtypes.float32, shape=(2))
    mul1 = input1 * feed1

  with ops.device('/job:worker/task:0/cpu:1'):
    input2 = constant_op.constant(3, dtypes.float32)
    feed1 = array_ops.placeholder(dtypes.float32, shape=(2))
    mul2 = input2 * feed1

  with ops.device('/job:worker/task:1/cpu:0'):
    sum1 = mul1 + mul2


ones = np.ones([2])
run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
run_metadata = config_pb2.RunMetadata()
with session.Session(server1.target, config=config, graph=g) as sess:
  output = sess.run(sum1,
                    options=run_options,
                    run_metadata=run_metadata,
                    feed_dict = {
                       feed1: ones,
                       feed2: ones,
                    })
  #TODO body

len([
   node_stats
   for dev_stats in run_metadata.step_stats.dev_stats
   for node_stats in dev_stats.node_stats
   if '/job:worker/replica:0/task:1/device:CPU:0' ==
   dev_stats.device and 'Const' == node_stats.node_name
])


my_log_dir = "temp",
number_of_steps = 10000
save_summaries_secs = 5
save_interval_secs = 5
learning_rate = 
momentum = 

trace_every_n_steps = number_of_steps/100 if number_of_steps > 100 ? else 1
log_every_n_steps = number_of_steps/100 if number_of_steps > 100 ? else 1
summary_writer = tf.summary.FileWriter(self.config.train_dir, sess.graph)

images, labels = LoadData(...)
predictions = MyModel(images)

slim.losses.log_loss(predictions, labels)
total_loss = slim.losses.get_total_loss()


optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
opt = tf.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=50, total_num_replicas=50)

train_op = create_train_op(total_loss, optimizer)

checkpoint_path = '/path/to/old_model_checkpoint'
variables_to_restore = slim.get_model_variables()
init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        checkpoint_path, variables_to_restore)
# Create an initial assignment function.
def InitAssignFn(sess):
  sess.run(init_assign_op, init_feed_dict)

train_step_kwargs = {
    "logdir": my_log_dir
    "should_trace": "true",
    "summary_writer": summary_writer,
    "should_log": should_log
}

saver = tf.train.Saver(tf.global_variables())

slim.learning.train(train_op, my_log_dir,
        train_step_kwargs = train_step_kwargs,
        log_every_n_steps = log_every_n_steps,
        number_of_steps = number_of_steps,
        init_fn = InitAssignFn,
        save_summaries_secs = save_summaries_secs,
        saver = saver,
        save_interval_secs = save_interval_secs,
        trace_every_n_steps = trace_every_n_steps,
        sync_optimizer = opt)



# Load the Pascal VOC data
image, label = MyPascalVocDataLoader(...)
images, labels = tf.train.batch([image, label], batch_size=32)

# Create the model
predictions = vgg.vgg_16(images)

train_op = slim.learning.create_train_op(...)

# Specify where the Model, trained on ImageNet, was saved.
model_path = '/path/to/pre_trained_on_imagenet.checkpoint'

# Specify where the new model will live:
log_dir = '/path/to/my_pascal_model_dir/'

# Restore only the convolutional layers:
variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

# Start training.
slim.learning.train(train_op, log_dir, init_fn=init_fn)



### 源码分析

### learning

def clip_gradient_norms(gradients_to_variables, max_norm)

对  gradients_to_variables 中的每个变量计算  clip_by_norm，之后返回计算后的 list


def multiply_gradients(grads_and_vars, gradient_multipliers)

遍历 grads_and_vars(grad, var), 如果其中的 var 存在于 gradient_multipliers 中，
   grad *= gradient_multipliers[var]

def add_gradients_summaries(grads_and_vars)

将  grads_and_vars 中的  grad 写入 summary(以两种形式，histogram 和 clip_ops.global_norm)

def create_train_op(total_loss, optimizer,
        global_step=_USE_GLOBAL_STEP,
        update_ops=None,
        variables_to_train=None,
        clip_gradient_norm=0,
        summarize_gradients=False,
        gate_gradients=tf_optimizer.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        gradient_multipliers=None,
        check_numerics=True)

1. 初始化 global_step
2. 如果  update_ops 不为空，运行 update_ops
3. 创建梯度下降
4. 运行 transform_grads_fn 分别计算 multiply_gradients 和  clip_gradient_norms
5. 计算梯度下降
6. 将 grad 写入 summary
返回所有 train_op

def train_step(sess, train_op, global_step, train_step_kwargs)

def train(train_op,
          logdir,
          train_step_fn=train_step,
          train_step_kwargs=_USE_DEFAULT,
          log_every_n_steps=1,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=600,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0,
          saver=None,
          save_interval_secs=600,
          sync_optimizer=None,
          session_config=None,
          trace_every_n_steps=None)

1. 参数校验
2. 如果 global_step 没有初始化，初始化之
3. 初始化 init_op, ready_op, local_init_op
4. 如果 sync_optimizer 不为空，遍历  sync_optimizer 调用对应的 chief_init_op，local_step_init_op, ready_for_local_init_op
5. 初始化  saver, summary_op, summary_writer
6. 初始化  supervisor 对象
7. 创建一个 session, 开启相关的队列和服务(summar, saver 已经初始化操作)
8. 循环执行 train_step，直到出现异常或运行指定的步骤

def train_step(sess, train_op, global_step, train_step_kwargs)

1. 运行 train_op
2. 写相关日志
