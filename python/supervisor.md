

将 SessionManager 的封装，附带支持 summary, saver, checkpoint，并且支持集群，对于 is_chief 为 True 的 supervisor 会
执行初始化和启动线程执行 summary 和 saver 的工作。此外，还支持增加其他自定义服务，和对 summary 和
saver 的自定义实现。

此外，如果某个  supervisor 由于异常或宕机等等原因终止，重新启动，就会从 chief
supervisor 的  checkpoint 重新获取并初始化。

对 session 的管理

is_chief : 类似集群中的 master, 在分布式的  tensorflow 中，is_chief 为 True 的节点比其他节点多了以下操作：

1. 初始化模型参数
2. 从 checkpoint 恢复模型参数
3. 运行一些线程服务, 写 summary, saver，统计操作

### 关键变量

ops.GraphKeys.SUMMARY_OP : \_summary.merge_all()
ops.GraphKeys.GLOBAL_STEP : graph.get_tensor_by_name("global_step:0")
ops.GraphKeys.READY_OP: variables.report_uninitialized_variables()
ops.GraphKeys.READY_FOR_LOCAL_INIT_OP: 
ops.GraphKeys.SAVERS : saver.Saver()

### 线程服务

只有  is_chief 为 true 时，才会启动该线程

SVSummaryThread: 记录 global_step 到 summary 路径，供 tensorboard 查看
SVStepCounterThread: 记录每秒运行的  step 数量
SVTimerCheckpointThread: 记录 global_step 到 checkpoint 路径

### 队列服务

ops.GraphKeys.QUEUE_RUNNERS 中的每个元素一个线程


### 源码分析

def prepare_or_wait_for_session(self, master="", config=None,
    wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)

1. 如果 is_chief 为 True，创建 self._session_manager.prepare_session，开始线程服务，继续 3
2. 如果 is_chief 为 False，创建 self._session_manager.wait_for_session，继续 3
3. 开始队列服务

def loop(self, timer_interval_secs, target, args=None, kwargs=None)

创建一个线程以  timer_interval_secs  间隔调用  target(args,
kwargs)，主要用于增加新的服务

def summary_computed(self, sess, summary, global_step=None)

将  global_step 写入 summary

def \_verify_setup(self)

如果  is_chief 为  False, 所有的变量都要设置 device

def managed_session(self, master="", config=None, start_standard_services=True, close_summary_writer=True)

调用 prepare_or_wait_for_session 创建一个  session, 如果异常或正常执行完成，关闭 session，

 yield 和 @contextlib.contextmanager 的结合， 因此，可以以 with 方式执行
