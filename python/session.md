
## SessionManager

对 session 和  saver 的封装，在启动的时候，优先用 saver 从 checkpoint
恢复，如果恢复失败，调用初始化操作

Operation 初始化顺序

* init_op
* init_fn
* self._ready_for_local_init_op
* self._local_init_op
* self._ready_op


def \_restore_checkpoint(self, master, saver=None,
    checkpoint_dir=None, checkpoint_filename_with_path=None,
    wait_for_checkpoint=False,
    max_wait_secs=7200,
    config=None):

1. 创建 session
2. 从 checkpoint_dir 或  checkpoint_filename_with_path 恢复之前
运行状态

def prepare_session(self, master, init_op=None, saver=None,
        checkpoint_dir=None,
        checkpoint_filename_with_path=None,
        wait_for_checkpoint=False,
        max_wait_secs=7200,
        config=None,
        init_feed_dict=None,
        init_fn=None)

1. 创建 session
2. 从 checkpoint_dir 或 checkpoint_filename_with_path 恢复之前运行状态
3. 如果恢复失败，调用 init_op 和 init_fn 初始化
4. 如果 self._local_init_op 不为空，依次运行 self._ready_for_local_init_op, self._local_init_op 初始化
5. 如果 self._ready_op 不为空，运行 self._ready_op 初始化

def recover_session(self, master, saver=None,
        checkpoint_dir=None,
        checkpoint_filename_with_path=None,
        wait_for_checkpoint=False,
        max_wait_secs=7200,
        config=None)

1. 创建 session
2. 从 checkpoint_dir 或  checkpoint_filename_with_path 恢复之前运行状态
3. 如果 self._local_init_op 不为空，依次运行 self._ready_for_local_init_op, self._local_init_op 初始化
4. 如果 self._ready_op 不为空，运行 self._ready_op 初始化

def wait_for_session(self, master, config=None, max_wait_secs=float("Inf"))

1. 创建 session
2. 如果 self._local_init_op 不为空，依次运行 self._ready_for_local_init_op, self._local_init_op 初始化
3. 如果 self._ready_op 不为空，运行 self._ready_op 初始化
4. 如果失败，从1 开始重新运行，直到超过  max_wait_secs 或成功

## MonitorSession

SessionCreator
  ChiefSessionCreator
  WorkerSessionCreator
  \_CoordinatedSessionCreator

class SessionCreator(object)

创建 SessionManager 的工厂类

class ChiefSessionCreator(SessionCreator)

创建一个 prepare_session

class WorkerSessionCreator(SessionCreator)

创建一个 wait_for_session

class \_CoordinatedSessionCreator(SessionCreator)

1. 创建一个 session
2. 开启线程开始运行 ops.GraphKeys.QUEUE_RUNNERS
3.

\_WrappedSession
  \_RecoverableSession
  \_CoordinatedSession

#### _WrappedSession

class \_WrappedSession(object)

对 SessionManager 的简单包装

并提供了

* should_stop
* close
* run

等操作

#### _RecoverableSession

特点是  run 会一直执行

class \_RecoverableSession(_WrappedSession

def run(self, fetches, feed_dict=None, options=None, run_metadata=None)

一直循环调用 SessionManager 的 run 方法, 需要注意的是 SessionManager
有传入参数决定

### _CoordinatedSession

class \_CoordinatedSession(_WrappedSession)

将 coordinator 与 SessionManager 相结合

### \_HookedSession

遍历 hook, run 之前运行 before_run, 之后运行 after_run
```python
with MonitoredTrainingSession(hooks=your_hooks, ...) as sess:
  while not sess.should_stop():
    sess.run(your_fetches)
```
上面等同于下面的代码
```python
  call hooks.begin()
  sess = tf.Session()
  call hooks.after_create_session()
  while not stop is requested:
    call hooks.before_run()
    try:
      results = sess.run(merged_fetches, feed_dict=merged_feeds)
    except (errors.OutOfRangeError, StopIteration):
      break
    call hooks.after_run()
  call hooks.end()
  sess.close()
```


### SessionRunHook

class SessionRunHook(object)
* def begin(self)
* def after_create_session(self, session, coord)
* def before_run(self, run_context)
* def after_run(self, run_context,run_values)
* def end(self, session)

class SessionRunContext(object) : sess 与 sess 的参数组合起来
class SessionRunValues(collections.namedtuple("SessionRunValues", ["results", "options", "run_metadata"]))
class SessionRunArgs(collections.namedtuple("SessionRunArgs", ["fetches", "feed_dict", "options"]))

真搞不清楚这些类存在的必要性


### _MonitoredSession

问题：执行结束由谁来控制?

与原生的 SessionManager 主要的区别就是加上了 hook 功能

初始化过程中

* hook.begin()
* 创建 session，如果 checkpoint 存在，恢复变量
* 启动 queue
* hook.after_create_session()

调用 run 方法

* hook.before_run(run_context) : 参数 SessionRunContext，返回 SessionRunArgs

* session.run()
* hook.after_run()
* 返回 session.run() 的结果，如果发送异常，重新初始化或者尝试恢复 session

调用 close 方法

hook.end()
线程停止
关闭 queue 和 session

创建 session 之后，调用 hook.after_create_session
结束调用 hook.end

class \_MonitoredSession

def \__init__(self, session_creator, hooks, should_recover, stop_grace_period_secs=120)

如果  should_recover 为 True, 遇到 errors.AbortedError, errors.UnavailableError 错误，会一直重试，直到成功，run 也是一直运行

def run(self, fetches, feed_dict=None, options=None, run_metadata=None)

调用 self.sess.run, 最终映射到  prepare_session 或 wait_for_session

self.sess 最终映射到 ChiefSessionCreator(prepare_session)

SingularMonitoredSession
    session_creator = ChiefSessionCreator
    \_MonitoredSession(session_creator, hooks)
        \_CoordinatedSessionCreator(session_creator, hooks).create_session()
            self.hooks = hooks
            self.tf_sess = self._session_creator.create_session()
            \_CoordinatedSession(\_HookedSession(self.tf_sess, self.hooks), self.coord)
                self.sess = \_HookedSession(self.tf_sess, self.hooks)
                        \_HookedSession(_WrappedSession(self.tf_sess), self.hooks), self.coord)
                            self.sess = self.tf_sess

MonitoredSession(session_creator, hooks)
    \_MonitoredSession(session_creator, hooks)
        self._sess = \_CoordinatedSessionCreator(session_creator, hooks).create_session()
            self._hooks = hooks
            self._session_creator = session_creator
            self.tf_sess = self._session_creator.create_session()
            return  \_CoordinatedSession(\_HookedSession(self.tf_sess, self.hooks), self.coord)
                        self._coord = self.corrd
                        \_WrappedSession(\_HookedSession(self.tf_sess, self.hooks))
                            self._sess = \_HookedSession(self.tf_sess, self.hooks)
                                             self.hook = self.hooks
                                             \_WrappedSession(self.tf_sess)
                                                    self.sess = self.tf_sess

\_MonitorSession.run(fetches, feed_dict, options, run_metadata)
    self._sess.run(fetches, feed_dict, options, run_metadata)
        \_CoordinatedSession.run(fetches, feed_dict, options, run_metadata)
            \_WrappedSession.run(fetches, feed_dict, options, run_metadata)
                \_HookedSession(self.tf_sess, self.hooks).run(fetches, feed_dict, options, run_metadata)
                    actual_fetches = {'caller': fetches}
                    run_context = session_run_hook.SessionRunContext(
                        original_args=session_run_hook.SessionRunArgs(fetches, feed_dict),
                        session=self._sess)
                    options = options or config_pb2.RunOptions()
                    request = [ feed_dict.hooks.before_run() for hook in hooks ]
                    feed_dict[hook] = hook
                    run_metadata = run_metadata or config_pb2.RunMetadata()
                    outputs = \_WrappedSession.run(self,
                                                  fetches=actual_fetches,
                                                  feed_dict=feed_dict,
                                                  options=options,
                                                  run_metadata=run_metadata)
                        outputs = self.tf_sess.run(
                                                  fetches=actual_fetches,
                                                  feed_dict=feed_dict,
                                                  options=options,
                                                  run_metadata=run_metadata))
                    [ hooks.after_run() for hook in hooks ]
                    return outputs["caller"]

由此可见，_MonitorSession 提供了更好的封装


def MonitoredTrainingSession(master='',
                            is_chief=True,
                            checkpoint_dir=None,
                            scaffold=None,
                            hooks=None,
                            chief_only_hooks=None,
                            save_checkpoint_secs=600,
                            save_summaries_steps=USE_DEFAULT,
                            save_summaries_secs=USE_DEFAULT,
                            config=None,
                            stop_grace_period_secs=120,
                            log_step_count_steps=100)

对 MonitoredSession 的进一步封装，默认提供了

 StepCounterHook
 CheckpointSaverHook,
 SummarySaverHook

三种  hook, 因此，绝大多数情况下，不需要用户自己再写额外代码

### Scaffold

主要是一些初始化工作


### 例子

```python
# because it has provides saver and summary for chief, so it doesn't need to add hook at most time
master = "" # master address
is_chief = True
checkpoint_dir = "/tmp/checkpoint"
hooks = [] # for master or slave node hook
chief_only_hooks = [] # only for master node
save_checkpoint_secs=600
config = None # some config for session
with MonitoredTrainingSession(master,
    is_chief = is_chief,
    scaffold=None,
    hook = None,
    chief_only_hooks=None,
    checkpoint_dir = checkpoint_dir,
    save_checkpoint_secs = save_checkpoint_secs,
    save_summaries_steps = 100,
    save_summaries_secs = 100,
    config = config,
    stop_grace_period_secs=120,
    log_step_count_steps=100
    ) as sess:
  while not sess.should_stop():
     loss = sess.run(train_op)
  return loss
```

集群主节点
```python
saver_hook = CheckpointSaverHook(...)
summary_hook = SummarySaverHook(...)
session_creator = ChiefSessionCreator(master=..., config=...)
with MonitoredSession(session_creator,
                      hooks = [saver_hook, summary_hook],
                      stop_grace_period_secs=120) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```

集群从节点
  ```python
session_creator=WorkerSessionCreator(master=..., config=...)
with MonitoredSession(session_creator, stop_grace_period_secs=120)) as sess:
  while not sess.should_stop():
    sess.run(train_op)

  ```

单机版
```python
saver_hook = CheckpointSaverHook(...)
summary_hook = SummarySaverHook(...)
with SingularMonitoredSession(hooks = [saver_hook, summary_hook],
                      stop_grace_period_secs=120) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```
## Hook

Hook 机制主要在 session.run 之前和之后做一些操作。如果要实现自己的
Hook，只有继承  SessionRunHook 并实现相关接口即可。

LoggingTensorHook : 指定 step  间隔或时间间隔，sess.run 获取  tensor 的值，并将结果记录到日志文件
StopAtStepHook : 控制执行到指定的步数后，停止执行
CheckpointSaverListener : 在 saver 之前，之后做一些工作
CheckpointSaverHook : 记录 global_step, grap, graph_meta 到  checkpoint_dir
StepCounterHook : 将执行的指定步数所需要的时间记录到  summary
NanTensorHook :  记录  loss_tensor 的值，如果为无限大，就记录错误日志
SummarySaverHook : 将 scaffold.summary_op(或 summary_op) 与 global_step 写入  summary, 
GlobalStepWaiterHook : 让  global_step 运行指定步数之后开始训练
FinalOpsHook : 在训练完之后执行某些操作
FeedFnHook :  设置 feed
\_StopAfterNEvalsHook :  控制 evaluate_repeatedly 每轮执行 eval_op 的次数，类似于 batch_size
SummaryAtEndHook :  控制在训练完之后执行的操作
