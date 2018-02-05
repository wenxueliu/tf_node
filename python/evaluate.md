

evaluate_once : 执行 eval_op 的次数由  StopAfterNEvalsHook 来决定，slim 中的 evaluate_once 增加了 StopAfterNEvalsHook，因此，使用起来更加方便。

tensorflow/contrib/slim/python/slim/evaluation.py

def evaluate_once(master,
                  checkpoint_path,
                  logdir,
                  num_evals=1,
                  initial_op=None,
                  initial_op_feed_dict=None,
                  eval_op=None,
                  eval_op_feed_dict=None,
                  final_op=None,
                  final_op_feed_dict=None,
                  summary_op=_USE_DEFAULT,
                  summary_op_feed_dict=None,
                  variables_to_restore=None,
                  session_config=None)

1. 增加 hooks, evaluation.StopAfterNEvalsHook, evaluation.SummaryAtEndHook 到 hook
2. 执行 hook 和 eval_op 操作 num_evals 次
3. 执行 final_op

首先会尝试从 checkpoint_path 中加载数据，之后对 eval_op 执行 num_evals 次
最后会运行 final_op 操作(如果 final_op 不为空)

参考 evaluation.evaluate_once

def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    initial_op=None,
                    initial_op_feed_dict=None,
                    init_fn=None,
                    eval_op=None,
                    eval_op_feed_dict=None,
                    final_op=None,
                    final_op_feed_dict=None,
                    summary_op=_USE_DEFAULT,
                    summary_op_feed_dict=None,
                    variables_to_restore=None,
                    eval_interval_secs=60,
                    max_number_of_evaluations=None,
                    session_config=None,
                    timeout=None,
                    hooks=None)

1. 增加 hooks, evaluation.StopAfterNEvalsHook, evaluation.SummaryAtEndHook 到 hook
2. 如果 variables_to_restore 不为空，用 saver 恢复  variables_to_restore 中的变量
3. 创建 eval_step 加入 eval_ops
4. 创建更新 eval_step 的  Operation, 设置 hook 中的 StopAfterNEvalsHook 用以控制终止条件
5. 用 final_op, final_op_feed_dict 构造 FinalOpsHook，加入 hooks
6. eval_interval_secs 间隔遍历 checkpoint_dir 获取新的 checkpoint 当有新的 checkpoint，
就执行 hooks 和 eval_op 指定次数，该次数由 StopAfterNEvalsHook
初始化的时候指定；当超时没有获取到新的  checkpoint 或者执行 num_evals 次之后，返回

总共执行 eval_ops 的次数由 num_evals * max_number_of_evaluations 来决定

tensorflow/python/training/evalution.py

def \_get_or_create_eval_step()

获取或创建一个 eval_step 变量，同时保存在 ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP

def \_evaluate_once(checkpoint_path,
                master='',
                scaffold=None,
                eval_ops=None,
                feed_dict=None,
                final_ops=None,
                final_ops_feed_dict=None,
                hooks=None,
                config=None)

1. 增加 hooks, evaluation.StopAfterNEvalsHook, evaluation.SummaryAtEndHook 到 hook
2. 执行 eval_op 操作，具体的执行多少次，需要在 hooks 中增加 evaluation.StopAfterNEvalsHook 来指定
执行完之后，执行 final_op

```
tensorflow/contrib/training/python/training/evaluation.py
```

def wait_for_new_checkpoint(checkpoint_dir,
                        last_checkpoint=None,
                        seconds_to_sleep=1,
                        timeout=None)

等待 timeout 直到超时或者 checkpoint_dir 中有新的 checkpoint 文件生产

timeout 是总共等待的时间，而 seconds_to_sleep 每次等待  checkpoint
出现的等待时间，因此，正常情况下应该小于 timeout

def checkpoints_iterator(checkpoint_dir,
                    min_interval_secs=0,
                    timeout=None,
                    timeout_fn=None)

返回迭代器，每隔 min_interval_secs 获取新的 checkpoint

timeout_fn 控制超时没有获取到新的  checkpoint 该如何操作，如果 timeout_fn
为 None 或者返回 True, 重新等待，否则迭代返回空，结束迭代


def \_scaffold_with_init(scaffold, saver, checkpoint_path)

用 scaffold 初始化一个 monitored_session.Scaffold 对象


def evaluate_repeatedly(checkpoint_dir,
                        master='',
                        scaffold=None,
                        eval_ops=None,
                        feed_dict=None,
                        final_ops=None,
                        final_ops_feed_dict=None,
                        eval_interval_secs=60,
                        hooks=None,
                        config=None,
                        max_number_of_evaluations=None,
                        timeout=None,
                        timeout_fn=None)

1. 创建 eval_step 加入 eval_ops
2. 创建更新 eval_step 的  Operation, 设置 hook 中的 StopAfterNEvalsHook 用以控制终止条件
3. 用 final_op, final_op_feed_dict 构造 FinalOpsHook，加入 hooks
4. eval_interval_secs 间隔遍历 checkpoint_dir 获取新的 checkpoint 当有新的 checkpoint，
就执行 hooks 和 eval_op 指定次数，该次数由 StopAfterNEvalsHook 初始化的时候指定

注:
hook 中必须包含  StopAfterNEvalsHook 用以控制每次 eval_op 执行的次数， 类似于  batch_size;
max_number_of_evaluations 控制执行多少轮，类似于 batch_num; batch_size * batch_num 为总执行次数

是对 MonitoredSession 的封装，因此，详细实现还要结合 session.md
