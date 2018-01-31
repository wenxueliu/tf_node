
对  session 和  saver 的封装，在启动的时候，优先用 saver 从 checkpoint
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
