

主要是协调多个线程的执行，将多个线程通过 register_thread  注册或传递给 join 中的 threads
参数，通过 request_stop() 请求停止线程，需要注意的是调用 request_stop()
并不会立马停止线程执行，而是等待 stop_grace_period_secs 秒，如果此时线程仍然没有
执行完成，就会抛出异常或记录日志，所以，基本就是实现了线程组的功能


def request_stop(self, ex=None)

在没有 join 之前，多次调用没有任何问题，如果在 join 之后调用，那么就会

def should_stop(self)

检查是否已经调用 self.request_stop()

def stop_on_exception(self)

如果希望在线程运行过程中出现任何异常就退出线程，就可以用该方法

```
with not stop_on_exception():
   body
```
 这样，在 body 中抛出异常，就会执行 self.request_stop()

def register_thread(self, thread)

将  thread 加入 self._registered_threads


def join(self, threads=None, stop_grace_period_secs=120, ignore_live_threads=False)

1. 取 threads 和 self._registered_threads 的并集
2. 如果所有线程都执行完成或者调用了 request_stop()，就等待 stop_grace_period_secs(默认120) 秒
线程退出时间，如果此时线程仍然没有执行完， 就记录没有执行完的线程
3. 如果 self._exc_info_to_raise 不为空，就抛出该异常，否则继续4
4. 检查是否存在没有执行完的线程，如果发出 request_stop 之后所有的线程都执行完了， 返回，否则继续5
5. 如果设置了如果设置 ignore_live_threads 为  True，就将没有停止的线程打日志，如果设置
 ignore_live_threads 为 False, 就抛出  RuntimeError
