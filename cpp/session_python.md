
### python

InteractiveSession 主要用于 ipython 调试使用

关键变量

graph
config
session
opened
closed
add_shapes

关键方法

graph(self)
graph_def(self)
sess_str(self)

as_default(self)

    ```python
    c = tf.constant(..)
    sess = tf.Session()

    with sess.as_default():
      assert tf.get_default_session() is sess
      print(c.eval())
    ```

    *N.B.* The `as_default` context manager *does not* close the
    session when you exit the context, and you must close the session
    explicitly.

    Alternatively, you can use `with tf.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default session is a property of the current thread. If you
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

    *N.B.* Entering a `with sess.as_default():` block does not affect
    the current default graph. If you are using multiple graphs, and
    `sess.graph` is different from the value of @{tf.get_default_graph},
    you must explicitly enter a `with sess.graph.as_default():` block
    to make `sess.graph` the default graph.

    ```python
    c = tf.constant(...)
    sess = tf.Session()
    with sess.as_default():
      print(c.eval())
    # ...
    with sess.as_default():
      print(c.eval())

    sess.close()
    ```
run(self, fetches, feed_dict=None, options=None, run_metadata=None):

由具体的 C 实现

    The `fetches` argument may be a single graph element, or an arbitrarily
    nested list, tuple, namedtuple, dict, or OrderedDict containing graph
    elements at its leaves.  A graph element can be one of the following types:

    * An @{tf.Operation}.
      The corresponding fetched value will be `None`.
    * A @{tf.Tensor}.
      The corresponding fetched value will be a numpy ndarray containing the
      value of that tensor.
    * A @{tf.SparseTensor}.
      The corresponding fetched value will be a
      @{tf.SparseTensorValue}
      containing the value of that sparse tensor.
    * A `get_tensor_handle` op.  The corresponding fetched value will be a
      numpy ndarray containing the handle of that tensor.
    * A `string` which is the name of a tensor or operation in the graph.

    The value returned by `run()` has the same shape as the `fetches` argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    ```python
       a = tf.constant([10, 20])
       b = tf.constant([1.0, 2.0])
       # 'fetches' can be a singleton
       v = session.run(a)
       # v is the numpy array [10, 20]
       # 'fetches' can be a list.
       v = session.run([a, b])
       # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
       # 1-D array [1.0, 2.0]
       # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
       MyData = collections.namedtuple('MyData', ['a', 'b'])
       v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
       # v is a dict with
       # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
       # 'b' (the numpy array [1.0, 2.0])
       # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
       # [10, 20].
    ```

partial_run(self, handle, fetches, feed_dict=None)


    ```python
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    res = sess.partial_run(h, r2, feed_dict={c: res})
    ```

partial_run_setup(self, fetches, feeds=None)

TODO

make_callable(self, fetches, feed_list=None, accept_options=False):

TODO

\_do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata):

对 target_list, fetch_list, feed_list 做转换，不同的 handle 是否为 None,
调用不同的函数

\_register_dead_handle(self, handle):

当 self._dead_handles 的数量超过 10 个, 经过处理后，加入 feed 和 fetches, 之后调用 self.run

所做的处理:

从 graph 中的 handle 对应的设备删除 deleter_key 对应的 handler, 将被删除的
handler 加入 graph._handle_deleters

\_update_with_movers(self, feed_dict, feed_map):

TODO

