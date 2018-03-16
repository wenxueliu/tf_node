


def Assert(condition, data, summarize=None, name=None)

TODO
如果 data 的每个元素是 string 或 int，
if condition = True,  什么也不做
if condition = False，打印 data

否则

  ```python
  # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  with tf.control_dependencies([assert_op]):
    ... code using x ...
  ```
