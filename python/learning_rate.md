
目前支持的  learning_rate 计算方法包括

exponential_decay
piecewise_constant
polynomial_decay
natural_exp_decay
inverse_time_decay


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

计算方法

    p = global_step / decay_steps
    decayed_learning_rate = learning_rate * decay_rate ^ (p)

其中，当  staircase 为 True 时， p = math_ops.floor(p)

 例子

   ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                   100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
  )
```


def piecewise_constant(x, boundaries, values, name=None)

计算方法

当 x < boundaries[0], 返回 values[0]
当 x > boundaries[0] & x < boundaries[1], 返回 values[1]
当 x > boundaries[1] & x < boundaries[2], 返回 values[2]
以此类推
当 x > boundaries[n] 为  values[-1]

例子
  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

def polynomial_decay(learning_rate, global_step, decay_steps,
    end_learning_rate=0.0001, power=1.0,
    cycle=False, name=None):

计算方法

  global_step = min(global_step, decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                    (1 - global_step / decay_steps) ^ (power) +
                    end_learning_rate

当 global_step 小于  decay_steps 时，逐渐减少;
当 global 大于  decay_steps 时，为 end_learning_rate

当设置 cycle 为 True 时
  ```python
  decay_steps = decay_steps * ceil(global_step / decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                    (1 - global_step / decay_steps) ^ (power) +
                    end_learning_rate

会发现此时, learning_rate 会是一个不断减小的波浪的形状。

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  end_learning_rate = 0.01
  decay_steps = 10000
  learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                  decay_steps, end_learning_rate,
                                  power=0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
     tf.train.GradientDescentOptimizer(learning_rate)
     .minimize(...my loss..., global_step=global_step)
  )
  ```


def natural_exp_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None):

计算方法

p = global_step / decay_steps
decayed_learning_rate = learning_rate * exp(-decay_rate * p)

当 staircase = True 时 p = math_ops.floor(p)

例子

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  k = 0.5
  learning_rate = tf.train.exponential_time_decay(learning_rate, global_step, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
     tf.train.GradientDescentOptimizer(learning_rate)
     .minimize(...my loss..., global_step=global_step)
  )
  ```

def inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

计算方法

p = global_step / decay_steps
decayed_learning_rate = learning_rate / (1 + decay_rate * p)

当 staircase = True 时 p = math_ops.floor(p)
  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  k = 0.5
  learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
     tf.train.GradientDescentOptimizer(learning_rate)
     .minimize(...my loss..., global_step=global_step)
  )
  ```

