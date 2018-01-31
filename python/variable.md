
## Variable

设计源代码

tensorflow/core/kernels/variable_op.cc
tensorflow/contrib/framework/python/ops/variables.py
tensorflow/python/framework/ops.py
tensorflow/python/ops/variables.py
tensorflow/python/ops/array_ops.py

默认 shape 和 dtype 是固定的，通过 validate_shape=False 使得 shape 和 dtype 可变

### 变量类型

全局变量为分布式环境所有主机的变量，
保存在 ops.GraphKeys.GLOBAL_VARIABLES 中, 所有变量默认都加入全局变量，
通过 tf.global_variables() 获取所有全局变量

本地变量一般用于中间结果，临时数据, 为每台机器的变量，保存在 ops.GraphKeys.LOCAL_VARIABLES 中，
通过 tf.contrib.framework.local_variable() 或 variable_scope.variable() 将一个变量加入本地变量，
通过 tf.local_variables() 获取本地变量

模型变量用于模型中定义的变量，比如 FP 中。在 ops.GraphKeys.MODEL_VARIABLES 中，
通过 tf.contrib.framework.model_variable() 或 tf.contrib.framework.add_model_variable(var),
variable_scope.variable() 将一个变量加入局部变量，
通过 tf.model_variables() 获取模型变量

模型变量一个重要的用途是从  checkpoint 中恢复，比如你正在训练一个模型，结果中途宕机或程序出现 bug
导致退出，为了避免从头开始训练，一般的做法是每训练 N 步，将模型参数保存到文件中，这样下次启动就从
上次训练结束的位置重新开始训练

模型变量还有一个重要的用途是，在迁移学习的时候，你已经有另外一个模型的参数，为了将该参数应用的新的
模型，将之前模型的参数保存起来，新的模型可以选择性地添加部分模型参数。

可训练变量被 optimizer 训练的变量, 保存在 ops.GraphKeys.TRAINABLE_VARIABLES 中，
在创建变量的时候，通过指定参数 trainable 为 True，使其加入可训练变量
通过 tf.trainable_variables() 获取可训练变量

移动平均变量，保存在 ops.GraphKeys.MOVING_AVERAGE_VARIABLES
通过 tf.moving_average_variables() 获取移动平均变量


### 变量初始化

tf.variables_initializer(var_list, name="init") : 初始化 var_list 中的变量
tf.global_variables_initializer() : 初始化所有全局变量
tf.local_variables_initializer() : 初始化所有本地变量
tf.assert_variables_initialized(var_list) : 检查 var_list 中的变量是否初始化，如果 var_list 为 None, 检查全局变量和本地变量
tf.report_uninitialized_variables(var_list=None, name="report_uninitialized_variables"): 返回 var_list 没有初始化的变量，如果 var_list 为 None, 检查全局变量和本地变量

初始化变量的方式

1. initializer op
2. 从保存的文件中恢复


    # Add an Op to initialize global variables.
    init_op = tf.global_variables_initializer()

    # Launch the graph in a session.
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(init_op)
        # ...you can now run any Op that uses variable values...



    ```python
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```
    ```python
    # Initialize 'v' with a random tensor.
    v = tf.Variable(tf.truncated_normal([10, 40]))
    # Use `initialized_value` to guarantee that `v` has been
    # initialized before its value is used to initialize `w`.
    # The random values are picked only once.
    w = tf.Variable(v.initialized_value() * 2.0)
    ```
    ```python
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        v.load([2, 3], sess)
        print(v.eval(sess)) # prints [2 3]
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        v.load([3, 4], sess)
        print(v.eval()) # prints [3 4]
    ```

如果初始化的变量依赖另一个变量，可以通过 initialized_value() 确保变量的初始化顺序

GraphKeys.GLOBAL_VARIABLES
GraphKeys.TRAINABLE_VARIABLES : 当变量初始化时，设置trainable 为 True

\_snapshot : 当前对象对应的 Tensor 对象, array_ops.identity(self._variable, name="read")
\_save_slice_info 
\_initializer_op : 初始化的操作函数
\_variable : 由 C API 创建的变量
\_initial_value : initial_value 转为 Tensor
\_initializer_op : self._variable.assign(self._initial_value).op

关键方法

name(self): self._variable.name
initializer(self): self._initializer_op
device(self): self._variable.device
dtype(self): self._variable.dtype
op(self): self._variable.op
graph(self): self._variable.graph
shape(self): self._variable.get_shape()
get_shape(self): self.shape

\_AsTensor(self) : 返回 self._snapshot
value(self) : 同 \_AsTensor, 获取当前变量的 value
read_value(self): array_ops.identity(self._variable, name="read")
\_ref(self) : 返回 self._variable
set_shapt() : 设置 self._variable 和 self._snapshot 的 shape
eval(self)  : self._variable.eval(session=session)
\_strided_slice_assign() :TODO

initial_value(self): 返回 self._initial_value, 具体由 C 代码实现
assign(self, value, use_locking=False): 赋值, 具体由 C 代码实现
assign_add(self, delta, use_locking=False): 给当前变量增加一个值, 具体由 C 代码实现
assign_sub(self, delta, use_locking=False): 给当前变量减少一个值, 具体由 C 代码实现
scatter_sub(self, sparse_delta, use_locking=False): TODO
count_up_to(self, limit): 增加到 limit, 具体由 C 代码实现
load(self, value, session=None): 加载一个新值到变量,问题与 assign 有什么区别? session.run(self._initializer_op, {self._initializer_op.inputs[1]: value})
\_TensorConversionFunction(v, dtype=None, name=None, as_ref=False): 将 v 转为 Tensor, 根据 as_ref 指明是传值还是引用

\_OverloadAllOperators() : 重载 Tensor 所允许重载的的所有操作
\_OverloadOperator(operator):  重载 operator 操作
\_build_initializer_expr(self, initial_value): 根据 initial_value 不同类型进行初始化
\_find_initialized_value_for_variable(self, variable_op): 从ops.GraphKeys.GLOBAL_VARIABLES 和 ops.GraphKeys.LOCAL_VARIABLES找到和 variable_op.node_def.name 的 Variable
to_proto(self, export_scope=None): 将 Variable 转换为 VariableDef
from_proto(variable_def, import_scope=None): 将 VariableDef 转换为 Variable

cpp 实现参考 ./tensorflow/tensorflow/core/ops/state_ops.cc

通过 ops.register_dense_tensor_like_type 可以查到所有的 Tensor 类型
通过 ops.register_tensor_conversion_function 可以查到基本类型是如何转换为 Tensor 类型的


### 创建变量

local_variable(initial_value, validate_shape=True, name=None)


### 初始化变量

zero_initializer(ref, use_locking=True, name="zero_initializer")
add_model_variable(var) : 将  var 加入 ops.GraphKeys.MODEL_VARIABLES
assign_from_values(var_names_to_values) : TODO

### 获取变量

get_variables(scope=None, suffix=None, collection=ops.GraphKeys.GLOBAL_VARIABLES)
get_model_variables(scope=None, suffix=None)
get_local_variables(scope=None, suffix=None)
get_trainable_variables(scope=None, suffix=None)
get_variables_to_restore(include=None, exclude=None)
get_variables_by_suffix(suffix, scope=None)
get_variables_by_name(given_name, scope=None)
get_unique_variable(var_op_name)
get_variable_full_name(var)
assign_from_checkpoint(model_path, var_list, ignore_missing_vars=False)
assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False, reshape_variables=False)
filter_variables(var_list, include_patterns=None, exclude_patterns=None, reg_search=True)

### 特殊变量

assert_global_step(global_step_tensor)
assert_or_get_global_step(graph=None, global_step_tensor=None)
get_global_step(graph=None)
create_global_step(graph=None)
get_or_create_global_step(graph=None)


## 实例

```
import tensorflow as tf
w = tf.Variable('a', trainable=True)
w1  = tf.Variable(w.initialized_value())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print sess.run(w)
print sess.run(w1)
sess.run(tf.global_variables()) ## 返回全局变量的 list
sess.run(tf.trainable_variables()) ## 返回可训练变量的 list
```


## 源码分析

def assign_from_checkpoint(model_path, var_list, ignore_missing_vars=False)

1. 将  var_list 转为 dict 保存在 grouped_vars
2. 遍历 grouped_vars，读取 model_path 找到每个 key 对应的 value

返回值直接传递给  sess.run() 即可获取

def assign_from_checkpoint_fn(model_path, var_list,
    ignore_missing_vars=False, reshape_variables=False)

调用  saver.restore  从 model_path 恢复 var_list 对应的变量
