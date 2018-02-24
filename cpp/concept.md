
## 概念

control flow context

### tips:

获取一个 Tensor 的类型
[x.dtype.base_dtype for x in inputs]

ops.convert_to_tensor 将一个 python list, scalars, numpy arrays 对象转为 Tensor,

  ```python
  import numpy as np

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

dtypes.as_dtype(dtype) : 类型转换
dtype.is_compatible_with : 判断两个类型是否兼容
ops.add_to_collections : 将一个变量加入 collections 的每一个元素
tensor_shape.unknown_shape() : 初始化一个 shape

## 数组操作


array_ops -> gen_array_ops


expand_dims(input, axis=None, name=None, dim=None)

  ```python
  # 't' is a tensor of shape [2]
  shape(expand_dims(t, 0)) ==> [1, 2]
  shape(expand_dims(t, 1)) ==> [2, 1]
  shape(expand_dims(t, -1)) ==> [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  ```

setdiff1d(x, y, index_dtype=dtypes.int32, name=None)

broadcast_dynamic_shape(shape_x, shape_y)

broadcast_static_shape(shape_x, shape_y)

shape(input, name=None, out_type=dtypes.int32)

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```
size(input, name=None, out_type=dtypes.int32)

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```
rank(input, name=None)

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```
slice(input_, begin, size, name=None)

  ```python
  # 'input' is [[[1, 1, 1], [2, 2, 2]],
  #             [[3, 3, 3], [4, 4, 4]],
  #             [[5, 5, 5], [6, 6, 6]]]
  tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
  tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                              [4, 4, 4]]]
  tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                             [[5, 5, 5]]]
  ```
strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None, name=None):

assign(val, name=None)


parallel_stack(values, name="parallel_stack"):
  ```python
  # 'x' is [1, 4]
  # 'y' is [2, 5]
  # 'z' is [3, 6]
  parallel_stack([x, y, z])  # => [[1, 4], [2, 5], [3, 6]]
  ```
stack(values, axis=0, name="stack")

  ```python
  # 'x' is [1, 4]
  # 'y' is [2, 5]
  # 'z' is [3, 6]
  stack([x, y, z])  # => [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
  stack([x, y, z], axis=1)  # => [[1, 2, 3], [4, 5, 6]]
  ```
  ```python
  tf.stack([x, y, z]) = np.asarray([x, y, z])
  ```

concat(values, axis, name="concat")
  ```python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
  tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
  ```
  ```python
  tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
  ```
  ```python
  tf.stack(tensors, axis=axis)
  ```

boolean_mask(tensor, mask, name="boolean_mask")
  ```python
  # 1-D example
  tensor = [0, 1, 2, 3]
  mask = np.array([True, False, True, False])
  boolean_mask(tensor, mask) ==> [0, 2]
  ```
  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = np.array([True, False, True])
  boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
  ```
sparse_mask(a, mask_indices, name=None)
  ```python
  # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
  # with shape [1000, 10]
  a.indices => [12, 26, 37, 45]
  tf.shape(a.values) => [4, 10]

  # `b` will be the subset of `a` slices at its second and third indices, so
  # we want to mask its first and last indices (which are at absolute
  # indices 12, 45)
  b = tf.sparse_mask(a, [12, 45])

  b.indices => [26, 37]
  tf.shape(b.values) => [2, 10]

  ```
### Device

### Graph

NOTE: 该类中的方法不是线程安全的

获取默认的图 tf.get_default_graph 或 tf.Graph.as_default


```
  g = tf.Graph()
  with g.as_default():
    # Define operations and tensors in `g`.
    c = tf.constant(30.0)
    assert c.graph is g
```


nodes_by_id : op.id 与 op 的 dict
nodes_by_name : op.name 与 op 的 dict
seed : 
finalized : finalized 为 True, 表明不可以增加 Operation
control_flow_context
functions : DefinedFunction 对象, function.name 与 function 的 dict
control_dependencies_stack : 保存所有 controller; create_op 的 operation 都加入其中的 controller
default_original_op : 用于 Operation
next_id_counter :
collections : 保存 key, value 的 dict, 其中 value 是一个 list, 同一 key 可以添加多个 value
name_stack : 是否已经存在命名空间, 命名空间支持层级嵌套, 参加 name_scope
handle_deleters : 保存已经被删除的 handler



关键操作

version(self) : op 中 id 的最大值
graph_def_versions(self):
seed(self):
finalized(self) : 是否可以增加 Operation

collections(self):
add_to_collection(self, name, value):
add_to_collections(self, names, value): 将 value 加入 collections 中 names 的每个元素对于的 value
get_collection_ref(self, name): 返回 name 对于的 values, 如果不存在就创建一个
get_collection(self, name, scope=None): scope 用于过滤
get_all_collection_keys(self):
clear_collection(self, name):
original_op(self, op): 在 with 设置 \_default_original_op 为 op
unique_name(self, name, mark_as_used=True): 根据 name 获取唯一的 name, 对于已经存在的 name, 变为 name_N 这里 N 为第几个
get_name_scope(self) : 

seed(self, seed) : 设置 seed
finalize(self) : 设置 finalized 为 True(表明不可以增加 Operation)
unsafe_unfinalize(self): 设置 finalized 为 False(表明可以继续增加 Operation)
add_op(self, op) : op.id 和 op.name 必须是不存在
as_graph_def(self, from_version=None, add_shapes=False): TODO
as_graph_def(self, from_version=None, add_shapes=False)
is_function(self, name)
get_function(self, name)
add_function(self, function)
building_function(self)
get_operations(self)
get_operation_by_name(self, name)
get_tensor_by_name(self, name)
get_operation_by_name_unsafe(self, name):
next_id(self): next_id_counter 加 1 后返回
last_id(self) :  next_id_counter 加 1 后返回



push_control_dependencies_controller(self, controller):
pop_control_dependencies_controller(self, controller):
current_control_dependencies(self):
record_op_seen_by_control_dependencies(self, op): 将 op 加入 control_dependencies_stack 中每个 controller 中
control_dependencies_for_inputs(self, input_tensors): 对于 input_tensors 中元素，不在 control_dependencies_stack 的任意元素中，返回之
original_op(self, op): 设置 default_original_op 为 op
set_shapes_for_outputs(op): 根据 op.type 找到 shape_func, 调用 shape_func 获取 shapes, 将 shapes 中的元素依次设置 op.outputs 的元素的 shape
as_graph_element(self, obj, allow_tensor=True, allow_operation=True): 返回 obj 对应的 op.output, obj 可以是 string, Tensor, Operation
colocate_with(self, op, ignore_existing=False):  在 with 语句中 设置 \_colocation_stack 只包括 op


attr_scope(self, attr_map)

在 with 块内，同样的 key, value, 用 attr_map 替代 self._attr_scope_map

    ```python
       with ops.Graph().as_default() as g:
         f_1 = Foo()  # No extra attributes
         with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=False)}):
           f_2 = Foo()  # Additional attribute _a=False
           with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=True)}):
             f_3 = Foo()  # Additional attribute _a=False
             with g._attr_scope({"_a": None}):
               f_4 = Foo()  # No additional attributes.
    ```

kernel_label_map(self, op_to_kernel_label_map):

在 with 语句块内，同样的 key, value, 用 op_to_kernel_label_map 替代 self._op_to_kernel_label_map

    ```python

        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.
    ```

gradient_override_map(self, op_type_map):

在 with 语句块内，同样的 key, value, 用 op_type_map 替代 self._gradient_override_map

    ```python
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      # ...

    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                              # gradient of s_2.
    ```


control_dependencies(self, control_inputs):


    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies([c, d]):
        # Ops constructed here run after `a`, `b`, `c`, and `d`.
    ```

    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies(None):
        # Ops constructed here run normally, not waiting for either `a` or `b`.
        with g.control_dependencies([c, d]):
          # Ops constructed here run after `c` and `d`, also not waiting
          # for either `a` or `b`.
    ```

    ```python
    # WRONG
    def my_func(pred, tensor):
      t = tf.matmul(tensor, tensor)
      with tf.control_dependencies([pred]):
        # The matmul op is created outside the context, so no control
        # dependency will be added.
        return t

    # RIGHT
    def my_func(pred, tensor):
      with tf.control_dependencies([pred]):
        # The matmul op is created in the context, so a control dependency
        # will be added.
        return tf.matmul(tensor, tensor)
    ```


\_apply_device_functions(self, op):

对 self._device_function_stack 排序后, 遍历 self._device_function_stack 对每个元素依次调用 op._set_device(device_function(op))

device(self, device_name_or_function): 在当前 with 语句中, 将 device_name_or_function 加入 \_device_function_stack 包括

    ```python
    with g.device('/gpu:0'):
      # All operations constructed in this context will be placed
      # on GPU 0.
      with g.device(None):
        # All operations constructed in this context will have no
        # assigned device.

    # Defines a function from `Operation` to device string.
    def matmul_on_gpu(n):
      if n.type == "MatMul":
        return "/gpu:0"
      else:
        return "/cpu:0"

    with g.device(matmul_on_gpu):
      # All operations of type "MatMul" constructed in this context
      # will be placed on GPU 0; all other operations will be placed
      # on CPU 0.
    ```

container(self, container_name):

在 with 语句块中，self._container 为 container_name

    ```python
    with g.container('experiment0'):
      # All stateful Operations constructed in this context will be placed
      # in resource container "experiment0".
      v1 = tf.Variable([1.0])
      v2 = tf.Variable([2.0])
      with g.container("experiment1"):
        # All stateful Operations constructed in this context will be
        # placed in resource container "experiment1".
        v3 = tf.Variable([3.0])
        q1 = tf.FIFOQueue(10, tf.float32)
      # All stateful Operations constructed in this context will be
      # be created in the "experiment0".
      v4 = tf.Variable([4.0])
      q1 = tf.FIFOQueue(20, tf.float32)
      with g.container(""):
        # All stateful Operations constructed in this context will be
        # be placed in the default resource container.
        v5 = tf.Variable([5.0])
        q3 = tf.FIFOQueue(30, tf.float32)

    # Resets container "experiment0", after which the state of v1, v2, v4, q1
    # will become undefined (such as uninitialized).
    tf.Session.reset(target, ["experiment0"])
    ```

as_default(self)

    临时设置 thread._global_default_graph 当前 Graph, 当 with 语句块中有效。
    提供了 get_default(self) 来获取, get_controller 来设置当前线程

    ```python
    # 1. Using Graph.as_default():
    g = tf.Graph()
    with g.as_default():
      c = tf.constant(5.0)
      assert c.graph is g

    # 2. Constructing and making default:
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      assert c.graph is g
    ```

create_op(self, op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True, compute_device=True): 创建一个 Operation，并加入当前 Graph 中，一般不建议用，应该用 tf.constant(), tf.Variable

    Operation 的 name 的命名规则：self._name_stack + "/" + name + "_" + i
    Operation 的 node_def.attr 由 attr_scope 设置
    Operation 的 control_inputs : 如果 inputs 中元素，不在 control_dependencies_stack 中任意元素中，就加入 control_inputs
    Operation 的 original_op 为 Graph default_original_op

name_scope(self, name):

    创建一个 name 的命名空间，临时修改 self._name_stack 为 name, 在当前 with 语句中有效，

    ```python
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, name="c")
      assert c.op.name == "c"
      c_1 = tf.constant(6.0, name="c")
      assert c_1.op.name == "c_1"

      # Creates a scope called "nested"
      with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"

        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
          nested_inner_c = tf.constant(20.0, name="c")
          assert nested_inner_c.op.name == "nested/inner/c"

        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
          nested_inner_1_c = tf.constant(30.0, name="c")
          assert nested_inner_1_c.op.name == "nested/inner_1/c"

          # Treats `scope` as an absolute name scope, and
          # switches to the "nested/" scope.
          with g.name_scope(scope):
            nested_d = tf.constant(40.0, name="d")
            assert nested_d.op.name == "nested/d"

            with g.name_scope(""):
              e = tf.constant(50.0, name="e")
              assert e.op.name == "e"
    ```


### Dtype

包含 real type 和 base type

通过 as_numpy_dtype 属性转为 numpy type

tf.as_dtype() 将 numpy type 转为为 tf.Dtype

核心在 types_pb2 上，将 np.dtype 与 Dtype 结合

判断 x 与 y 的类型是否兼容: x.is_compatible_with(y)


### 它们之间的关系

Tensor 可以作为 Operation 的输入

Tensor 通过 Operation 与 Graph 建立关系

Tensor 通过 Operation 与 Device 建立关系

Graph 由多个 Tensor 构建，一个 Graph 由一系列的 Operation 组成，

在 Tensor 的 eval 执行时，Session 已经创建, Graph 必须被启动

Variable 通过 value() 转为 Tensor

* node
* computational graph : 拓扑
* session : 运行时
* placeholder : 类似变量


## IndexedSlices

TODO

