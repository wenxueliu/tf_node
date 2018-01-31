
## state_ops

对  state_op 的理解存在问题

def variable_op(shape, dtype, name="Variable", set_shape=True, container="", shared_name="")

deprecated

def variable_op_v2(shape, dtype, name="Variable", container="", shared_name="")

def init_variable(v, init, name="init")

def is_variable_initialized(ref, name=None)

def assign_sub(ref, value, use_locking=None, name=None)

def assign_add(ref, value, use_locking=None, name=None)

def assign(ref, value, validate_shape=None, use_locking=None, name=None)

## clip_ops

问题：目前还不明白为什么要这样计算

def clip_gradient_norms(gradients_to_variables, max_norm)

def clip_by_norm(t, clip_norm, axes=None, name=None)

返回 t * clip_norm * min(1/sqrt(sum(t*t)), 1/clip_norm)

def clip_by_value(t, clip_value_min, clip_value_max, name=Noneo

```
clip_value_max > clip_value_min
if t > clip_value_max return clip_value_max
if t < clip_value_min return clip_value_min
if clip_value_min < t < clip_value_max, return t

clip_value_max < clip_value_min
if clip_value_min > t > clip_value_max return clip_value_min
if t < clip_value_max return clip_value_min
if clip_value_min < t, return clip_value_min

这个存在 bug, 要校验 clip_value_min 和 clip_value_max 的大小
```

def global_norm(t_list, name=None)

global_norm = sqrt(2 * sum([l2norm(t)**2 for t in t_list]))

def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)

1. if use_norm is None : global_norm = sqrt(2 * sum([l2norm(t)**2 for t in t_list]))
2. scale = clip_norm * min(1/clip_norm, 1/global_norm)
3. a = [ t * scale for t in t_list ]
返回 a, use_norm

def clip_by_average_norm(t, clip_norm, name=None)

返回 t * clip_norm * min(1/sqrt(sum(t*t)) * tf.size(a), 1/clip_norm)
 其中  tf.size(t) 是 t 中元素个数


