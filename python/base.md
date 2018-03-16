
## 模块

inspect : 类似反射的东西，重点补充
pickle
tarfile
urllib2
scipy.misc : 操作图片
PIL.Image
matplotlib.pyplot
from io import BytesIO
shutil: help(shutil) shutil.rmtree
glob : glob.glob
tempfile.mkdtemp()
copy.deepcopy
functools.partial
six : 解决 python2 与 python3 的兼容性问题


## 包导入

`__all__` : https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python


## 便利性函数

all : 全部为 True, 才为 True
any : 任意一个为 True 即为 True
del : 删除一个 list 的所有元素
callable : 对象是否可以调用
getattr : 检查并获取对象的属性
k,v = zip(a[0::2],a[1::2]) :  当 list 中元素顺序是  k,v,k,v 的时候有用
a[::-1] : 将 a 中的元素反转

## 拷贝与引用
```python
>>> a = [1, 2, 3, 4]
>>> b = [a]
>>> a[0]
1
>>> b[0]
[1, 2, 3, 4]
>>> b[0][0]
1
>>> b[0][0]  = 2
>>> b
[[2, 2, 3, 4]]
>>> a
[2, 2, 3, 4]
>>>
```
## 兼容性

from __future__ import print_function 用 print("abc") 代替 print abc

## 迭代器

zip 将两个集合一起遍历 for l1,l2 in zip(list1, list2)
enumerate 遍历集合时，包括索引  for idx, e in enumerate(list)

for key, new_value in six.iteritems(kwargs)

for key, new_value in kwargs.items()

## 数值

float("Inf")

## 装饰器

  @classmethod
  @property   : 可以直接当属性用
  @staticmethod :  声明为静态方法 直接通过类名调用
  @contextlib.contextmanager :
  with  : https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/
  yield : https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/

## with 语句

1. with 本质是什么
2. 为什么要把一个语句块加入  with
3. with 执行流程


## 抽象方法
```python
class DataDecoder(object):
  """An abstract class which is used to decode data for a provider."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def decode(self, data, items):
    pass

  @abc.abstractmethod
  def list_items(self):
    pass
```


## 判断

判断 None :  if a is not None 或  if a is None
判断类型  : isinstance(queue_closed_exception_types, tuple)
判断子类  : issubclass(t, errors.OpError)
判断 True, False : if a 或  if not a
判断可调用 : callable(to_proto)


## 异常

``` python
    raise ValueError("Must provide queue and enqueue_ops.")

    raise TypeError(
        "queue_closed_exception_types, when provided, "
        "must be a tuple of tf.error types, but saw: %s"
        % queue_closed_exception_types)
```
## 弱引用

weakref.WeakKeyDictionary()


## 工具函数

#### sorted, six.iterkeys

def device_key(dev):
  return "" if dev is None else dev
for dev in sorted(six.iterkeys(ops_on_device), key=device_key)

#### filter

def odd(num):
  return True if num % 2 == 0 else False

filter(odd, [1, 2, 3, 4])

filter(None, [1, 2, 3, 4])

filter(None, [1, -2, 0, 4])

#### 控制子类可以实现的方法

``` python
class Estimator(object):

  def _assert_members_are_not_overridden(self):
    allowed_overrides = set(['_call_input_fn', '_create_global_step'])
    estimator_members = set([m for m in Estimator.__dict__.keys()
                             if not m.startswith('__')])
    subclass_members = set(self.__class__.__dict__.keys())
    common_members = estimator_members & subclass_members - allowed_overrides
    overridden_members = [
        m for m in common_members
        if Estimator.__dict__[m] != self.__class__.__dict__[m]]
    if overridden_members:
      raise ValueError(
          'Subclasses of Estimator cannot override members of Estimator. '
          '{} does override {}'.format(self.__class__, overridden_members))
```

#### 对函数为参数的校验

``` python
_VALID_MODEL_FN_ARGS = set(
    ['features', 'labels', 'mode', 'params', 'self', 'config'])

def _verify_model_fn_args(model_fn, params):
  """Verifies model fn arguments."""
  args = set(util.fn_args(model_fn))
  if 'features' not in args:
    raise ValueError('model_fn (%s) must include features argument.' % model_fn)
  if params is not None and 'params' not in args:
    raise ValueError('model_fn (%s) does not include params argument, '
                     'but params (%s) is passed to Estimator.' % (model_fn,
                                                                  params))
  if params is None and 'params' in args:
    logging.warning('Estimator\'s model_fn (%s) includes params '
                    'argument, but params are not passed to Estimator.',
                    model_fn)
  if tf_inspect.ismethod(model_fn):
    if 'self' in args:
      args.remove('self')
  non_valid_args = list(args - _VALID_MODEL_FN_ARGS)
  if non_valid_args:
    raise ValueError('model_fn (%s) has following not expected args: %s' %
                     (model_fn, non_valid_args))
```

### 获取函数的参数



``` python
import inspect

def fn_args(fn):
  # Handle callables.
  if hasattr(fn, '__call__') and inspect.ismethod(fn.__call__):
    return tuple(inspect.getargspec(fn.__call__).args)

  # Handle functools.partial and similar objects.
  if hasattr(fn, 'func') and hasattr(fn, 'keywords') and hasattr(fn, 'args'):
    # Handle nested partial.
    original_args = fn_args(fn.func)
    if not original_args:
      return tuple()

    return tuple([
        arg for arg in original_args[len(fn.args):]
        if arg not in set((fn.keywords or {}).keys())
    ])

  # Handle function.
  return tuple(inspect.getargspec(fn).args)
```


### collections.namedtuple

``` python
class Point(collections.namedtuple('Point', [x, y])):
  def __new__(x, y):
    some check at here
    return super(Point, cls).__new__(cls, x, y))
```

### 设计典范

tensorflow/python/estimator/run_config.py 配置类的典范

1. 提供了灵活的参数设置方案
2. 参数校验的方式很赞



## 附录

```
#!/usr/bin/env python
# encoding: utf-8

import threading
import contextlib as _contextlib

def contextmanager(target):
  """A tf_decorator-aware wrapper for `contextlib.contextmanager`.

  Usage is identical to `contextlib.contextmanager`.

  Args:
    target: A callable to be wrapped in a contextmanager.
  Returns:
    A callable that can be used inside of a `with` statement.
  """
  context_manager = _contextlib.contextmanager(target)
  #return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
  return context_manager

class Name(object):
    def __init__(self, name):
        self.name = name;

    def get_name(self):
        return self.name

    def as_default(self):
        return _default_graph_stack.get_controller(self)

    def set_default(self, name):
        return _default_graph_stack.get_controller(self)

class _DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(_DefaultStack, self).__init__()
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  @contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    try:
      self.stack.append(default)
      yield default
    finally:
        self.stack.remove(default)

class _DefaultGraphStack(_DefaultStack):
  """A thread-local stack of objects for providing an implicit default graph."""

  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self):
    """Override that returns a global default if the stack is empty."""
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph()
    return ret

  def _GetGlobalDefaultGraph(self):
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Name("1")
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

_default_graph_stack = _DefaultGraphStack()


if __name__ == "__main__":
  print _default_graph_stack.is_cleared()
  old = Name("2")
  with old.as_default() as new:
      print _default_graph_stack.is_cleared()
  print _default_graph_stack.is_cleared()
```
