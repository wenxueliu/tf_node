
## 设计模式


## 库

six : 解决 python2 与 python3 的兼容性问题


## 便利性函数

all : 全部为 True, 才为 True
any : 任意一个为 True 即为 True
del : 删除一个 list 的所有元素
callable : 对象是否可以调用
getattr : 检查并获取对象的属性

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


## with 语句

1. with 本质是什么
2. 为什么要把一个语句块加入  with
3. with 执行流程


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
