
## 模块

pickle
tarfile
urllib2
scipy.misc : 操作图片
PIL.Image
matplotlib.pyplot
from io import BytesIO
shutil: help(shutil) shutil.rmtree
glob : glob.glob


## 迭代器

zip 将两个集合一起遍历 for l1,l2 in zip(list1, list2)
enumerate 遍历集合时，包括索引  for idx, e in enumerate(list)

## 数值

float("Inf")

## 装饰器

  @classmethod
  @property   : 可以直接当属性用
  @staticmethod :  声明为静态方法
  @contextlib.contextmanager :
  with  : https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/
  yield : https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/

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


##

def device_key(dev):
  return "" if dev is None else dev
for dev in sorted(six.iterkeys(ops_on_device), key=device_key)
