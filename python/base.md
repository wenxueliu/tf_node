
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



