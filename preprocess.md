

## 输入数据

pandas 库
numpy 库
tf.SparseTensor: sparse array

将有参数的函数包装为无参数的函数的三种方法


### 方法一
def my_input_function_training_set():
  return my_input_function(training_set)

classifier.fit(input_fn=my_input_fn_training_set, steps=2000)

### 方法二
lassifier.fit(input_fn=functools.partial(my_input_function,
                                          data_set=training_set), steps=2000)

### 方法三

classifier.fit(input_fn=lambda: my_input_fn(training_set), steps=2000)
