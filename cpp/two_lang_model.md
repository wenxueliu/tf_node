## 脚本语言

### 两种语言看世界

The two-language model of computing is extremely powerful because it exploits the strengths of each language.
C/C++ can be used for maximal performance and complicated systems programming tasks. Scripting languages can
be used for rapid prototyping, interactive debugging, scripting, and access to high-level data structures
such associative arrays.

In this programming model, the scripting language interpreter is used for high level control
whereas the underlying functionality of the C/C++ program is accessed through special scripting
language "commands."

most languages define a special API for adding new commands. Furthermore, a special foreign function interface
defines how these new commands are supposed to hook into the interpreter.

### 脚本语言调用底层语言的过程

### wrap 函数

Typically, when you add a new command to a scripting interpreter you need to do two things

In order to access this function from a scripting language, it is necessary to write a special "wrapper" function
that serves as the glue between the scripting language and the underlying C function. A wrapper function must do
three things :

* Gather function arguments and make sure they are valid.
* Call the C function.
* Convert the return value into a form recognized by the scripting language.

the final step is to tell the scripting language about the new function. This is usually done
in an initialization function called by the language when the module is loaded.

要考虑到一些因素

#### Variable linking

Variable linking refers to the problem of mapping a C/C++ global variable to a variable in the scripting language interpreter.

#### Constants

To make constants available, their values can be stored in scripting language variables such as
$RED, $BLUE, and $GREEN. Virtually all scripting languages provide C functions for creating
variables so installing constants is usually a trivial exercise.

#### Structures and classes

脚本语言在访问简单的函数和变量时没有什么问题，但是在处理类和变量时存在一些问题。

The most straightforward technique for handling structures is to implement a collection of accessor
functions that hide the underlying representation of a structure


#### Proxy classes

In certain cases, it is possible to use the low-level accessor functions to create a proxy class,
also known as a shadow class. A proxy class is a special kind of object that gets created in a
scripting language to access a C/C++ class (or struct) in a way that looks like the original structure
(that is, it proxies the real C++ class). For example, if you have the following C++ definition :

```cpp
class Vector {
public:
  Vector();
  ~Vector();
  double x, y, z;
};
```

A proxy classing mechanism would allow you to access the structure in a more natural manner
from the interpreter. For example, in Python, you might want to do this:

```python
>>> v = Vector()
>>> v.x = 3
>>> v.y = 4
>>> v.z = -13
>>> ...
>>> del v
```

When proxy classes are used, two objects are really at work--one in the scripting language,
and an underlying C/C++ object. Operations affect both objects equally and for all practical
purposes, it appears as if you are simply manipulating a C/C++ object.


### 编译脚本语言扩展

The final step in using a scripting language with your C/C++ application is adding your extensions to the scripting language itself. There are two primary approaches for doing this. The preferred technique is to build a dynamically loadable extension in the form of a shared library. Alternatively, you can recompile the scripting language interpreter with your extensions added to it.

#### 动态库和动态加载

将 C/C++ 代码编译为动态库

gcc -fpic -c example.c example_wrap.c -I/usr/local/include
gcc -shared example.o example_wrap.o -o example.so

在脚本语言中加载动态库

import example

在加载动态库的时候，经常出现某个动态库没有找到的情况，解决办法就是在链接的时候
将你需要的动态库加入链接选项

#### 静态库

With static linking, you rebuild the scripting language interpreter with extensions.
The process usually involves compiling a short main program that adds your customized
commands to the language and starts the interpreter. You then link your program with
a library to produce a new scripting language executable.

Although static linking is supported on all platforms, this is not the preferred
technique for building scripting language extensions. In fact, there are very few
practical reasons for doing this--consider using shared libraries instead.
