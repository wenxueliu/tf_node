

https://zhuanlan.zhihu.com/p/29268015
https://zhuanlan.zhihu.com/p/31162922


SWIG(Simplified Wrapper and Interface Generator)

SWIG (Simplified Wrapper and Interface Generator) is a software development tool for building scripting language interfaces to C and C++ programs. Originally developed in 1995, SWIG was first used by scientists in the Theoretical Physics Division at Los Alamos National Laboratory for building user interfaces to simulation codes running on the Connection Machine 5 supercomputer. In this environment, scientists needed to work with huge amounts of simulation data, complex hardware, and a constantly changing code base. The use of a scripting language interface provided a simple yet highly flexible foundation for solving these types of problems. SWIG simplifies development by largely automating the task of scripting language integration--allowing developers and users to focus on more important problems.

Although SWIG was originally developed for scientific applications, it has since evolved into a general purpose tool that is used in a wide variety of applications- -in fact almost anything where C/C++ programming is involved.

## 安装
```shell
#! /bin/bash
swig="swig-3.0.12"
swig_package="${swig}.tar.gz"
wget -c http://prdownloads.sourceforge.net/swig/${swig_package}
tar xfvz ${swig_package}
cd $swig
./configure --without-pcre
make
#make install # 如你想安装的话，该句必须执行
```

## 例子

### 例1

在  ex1 文件夹下
```c
/* File : example.c */
#include <time.h>
double My_variable = 3.0;

int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}

int my_mod(int x, int y) {
    return (x%y);
}

char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
}
```
```swig
/* example.i */
%module example
%{
/* Put header files here or function declarations like below */
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
%}

extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
```

#### 生成包装文件

$ `swig -python example.i`

如果你没有安装  swig，那么可以采用下面的方式

$ `swig_dir=../swig`
$ `${swig_dir}/swig -python -I${swig_dir}/Lib/ -I${swig_dir}/Lib/python example.i`

#### 编译

$ `gcc -fPIC -c example.c example_wrap.c -I/usr/include/python2.7/`

#### 生成动态库

$ `ld -shared example.o example_wrap.o -o _example.so`

或编译成模块

```python
#!/usr/bin/env python

"""
etup.py file for SWIG C\+\+/Python example
"""
from distutils.core import setup, Extension
example_module = Extension('_example',
sources=['example.c', 'example_wrap.c',],
)
setup (name = 'example',
version = '0.1',
author = "test",
description = """Simple swig C\+\+/Python example""",
ext_modules = [example_module],
py_modules = ["example"],
)
```

$ `python setup.py build_ext --inplace`



注：swig生成的扩展模块对象名必须使用python模块名并在前面加上下划线_

#### 测试

$ python
Python 2.7.12 (default, Nov 20 2017, 18:23:56)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Tue Mar 13 15:27:18 2018\n'

如果 import example 出现导入错误，就将包含  exmaple 相关的文件夹加入  sys.path
即可

>>> import sys
>>> sys.path.append("ex1")

### 例2

```cpp
// pair.h.  A pair like the STL
#ifndef __PAIR_H__
#define __PAIR_H__
template<class T1, class T2> struct pair {
    T1 first;
    T2 second;
    pair() : first(T1()), second(T2()) { };
    pair(const T1 &f, const T2 &s) : first(f), second(s) { }
};
#endif
 ```

```swig
// pair.i - SWIG interface
%module pair
%{
#include "pair.h"
%}

// Ignore the default constructor
%ignore std::pair::pair();

// Parse the original header file
%include "pair.h"

// Instantiate some templates

%template(pairii) std::pair<int,int>;
%template(pairdi) std::pair<double,int>;
```
#### 生成包装文件

$ `swig -python -c++ example.i`

如果你没有安装  swig，那么可以采用下面的方式

$ `${swig_dir}/swig -python -c++ -I${swig_dir}/Lib/ -I${swig_dir}/Lib/python pair.i`

#### 编译

$ `c++ -fPIC -c pair_wrap.cxx -I/usr/include/python2.7/`

#### 生成动态库

$ `c++ -shared pair_wrap.o -o _pair.so`

#### 测试

$ python
Python 2.7.12 (default, Nov 20 2017, 18:23:56)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import pair
>>> a = pair.pairii(3,4)
>>> a.first
3
>>> a.second
4
>>> a.second = 16
>>> a.second
16
>>> b = pair.pairdi(3.5,8)
>>> b.first
3.5
>>> b.second
8

此外，在 swig 安装包的  Example 路径下包含各种语言大量的例子。可以参考


## c++ 与  python 交互

虽然 SWIG 可以解析绝大多数 C/C++ 的声明，但并没有提供一个完整的  C/C++ 解析器。

#### 目前支持的

* Full C99 preprocessing.
* All ANSI C and C++ datatypes.
* Functions, variables, and constants.
* Classes.
* Single and multiple inheritance.
* Overloaded functions and methods.
* Overloaded operators.
* C++ templates (including member templates, specialization, and partial specialization). Namespaces.
* Variable length arguments.
* C++ smart pointers.

#### 不支持的

1. Non-conventional type declarations

```
/* Non-conventional placement of storage specifier (extern) */ const int extern Number;
/* Extra declarator grouping */ Matrix (foo); // A global variable
/* Extra declarator grouping in parameters */ void bar(Spam (Grok)(Doh));
```

2. Certain advanced features of C++ such as nested classes are not yet fully supported.

TODO

%module

%module代表的是当前.i模板所在的模块，相对应的，该.i文件也会生成相应的接口文件，命名就与%module声明的一样。所以该语法一般用在模板的开头。
%include

就像C/C++一样，include会将需要生成接口的文件进行生成。是必不可少的语法。
%{%}

这个关键字帮助我们在.cxx中加入一些代码，例如我们最常用的#include，这样我们才可以让.cxx调用到相应的代码。
使用C++/STL

我们可以通过包含各种swig所包含的.i文件来帮助我们实现STL库。

例如%incude “std_string.i”、%include “std_vector.i”

    namespace std {
    %template(BoolVector) vector<bool>;
    }；

使用这样的定义方式，Swig会为我们生成一个名为BoolVector的类型而不是未知类型。我们可以在目标语言中创建C++中的STL并且与C++中的Vector进行互操作。

需要注意的是，如果我们使用自定义类型而非基本类型或者使用指针作为模板类型，我们则需要事先导出自定义类型的定义，否则就会得到SWIGTYPE_p_类型名这样定义作为类型模板的Vector定义，这往往不是我们想要的。
使用指针

定义指针的方法如下：

    %pointer_class(bool, BoolPointer);

通过这个定义我们Swig会为我们生成指针相对应的类，Swig再会生成类似于SWIGTYPE_p_bool这样的未定义类型，而是直接使用BoolPointer，并且我们能够自己在目标语言中申请内存，并且自己对内存进行管理。
使用数组

定义数组的方法如下。

    %array_class(unsigned char, UnsignedCharArray);

通过这种方式我们可以导出相应的数组类型。我们可以在目标语言中创建C++中的数组，并且与C++中的数组进行互操作。
typemap

有时候我们会对导出的内容不满意，例如C++中导出的函数中的参数类型为char*，但是到了CSharp中被自动转换成了string，如果我们同样想用数组来接收，则需要通过typemap来进行类型映射。

我们通过该代码生成相应的接口，我写了一个简单的测试类：

## 参考

http://www.swig.org/tutorial.html
https://github.com/swig/swig/wiki/FAQ#shared-libraries

## 附录

### How do I create shared libraries for Linux?

For C:

$ cc -fpic -c $(SRCS)
$ ld -shared $(OBJS) -o module.so

For C++:

$ c++ -fpic -c $(SRCS)
$ c++ -shared $(OBJS) -o module.so

If you are using GCC 4.x under Ubuntu and using python 2.6 try the following

$ swig -python module.i
$ gcc -fpic -I/usr/include/python2.6 -c module_wrap.c
$ gcc -shared module_wrap.o -o module.so
