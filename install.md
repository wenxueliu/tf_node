
### python

sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate
pip install --upgrade tensorflow

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

deactivate

https://www.tensorflow.org/install/install_linux



### C++ 环境

参考 https://docs.bazel.build/versions/master/install-ubuntu.html

docker run -it tensorflow/tensorflow
apt-get update
apt-get install openjdk-8-jdk

docker 文件夹


## QA

在mac上装一个tensorflow做一些小的实验还是蛮方便的。用virtualenv的方式避免了很多麻烦。
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages tensorflow
pip install --upgrade
https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-py2-none-any.whl

但是在运行的时候发现报错：

    from matplotlib.backends import _macosx
        RuntimeError: Python is not installed as a framework. The Mac OS X
        backend will not be able to function correctly if Python is not
        installed as a framework. See the Python documentation for more
        information on installing Python as a framework on Mac OS X. Please
        either reinstall Python as a framework, or try one of the other
        backends. If you are Working with Matplotlib in a virtual enviroment see
        ‘Working with Matplotlib in Virtual environments’ in the Matplotlib FAQ

解决的方法也很简单：
vim ~/.matplotlib/matplotlibrc
然后输入以下内容：
backend: TkAgg

