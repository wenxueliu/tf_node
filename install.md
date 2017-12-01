
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
