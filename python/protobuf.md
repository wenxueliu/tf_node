



## 安装

### 依赖

$ sudo apt-get install autoconf automake libtool curl make g++ unzip

### 下载 tar 包

UNZIP_DEST=.
mkdir -p ${UNZIP_DEST}
PROTOBUF_VERSION=3.4.0
PROTOBUF_URL=https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip
wget -c ${PROTOBUF_URL}
unzip "${PROTOBUF_ZIP}" -d "${UNZIP_DEST}"
cp "${UNZIP_DEST}/bin/protoc" /usr/local/bin/

### 编译安装

$ ./configure
$ make
$ make check
$ sudo make install
$ sudo ldconfig # refresh shared library cache.


## 使用

sudo apt-get install pkg-config
pkg-config --libs --cflags protobuf

## 例子

http://www.cnblogs.com/stephen-liu74/archive/2013/01/02/2841485.html
http://www.cnblogs.com/stephen-liu74/archive/2013/01/08/2845994.html

## Tensorflow 中的 protobuf

版本：PROTOBUF_VERSION=3.3.0

1. mkdir tensorflow/core/framework/
2. 将 tensorflow 源码的 tensorflow/core/framework/ 中的 proto 文件拷贝到 tensorflow/core/framework 中
3. protoc --cpp_out=.  --proto_path=. tensorflow/core/framework/*.proto

### 生成解读

option cc_enable_arenas = true; //对 cpp 有效的的构造和析构优化，参考[这里](https://developers.google.com/protocol-buffers/docs/reference/arenas)
#define PROTOBUF_FINAL final //对虚函数不可以被覆写的标志。参考[这里](http://en.cppreference.com/w/cpp/language/final)

```
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
```

## 参考

https://github.com/google/protobuf/blob/master/src/README.md
http://www.cnblogs.com/stephen-liu74/archive/2013/01/02/2841485.html

http://colobu.com/2015/01/07/Protobuf-language-guide/
