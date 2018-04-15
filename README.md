
算起来辞职转 AI 已经有接近 4 个月了，近来偶有所得，而目前对于 tensorflow
的代码碎片不少（绝大多数来自官网），但是，对于 tensorflow 的源码解析的却
都是浅尝辄止，大概是牛逼的都忙着赚钱了吧。。。。

对阅读代码的思考，我常常在遇到难懂的地方，多想如果写代码的人能告诉我他的
思路多好，那样我就大概知道所以然了，理解起来就轻松多了。但是这只是想象。。


tensorflow 源码主要有两部分组成

1. 架构：这部分与软件设计相关，如模块设计，设计模式等
2. 机器学习领域相关：这部分需要有基本的机器学习相关知识


## 介绍

在开始之前，必须阅读的 tensorflow 的白皮书(见 white_paper_1，white_paper_2)
这里对理解 TensorFlow 的架构有非常重要的作用， 我刚开始没有读，从代码中理解，
理解很多概念都是读完才明白，如果一开始就看了白皮书，那么，看代码会轻松很多。

使用的初学者，建议至少读 white_paper_1 的前 4 部分。

如果有一定使用经验的建议都读

我自己的解读见 design.md，不得不说  paper 写得太好了

## TensorFlow 依赖

gtl
Arena
Protobuf

### 命令行解析

tensorflow/core/util/command_line_flags.h

## 阅读顺序

系统 main 函数

core/distributed_runtime/rpc/grpc_tensorflow_server.cc

服务初始化

core/distributed_runtime/server_lib.h
core/distributed_runtime/server_lib.cc
core/distributed_runtime/rpc/grpc_server_lib.h
core/distributed_runtime/rpc/grpc_server_lib.cc

服务运行

### 模型训练

1. 多个模型并行训练
2. 参数固定
