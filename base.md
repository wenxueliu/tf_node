
Tensorflow 基础设施

## 设计模式

Builder
Singleton : OpRegistry
Factory :
Proxy : OpDefBuilderWrapper, RenameDevice
抽象工厂模式 DeviceFactory

队列 : Rendezvous 是非常巧妙的无阻塞队列设计

## 类的设计原则

所有的类遵循

1. 通过 Registery 注册
2. Factory 创建
3. 具体的实现

## 返回状态


class Status {
  struct State {
    tensorflow::error::Code code;
    string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;
}


enum Code {
  OK = 0;
  CANCELLED = 1;
  UNKNOWN = 2;
  INVALID_ARGUMENT = 3;
  DEADLINE_EXCEEDED = 4;
  NOT_FOUND = 5;
  ALREADY_EXISTS = 6;
  PERMISSION_DENIED = 7;
  UNAUTHENTICATED = 16;
  RESOURCE_EXHAUSTED = 8;
  FAILED_PRECONDITION = 9;
  ABORTED = 10;
  OUT_OF_RANGE = 11;
  UNIMPLEMENTED = 12;
  INTERNAL = 13;
  UNAVAILABLE = 14;
  DATA_LOSS = 15;
  DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 20;
}


## CPU 信息

获取 CPU 信息的方式非常流弊，不是读 /proc/cpuinfo 文件而是直接通过汇编指令获取。
详细参考 tensorflow/core/platform/cpu_info.cc

  asm("mov %%rbx, %%rdi\n"                 \
      "cpuid\n"                            \
      "xchg %%rdi, %%rbx\n"                \
      : "=a"(a), "=D"(b), "=c"(c), "=d"(d) \
      : "a"(a_inp), "2"(c_inp))


## 线程池

tensorflow/core/lib/core/threadpool.h
tensorflow/core/lib/core/threadpool_test.cc : 测试代码例子很易读

Eigen::ThreadPoolTempl


ThreadPool : 由 ThreadPool:Impl 实现
  std::unique_ptr<Impl> impl_;

ThreadPool:Impl : 继承了 Eigen::ThreadPoolTempl<EigenEnvironment>
    Env
    ThreadOptions
    name
    num_threads
    low_latency_hint

Eigen::ThreadPoolTempl<EigenEnvironment> 核心实现，EigenEnvironment
为运行时环境，通过模板可以自定义运行环境.
详细参考 third_party/eigen3/unsupported/Eigen/CXX11/

Eigen::ThreadPoolDevice

EigenEnvironment
  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

Env : 用于获取系统的文件系统及其他信息
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  EnvTime* envTime = EnvTime::Default();

EnvWrapper  : 代理 Env, 方便替换 Env
  Env* target_;

ThreadOptions : 线程选项
  size_t stack_size = 0;  // 0: use system default value
  size_t guard_size = 0;  // 0: use system default value

message ThreadPoolOptionProto {
  int32 num_threads = 1; 线程池大小，0 意味着由系统决定
  string global_name = 2; //线程池的名称
};


### 通知

Notification
  mutex mu_; //多线程的锁
  condition_variable cv_; //条件变量，参见标准库
  bool notified_; // 是否已经调用

* WaitForNotificationWithTimeout(int64 timeout_in_us)
* HasBeenNotified()
* WaitForNotification()
* Notify()

通过 condition_variable 实现简单的锁，n 个线程调用
WaitForNotificationWithTimeout() 或 WaitForNotification() 之后
程序阻塞，直到超时或其他线程调用 Notify(), 才继续执行

注：Java 的 Object 类库级别支持类似语义

## 内存管理

Arena
  struct AllocatedBlock {
    char* mem;
    size_t size;
  };

  size_t remaining_;
  const size_t block_size_;
  char* freestart_;  // beginning of the free space in most recent block
  char* freestart_when_empty_;  // beginning of the free space when we're empty
  // STL vector isn't as efficient as it could be, so we use an array at first
  size_t blocks_alloced_;  // how many of the first_blocks_ have been alloced
  AllocatedBlock first_blocks_[16];  // the length of this array is arbitrary
  // if the first_blocks_ aren't enough, expand into overflow_blocks_.
  std::vector<AllocatedBlock>* overflow_blocks_;

## StringPiece

  const char* data_; //字符串首指针
  size_t size_;  //字符串长度


## Scanner


## Sycl

OpenCL 进行显卡编程

需要 设置 TENSORFLOW_USE_SYCL

## XLA


## CPP 语法

```
  char *ptr
  NodeItem* item = reinterpret_cast<NodeItem*>(ptr);
  new (item) NodeItem();
```



\__COUNTER__ 实质上是一个int，并且是具体的数，初值是0，每预编译一次其值自己加1

例子

REGISTER_KERNEL_BUILDER(Name("Variable")
REGISTER_SYSTEM_KERNEL_BUILDER(Name("_Arg").Device(DEVICE_CPU), ArgOp)

REGISTER_KERNEL_BUILDER(kernel_builder, ...)

声明 tensorflow::kernel_factory::OpKernelRegistrar registrar__body____COUNTER____object() 的函数

```
static ::tensorflow::kernel_factory::OpKernelRegistrar              \
    constexpr bool should_register___COUNTER____flag = SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__);
    registrar__body____COUNTER____object
        should_register___COUNTER____flag \
            ? ::tensorflow::register_kernel::kernel_builder.Build() \
            : nullptr,                                              \
            #__VA_ARGS__,                                               \
            [](::tensorflow::OpKernelConstruction* context)             \
                -> ::tensorflow::OpKernel* {                            \
              return new __VA_ARGS__(context);                          \
            });
```


REGISTER_SYSTEM_KERNEL_BUILDER(kernel_builder, ...)

声明 tensorflow::kernel_factory::OpKernelRegistrar registrar__body____COUNTER____object() 的函数

```
static ::tensorflow::kernel_factory::OpKernelRegistrar              \
    registrar__body____COUNTER____object
            ::tensorflow::register_kernel::system::kernel_builder.Build() \
            #__VA_ARGS__,                                               \
            [](::tensorflow::OpKernelConstruction* context)             \
                -> ::tensorflow::OpKernel* {                            \
              return new __VA_ARGS__(context);                          \
            });
```


// On some platforms, we would like to avoid using RTTI in order to have smaller
// binary sizes. The following #ifdef section provides a non-RTTI
// replacement for std::type_index (with a minimal set of functions needed by
// the TensorFlow framework, and more can be added if necessary).
```cpp
#if !defined(__GXX_RTTI) && !defined(_CPPRTTI)
class TypeIndex {
 public:
  TypeIndex(const TypeIndex& src) : hash_(src.hash_) {}
  TypeIndex& operator=(const TypeIndex& src) {
    hash_ = src.hash_;
    return *this;
  }
  bool operator==(const TypeIndex& rhs) const { return (hash_ == rhs.hash_); }
  bool operator!=(const TypeIndex& rhs) const { return (hash_ != rhs.hash_); }
  ~TypeIndex() {}

  const char* name() const { return "[RTTI disabled for Android]"; }
  uint64 hash_code() const { return hash_; }

  // Returns a TypeIndex object that corresponds to a typename.
  template <typename T>
  static TypeIndex Make() {
    static bool hash_bit[1];
    return TypeIndex(static_cast<uint64>(reinterpret_cast<intptr_t>(hash_bit)));
  }

 private:
  // We hide the constructor of the TypeIndex class. Use the templated
  // Make<T>() function to create a TypeIndex object.
  TypeIndex(const uint64 hash) : hash_(hash) {}
  uint64 hash_;
};

template <typename T>
inline TypeIndex MakeTypeIndex() {
  return TypeIndex::Make<T>();
}
#else  // __GXX_RTTI
typedef std::type_index TypeIndex;
template <typename T>
inline TypeIndex MakeTypeIndex() {
  return TypeIndex(typeid(T));
}
#endif  // __GXX_RTTI
```

#### 模板

例1
```cpp
// Functions to define quantization attribute of types.
struct true_type {
  static const bool value = true;
};
struct false_type {
  static const bool value = false;
};

// Default is_quantized is false.
template <typename T>
struct is_quantized : false_type {};

// Specialize the quantized types.
template <>
struct is_quantized<qint8> : true_type {};
template <>
struct is_quantized<quint8> : true_type {};
template <>
struct is_quantized<qint32> : true_type {};
template <>
struct is_quantized<qint16> : true_type {};
template <>
struct is_quantized<quint16> : true_type {};
```
默认都是 false_type, 对于某些类型为 true_type

例2

```cpp
template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value,
          bool = std::is_same<typename std::decay<T>::type,
                              ::tensorflow::Tensor>::value,
          bool = std::is_base_of<protobuf::MessageLite,
                                 typename std::decay<T>::type>::value>
struct TypeResolver {};
template <typename T>
void EncodeVariantImpl(const T& value, TypeResolver<T, true /* is_pod */>,
                       VariantTensorData* data);
```
通过 TypeResolver 对 T 进行类型标记

例3
``` cpp
    Buffer<T>* buf = new Buffer<T>(a, n);
    char* data = buf->template base<char>();
```
模板实例化

## 术语


Node :
Edege :
Operation :
Device : CPU 或 GPU, SYCLC
ResuoureHandler : 隶属于一个 Container，在一个 Device 上执行
Container :

