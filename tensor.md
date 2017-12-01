
## Tensor

A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.Session.

1. A Tensor can be passed as an input to another Operation. This builds a dataflow connection between operations, which enables TensorFlow to execute an entire Graph that represents a large, multi-step computation.

2. After the graph has been launched in a session, the value of the Tensor can be computed by passing it to tf.Session.run. t.eval() is a shortcut for calling tf.get_default_session().run(t).

参数是 Operation, Operation endpoint 的 value_index, 类型三要素

一个 Tensor 可以有多个 consumer, 每个 consumer 是 Operation, 此时该 Tensor 属于
Operation 的输入

一个 Tensor 必须属于一个 Operation，作为 Operation 的输入或输出

Tensor 必须是有类型的

Tensor 的 shape 可以通过 set_shape 被修改

可以用 tf.convert_to_tensor, tf.internal_convert_n_to_tensor 将 python list, numpy arrays, python scalars, Tensor 转换为 Tensor

```
d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
print(d.shape)

    TensorShape([Dimension(4), Dimension(2)])
```

关键属性

op : 所属的 Operation
shape :
value_index : 在 Operation 中的索引
dtype : 类型
shape :
consumers : 为所属 Operation
handle_data :

关键函数

op(self)
dtype(self)
graph(self)
id(self)
name(self)
set_shape(self, shape): 设置 shape
get_shape(self)
shape(self)
device(self)
value_index(self):
consumers(self):
add_consumer(self, consumer):
as_tf_output(self)
eval(self, feed_dict=None, session=None): TODO

scalar :  表示点，维度为 0
vector :  表示线，维度为 1
matrix :  表示面，维度为 2


## 数据结构

TensorProto <--> Tensor

Tensor -> TensorDescription

TensorBuffer ->  Tensor
TensorShape

TensorShapeProto -> TensorShape

TensorShape <--> PartialTensorShape

class TensorBuffer

class BufferBase : public TensorBuffer
  Allocator* const alloc_;

class Buffer : public BufferBase
  T* data_;
  int64 elem_;

class SubBuffer : public TensorBuffer  //与 root Buffer 共享 buf
  TensorBuffer* root_; //
  T* data_;     //保存数据的首指针
  int64 elem_;  //元素个数

class TensorReference //构造和移动都非常  cheap
  TensorBuffer* buf_;

message TensorSliceProto
  message Extent
    int64 start = 1;
    oneof has_length
      int64 length = 2;
  repeated Extent extent = 1;

class TensorSlice //表达 Tensor 在某一个维度的范围
  static const int64 kFullExtent;
  gtl::InlinedVector<int64, 4> starts_;  //开始的索引
  gtl::InlinedVector<int64, 4> lengths_; //长度, 如果值为 -1 表示所有元素

message TensorShapeProto
  message Dim
    int64 size = 1; //该维的元素个数， -1 表示  unknow, 在反序列化的时候会失败
    string name = 2;
  repeated Dim dim = 2; //第一个是  outermost dim, 最后一个是  inner dim
  bool unknown_rank = 3; // 如果为 ture, dim.size 必须为 0

class TensorShapeRep
  union {
    uint8 buf[16]; //buf[13] 存储 data_type, buf[14] 存储维度，buf[15] 存储 tag
    // Force data to be aligned enough for a pointer.
    Rep64* unused_aligner;
  } u_;
  int64 num_elements_; //默认 -1

tag 默认 REP16, data_type 默认 DT_INVALID, ndims_byte 默认 kUnknownRank

class TensorShapeBase : public TensorShapeRep
class TensorShape : public TensorShapeBase<TensorShape>
class PartialTensorShape : public TensorShapeBase<PartialTensorShape>

PartialTensorShape 与  TensorShape 的区别在于 PartialTensorShape 的某一个维度的 size 允许为 0

struct TensorShapeDim
  int64 size;

class TensorShapeUtils

message TensorDescription
  DataType dtype = 1;
  TensorShapeProto shape = 2;
  AllocationDescription allocation_description = 4;

message TensorProto {
  DataType dtype = 1;
  TensorShapeProto tensor_shape = 2;
  int32 version_number = 3;
  bytes tensor_content = 4; //
  //以_val 结尾只有一个有实际的用处，不用 oneof 是因为需要 repeated
  repeated int32 half_val = 13 [packed = true];
  repeated float float_val = 5 [packed = true];
  repeated double double_val = 6 [packed = true];
  repeated int32 int_val = 7 [packed = true];
  repeated bytes string_val = 8;
  repeated float scomplex_val = 9 [packed = true];
  repeated int64 int64_val = 10 [packed = true];
  repeated bool bool_val = 11 [packed = true];
  repeated double dcomplex_val = 12 [packed = true];
  repeated ResourceHandleProto resource_handle_val = 14;
  repeated VariantTensorDataProto variant_val = 15;
};

message VariantTensorDataProto
  string type_name = 1;
  bytes metadata = 2;
  repeated TensorProto tensors = 3;

class Tensor
  TensorShape shape_;
  TensorBuffer* buf_; //shape_.data_type() 类型的 Buffer






## 实例

### TensorShape

```cpp
  TensorShape s;
  EXPECT_EQ(s.dims(), 0);
  EXPECT_EQ(s.num_elements(), 1);

  TensorShape s({10, 5});

  s.set_dim(0, 20);
  ASSERT_EQ(2, s.dims());
  EXPECT_EQ(20, s.dim_size(0));
  EXPECT_EQ(100, s.num_elements());

  s.set_dim(1, 2);
  ASSERT_EQ(2, s.dims());
  EXPECT_EQ(2, s.dim_size(1));
  EXPECT_EQ(40, s.num_elements());

  TensorShape s({10, 5});
  s.RemoveDim(0);
  EXPECT_EQ(5, s.num_elements());
  ASSERT_EQ(1, s.dims());

  TensorShape s({10, 5, 20});
  EXPECT_EQ(1000, s.num_elements());
  s.set_dim(1, 0);
  EXPECT_EQ(0, s.num_elements());
  s.set_dim(1, 7);
  EXPECT_EQ(1400, s.num_elements());

  TensorShape s({10, 2147483648});
  TensorShape s2;
  s2.AppendShape(s);
  EXPECT_EQ(10, s2.dim_size(0));
  EXPECT_EQ(2147483648, s2.dim_size(1));

  TensorShape s({});
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_INVALID);
  TensorShapeTestHelper::set_data_type(&s, DT_INT32);
  s.AddDim(1);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_INT32);
  s.AddDim(100000);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_INT32);
  TensorShapeTestHelper::set_data_type(&s, DT_UINT16_REF);
  s.AddDim(2);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_UINT16_REF);
  s.AddDim(4);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_UINT16_REF);
  s.AddDim(3);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s), DT_UINT16_REF);
  TensorShapeTestHelper::set_data_type(&s2, DT_FLOAT);
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s2), DT_FLOAT);
  s2.Clear();
  EXPECT_EQ(TensorShapeTestHelper::data_type(&s2), DT_INVALID);

  EXPECT_TRUE(
      TensorShapeUtils::StartsWith(TensorShape({2, 3}), TensorShape({2, 3})));
  EXPECT_TRUE(TensorShapeUtils::StartsWith(TensorShape({2, 3, 4}),
                                           TensorShape({2, 3})));
  EXPECT_TRUE(
      TensorShapeUtils::EndsWith(TensorShape({2, 3}), TensorShape({2, 3})));
  EXPECT_TRUE(
      TensorShapeUtils::EndsWith(TensorShape({2, 3, 4}), TensorShape({3, 4})));
```

### TensorSlice

```cpp
  TensorSlice s(3);
  EXPECT_EQ("-:-:-", s.DebugString());
  EXPECT_TRUE(s.IsFull());

  TensorSlice s({{0, -1}, {0, 10}, {14, 1}, {0, -1}});
  EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
  EXPECT_TRUE(!s.IsFull());

  TensorSliceProto proto;
  const char* ptxt = R"PROTO(
    extent { }
    extent { start: 0 length: 10 }
    extent { start: 14 length: 1 }
    extent { }
  )PROTO";
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(ptxt, &proto));
  TensorSlice s(proto);
  EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
  EXPECT_TRUE(!s.IsFull());

  TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
  TensorSliceProto proto;
  s.AsProto(&proto);
  EXPECT_EQ(
      "extent { } "
      "extent { } "
      "extent { start: 1 length: 3 } "
      "extent { start: 4 length: 5 }",
      proto.ShortDebugString());
  EXPECT_TRUE(!s.IsFull());

  TensorSlice a = TensorSlice::ParseOrDie("-:-");
  TensorSlice b = TensorSlice::ParseOrDie("1,2:3,4");
  TensorSlice c;
  EXPECT_TRUE(a.Intersect(b, &c));
  EXPECT_EQ("1,2:3,4", c.DebugString());

  TensorSlice a = TensorSlice::ParseOrDie("1,5:2,6:3,7:5,10");
  TensorSlice b = TensorSlice::ParseOrDie("1,2:3,4:9,10:12,1");
  TensorSlice c;
  EXPECT_TRUE(a.Intersect(b, &c));
  EXPECT_EQ("1,2:3,4:9,1:12,1", c.DebugString());

  TensorSlice a = TensorSlice::ParseOrDie("1,1:-:4,1:2,6");
  TensorShape x({2, 4, 5, 8});
  TensorShape y;
  TF_EXPECT_OK(a.SliceTensorShape(x, &y));

  TensorSlice base = TensorSlice::ParseOrDie("-:-:-:-");
  TensorSlice sub = TensorSlice::ParseOrDie("-:1,2:-:3,4");
  TensorSlice relative;
  base.ComputeRelative(sub, &relative);
  EXPECT_EQ("-:1,2:-:3,4", relative.DebugString());

  TensorSlice base = TensorSlice::ParseOrDie("1,2:3,4:-:5,1");
  TensorSlice sub = TensorSlice::ParseOrDie("1,1:4,2:3,3:5,1");
  TensorSlice relative;
  base.ComputeRelative(sub, &relative);
  EXPECT_EQ("0,1:1,2:3,3:0,1", relative.DebugString());

  TensorSliceProto proto;
  const char* ptxt = R"PROTO(
    extent { }
    extent { start: 0 length: 10 }
    extent { start: 14 length: 1 }
    extent { }
  )PROTO";
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(ptxt, &proto));
  EXPECT_FALSE(TensorSlice::HasExtentLength(proto.extent(0)));
  EXPECT_TRUE(TensorSlice::HasExtentLength(proto.extent(1)));
  EXPECT_TRUE(TensorSlice::HasExtentLength(proto.extent(2)));
  EXPECT_FALSE(TensorSlice::HasExtentLength(proto.extent(3)));
  EXPECT_EQ(-1, TensorSlice::GetExtentLength(proto.extent(0)));
  EXPECT_EQ(10, TensorSlice::GetExtentLength(proto.extent(1)));
  EXPECT_EQ(1, TensorSlice::GetExtentLength(proto.extent(2)));
  EXPECT_EQ(-1, TensorSlice::GetExtentLength(proto.extent(3)));

  // [2:4, :, 3:]
  TensorSlice s({{2, 2}, {0, -1}, {3, 7}});
  // [:, 1:4, 2:4]
  TensorSlice other({{0, -1}, {1, 3}, {2, 2}});

  s.UpdateToCover(other);
  // [:, :, 2:]
  EXPECT_EQ("-:-:2,8", s.DebugString());

  TensorSlice slice(3);
  EXPECT_TRUE(slice.IsFull());

  TensorSlice slice2({{0, -1}});
  EXPECT_TRUE(slice2.IsFull());

  TensorSlice slice3({{0, -1}, {0, -1}, {14, 1}});
  EXPECT_TRUE(!slice3.IsFull());

  TensorSlice slice1(3);
  TensorSlice slice2({{0, -1}, {0, -1}, {0, -1}});
  EXPECT_TRUE(slice1 == slice2);
  EXPECT_TRUE(slice2 == slice1);

  Tensor x(DT_FLOAT, TensorShape({}));
  x.scalar<float>()() = 10.0;
  // Make y a deep copy of x and then change it.
  Tensor y = tensor::DeepCopy(x);
  y.scalar<float>()() = 20.0;
  // x doesn't change
  EXPECT_EQ(10.0, x.scalar<float>()());
  // Change x.
  x.scalar<float>()() = 30.0;
  // Y doesn't change.
  EXPECT_EQ(20.0, y.scalar<float>()());

  Tensor str1(DT_STRING, TensorShape({2}));
  str1.flat<string>()(0) = "foo1";
  str1.flat<string>()(1) = "foo2";
  Tensor str2 = tensor::DeepCopy(str1);
  str2.flat<string>()(0) = "bar1";
  str2.flat<string>()(1) = "bar2";
  EXPECT_NE(str2.flat<string>()(0), str1.flat<string>()(0));

  Tensor x(DT_INT32, TensorShape({10}));
  x.flat<int32>().setConstant(1);
  // Slice 'x' -- y still refers to the same buffer.
  Tensor y = x.Slice(2, 6);
  // Do a deep copy of y, which is a slice.
  Tensor z = tensor::DeepCopy(y);
  // Set x to be different.
  x.flat<int32>().setConstant(2);
  EXPECT_EQ(TensorShape({10}), x.shape());
  EXPECT_EQ(TensorShape({4}), y.shape());
  EXPECT_EQ(TensorShape({4}), z.shape());
  EXPECT_EQ(DT_INT32, x.dtype());
  EXPECT_EQ(DT_INT32, y.dtype());
  EXPECT_EQ(DT_INT32, z.dtype());
  // x and y should now all be '2', but z should be '1'.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(2, x.flat<int32>()(i));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(2, y.unaligned_flat<int32>()(i));
    EXPECT_EQ(1, z.flat<int32>()(i));
  }

  Tensor x(DT_STRING, TensorShape({4, 3}));
  for (int i = 0; i < 4 * 3; ++i) {
    x.flat<string>()(i) = strings::StrCat("foo_", i);
  }

  std::vector<Tensor> split;
  TF_ASSERT_OK(tensor::Split(x, {2, 1, 1}, &split));
  Tensor x_round_tripped;
  TF_ASSERT_OK(tensor::Concat(split, &x_round_tripped));
  ASSERT_EQ(x.shape(), x_round_tripped.shape());
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_EQ(x.flat<string>()(i), x_round_tripped.flat<string>()(i));
  }

  // Ensure that no memory is being shared between 'x' and 'x_round_tripped'.
  for (int i = 0; i < 4 * 3; ++i) {
    x_round_tripped.flat<string>()(i) = strings::StrCat("bar_", i);
  }
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_NE(x.flat<string>()(i), x_round_tripped.flat<string>()(i));
  }


```







### placeholder

初始化只需要 type 和 shape, 具体的值可以在 `Session.run()`, `Tensor.eval()`, 或 `Operation.run()`.
中通过 feed_dict 中初始化

  ```python
  x = tf.placeholder(tf.float32, shape=(1024, 1024))
  y = tf.matmul(x, x)

  with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
  ```

### constant

tensorflow/python/framework/constant_op.py



### 基本类型转为 Tensor


\_tensor_conversion_func_registry = { 0: [(Tensor, \_TensorTensorConversionFunction)]} register_dense_tensor_like_type(Tensor)

\_tensor_conversion_func_registry 的 key 是优先级，value 是一个 list, list
的每个元素是 (base_type, conversion_func) 的 tuple

99: [ (list, tuple), \_autopacking_conversion_function)]

100 : [ ((list, tuple), \_constant_tensor_conversion_function),
(np.ndarray, \_constant_tensor_conversion_function)
(np.generic, \_constant_tensor_conversion_function)
(object, \_constant_tensor_conversion_function)
(tensor_shape.TensorShape, \_tensor_shape_tensor_conversion_function)
(tensor_shape.Dimension, \_dimension_tensor_conversion_function)
(BaseStochasticTensor, BaseStochasticTensor._tensor_conversion_function)
(LabeledTensor, \_convert_labeled_tensor_to_tensor)
(Operation, \_operation_conversion_error)
(IndexedSlices, \_IndexedSlicesToTensor)
(ResourceVariable, \_dense_var_to_tensor)
(variables.Variable, variables.Variable._TensorConversionFunction)
(PartitionedVariable, PartitionedVariable._TensorConversionFunction)
]

### 初始化

tensorflow/python/ops/init_ops.py

zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
uniform_unit_scaling_initializer = UniformUnitScaling
variance_scaling_initializer = VarianceScaling
orthogonal_initializer = Orthogonal


  ```python
    >>> import numpy as np
    >>> import tensorflow as tf

    >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> # value = np.array(value)
    >>> # value = value.reshape([2, 4])
    >>> init = tf.constant_initializer(value)

    >>> print('fitting shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8

    >>> print('shape verification:')
    >>> init_verify = tf.constant_initializer(value, verify_shape=True)
    >>> with tf.Session():
    >>>   x = tf.get_variable('x', shape=[3, 4], initializer=init_verify)

    TypeError: Expected Tensor's shape: (3, 4), got (8,).
  ```
## TensorShape


## TensorShapeRep

最大维度 254

data_type : 数据类型(buf[13])
ndims_byte : 维度(buf[14])
tag : tag(buf[15])  REP16 = 0, REP32 = 1, REP_OUT_OF_LINE = 2

REP16 : 6 维，每维最大 2^16 - 1
REP32 : 3 维，每维最大 2^32 - 1
REP64 : 任意维，out of line vector


union {
  uint8 buf[16]; //倒数第二个元素为 维度, 最后一个元素为 tag
  // Force data to be aligned enough for a pointer.
  Rep64* unused_aligner;
} u_;
int64 num_elements_;

scalar : 标量
vector : 一维向量
matrix : 二维矩阵
squareMatrix : 平方矩阵, x 与 y 的数量相同

## TensorShapeUtils



## PersistentTensor

class PersistentTensor
  Tensor tensor_;


Tensor* PersistentTensor::AccessTensor(OpKernelContext* context)
  context->NotifyUseOfPersistentTensor(tensor_);
  return &tensor_;

## 源码解析


* Tensor();
* Tensor(DataType type, const TensorShape& shape);
* Tensor(Allocator* a, DataType type, const TensorShape& shape);
* Tensor(Allocator* a, DataType type, const TensorShape& shape, const AllocationAttributes& allocation_attr);
* explicit Tensor(DataType type);
* Tensor(const Tensor& other);
* Tensor(Tensor&& other);
* DataType dtype() // return shape_.data_type();
* TensorShape& shape() //return shape_;
* int dims() //return shape().dims(); }
* int64 dim_size(int d) // return shape().dim_size(d);
* int64 NumElements() //return shape().num_elements();
* bool IsSameSize(const Tensor& b) //
* bool IsInitialized() // return (buf_ != nullptr && buf_->data() != nullptr) || shape_.num_elements() == 0;
* size_t Tensor::TotalBytes() //tensor 总共占用的内存
* size_t Tensor::AllocatedBytes() // 如果 buf_ 不为空，用  buf_ 初始化 tensor_description,  返回 tensor_description.allocation_description().allocated_bytes(); 否则，返回 TotalBytes()
* bool IsAligned() // return reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0;
* bool Tensor::CanUseDMA() //如果是已经规定的类型，就返回 ture
* bool CopyFrom(const Tensor& other, const TensorShape& shape)
* Tensor Slice(int64 dim0_start, int64 dim0_limit) //返回维度从 dim0_start 到 dim0_limit 直接的部分，注意是引用
* bool Tensor::FromProto(const TensorProto& proto) //用 proto 初始化 Tensor

bool Tensor::FromProto(Allocator* a, const TensorProto& proto)

用 proto 初始化 Tensor
1. buf_ : proto.tensor_content 不为空，从 proto.tensor_content 解码, 否则从 proto 本身解码
2. shape_ : proto.tensor_shape()
3. dtype : proto.dtype()

void Tensor::AsProtoField(TensorProto* proto)

将 tensor 转为 TensorProto
1. proto.mutable_tensor_shape() 为 shape_
2. buf_ 编码为 proto.tensor_shape()


* TTypes<T>::Vec vec() //return tensor<T, 1>();
* TTypes<T>::Matrix matrix() //return tensor<T, 2>();
* TTypes<T, NDIMS>::Tensor tensor();
* TTypes<T, NDIMS>::Tensor bit_casted_tensor();
* TTypes<T>::Flat flat()
* TTypes<T>::UnalignedFlat unaligned_flat()
* TTypes<T, NDIMS>::Tensor flat_inner_dims();
* TTypes<T, NDIMS>::Tensor flat_outer_dims();
* TTypes<T, NDIMS>::Tensor flat_inner_outer_dims(int64 begin);
* TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64> new_sizes);
* TTypes<T, NDIMS>::Tensor bit_casted_shaped(gtl::ArraySlice<int64> new_sizes);
* TTypes<T, NDIMS>::UnalignedTensor unaligned_shaped(gtl::ArraySlice<int64> new_sizes);
* TTypes<T>::Scalar scalar();
* TTypes<T>::ConstVec vec() // return tensor<T, 1>();
* TTypes<T>::ConstMatrix matrix() // return tensor<T, 2>();
* TTypes<T, NDIMS>::ConstTensor tensor()
* TTypes<T, NDIMS>::ConstTensor bit_casted_tensor()
* TTypes<T>::ConstFlat flat() //return shaped<T, 1>({NumElements()});
* TTypes<T>::UnalignedConstFlat unaligned_flat() //return unaligned_shaped<T, 1>({NumElements()});
* TTypes<T, NDIMS>::ConstTensor shaped(gtl::ArraySlice<int64> new_sizes)
* TTypes<T, NDIMS>::ConstTensor bit_casted_shaped(gtl::ArraySlice<int64> new_sizes)
* TTypes<T, NDIMS>::UnalignedConstTensor unaligned_shaped(gtl::ArraySlice<int64> new_sizes)
* TTypes<T>::ConstScalar scalar() ;
* TTypes<T, NDIMS>::ConstTensor flat_inner_dims()
* TTypes<T, NDIMS>::ConstTensor flat_outer_dims()
* TTypes<T, NDIMS>::ConstTensor flat_inner_outer_dims(int64 begin)
* string Tensor::SummarizeValue(int64 max_entries) //将  tensor 最多 max_entries 个元素转换为字符串
* string Tensor::DebugString() //转换为 string
* void Tensor::FillDescription(TensorDescription* description) //用 Tensor 初始化 TensorDescription
* StringPiece Tensor::tensor_data() //将  buf_ 转为为 string
* bool Tensor::SharesBufferWith(const Tensor& b) //return buf_->root_buffer() == b.buf_->root_buffer();
* bool RefCountIsOne() //return buf_ != nullptr && buf_->RefCountIsOne() && buf_->root_buffer()->RefCountIsOne() && buf_->OwnsMemory();
* void CheckType(DataType expected_dtype) // CHECK_EQ(dtype(), expected_dtype)
* void CheckTypeAndIsAligned(DataType expected_dtype) //CHECK_EQ(dtype(), expected_dtype); CHECK(IsAligned());
* void CheckIsAlignedAndSingleElement() // CHECK(IsAligned()); CHECK_EQ(1, NumElements())
* void set_dtype(DataType t) //shape_.set_data_type(t);
* void FillDimsAndValidateCompatibleShape(gtl::ArraySlice<int64> new_sizes, Eigen::array<Eigen::DenseIndex, NDIMS>* dims) //new_size 填充到 dims

* Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);
* void set_shape(const TensorShape& shape)
* T* base() //buf_ == nullptr ? nullptr : buf_->base<T>();
* void FillDimsAndValidateCompatibleShape(Eigen::array<Eigen::DenseIndex, NDIMS>* dims, gtl::ArraySlice<int64> new_sizes) //new_size 填充到 dims
* void CopyFromInternal(const Tensor& other, const TensorShape& shape); //用 other  重置 this
* void UnsafeCopyFromInternal(const Tensor&, DataType dtype, const TensorShape&) // 用  other 重置 this

static gtl::InlinedVector<int64, 4> ComputeFlatOuterDims(gtl::ArraySlice<int64> orig, int64 num_out_dims);

将 org 的前 num_out_dims 个元素返回,  最后一个元素是剩余 orin 除去前 num_out_dims 个元素的乘积

static gtl::InlinedVector<int64, 4> ComputeFlatInnerDims(gtl::ArraySlice<int64> orig, int64 num_out_dims);

将 org 的后 num_out_dims 个元素返回, 第一个元素是剩余 orin 除去前  num_out_dims 个元素的乘积



Eigen::DSizes<Eigen::DenseIndex, NDIMS> TensorShape::AsEigenDSizesWithPadding()

复制 TensorShape, 将 NDIMS > dims() 的部分元素个数 1

TensorShapeBase<Shape>::TensorShapeBase(const TensorShapeProto& proto) //proto.dim() 作为TensorShapeBase 的 dim
TensorShapeBase<Shape>::TensorShapeBase(gtl::ArraySlice<int64> dim_sizes) //dim_sizes 作为TensorShapeBase 的 dim
TensorShapeBase<Shape>::TensorShapeBase() //默认值
void TensorShapeRep::SlowCopyFrom(const TensorShapeRep& b) //b 重置 this
int64 TensorShapeBase<Shape>::dim_size(int d) //根据 tag 返回对应的第 d 维度的大小
void TensorShapeBase<Shape>::RecomputeNumElements() //重新计算 num_elements 大小
void TensorShapeBase<Shape>::AddDim(int64 size) //增加一维，该维的元素个数为 size
void TensorShapeBase<Shape>::AppendShape(const TensorShapeBase& shape) //依次遍历  shape 的每一维增加到 this 后面
void TensorShapeBase<Shape>::InsertDim(int d, int64 size) //将 size 插入第 d 维之前
gtl::InlinedVector<int64, 4> TensorShapeBase<Shape>::dim_sizes() //返回每一维的大小的数组
void TensorShapeBase<Shape>::set_dim(int d, int64 size) //设置第 d 维的元素为 size
void TensorShapeBase<Shape>::RemoveDim(int d)  //删除第 d 维
bool TensorShape::IsSameSize(const TensorShape& b) //只有维度相同，每个维度的元素个数相同才是返回 true
void TensorShapeBase<Shape>::AsProto(TensorShapeProto* proto) //转换为 TensorShapeProto
TensorShapeIter<Shape> TensorShapeBase<Shape>::begin() // return TensorShapeIter<Shape>(static_cast<const Shape*>(this), 0);
TensorShapeIter<Shape> TensorShapeBase<Shape>::end() // return TensorShapeIter<Shape>(static_cast<const Shape*>(this), dims());
string TensorShapeRep::DebugString() // 格式如 "[2, 3, ?]"
bool TensorShapeUtils::StartsWith(const TensorShape& shape, const TensorShape& prefix) //shape 维度大于等于 prefix, 并且前  prefix.dims() 每维的元素个数与 prefix 完全相同
bool TensorShapeUtils::EndsWith(const TensorShape& shape, const TensorShape& suffix) //shape 维度大于等于 prefix, 并且最后 prefix.dims() 每维的元素个数与 prefix 完全相同
Status MakeShapeHelper(const T* dims, int64 n, Shape* out) //将 dims 中前 n 维加入 out
string TensorShapeUtils::ShapeListString( const gtl::ArraySlice<TensorShape>& shapes) //shapes 转为字符串，如 "[1, 2, ?]"

PartialTensorShape PartialTensorShape::Concatenate(int64 size) //增加一维，大小为 size
PartialTensorShape PartialTensorShape::Concatenate(const PartialTensorShape& shape) // 创建 PartialTensorShape, 将 shape 每维加入之
Status PartialTensorShape::MergeWith(const PartialTensorShape& shape, PartialTensorShape* result) // 将 dims_ 与shape.dims() 合并到 result, 注：前提是两个 dims 大小相同，并且每个维度的元素个数要不相同， 或者 一个不为 unknow 另一个为 unknow
bool PartialTensorShape::AsTensorShape(TensorShape* shape) //进行类型强制转换  注：前提是 num_elements_ 不为 -1
bool PartialTensorShape::IsIdenticalTo(const PartialTensorShape& shape) //只有维度相同，每个维度的元素个数相同才是返回 true
bool PartialTensorShape::IsCompatibleWith(const PartialTensorShape& shape) //在维度相同，每个维度有一个为 -1， 或者两个都不为 0 且相同
string PartialTensorShapeUtils::PartialShapeListString(const gtl::ArraySlice<PartialTensorShape>& shapes) //shapes 转为为字符串，如 "[1, 2, ?]"
bool PartialTensorShapeUtils::AreCompatible(gtl::ArraySlice<PartialTensorShape>& shapes0, gtl::ArraySlice<PartialTensorShape>& shapes1) //元素个数相同，且每个元素相互兼容
bool PartialTensorShapeUtils::AreIdentical(gtl::ArraySlice<PartialTensorShape>& shapes0, gtl::ArraySlice<PartialTensorShape>& shapes1) // 维度和每个维度的元素都相同
Status TensorShapeUtils::NumElements(gtl::ArraySlice<int64> shape, int64* num_elements) //shape 的元素个数保存在  num_elements




void Extend(int dim);
void AsProto(TensorSliceProto* proto) const;

void FillIndicesAndSizes(TensorShape& shape, Eigen::DSizes<Eigen::DenseIndex, NDIMS>* indices, Eigen::DSizes<Eigen::DenseIndex, NDIMS>* sizes)
遍历所有的的维度，如果 维度 d 是 full, length 设置 shape.dim_size(d) 对应的长度，如果没有，就保留 lengths_[d]

TensorSlice::TensorSlice(int dim) //确保有 dim 个维度，每个维度都是全部元素
TensorSlice::TensorSlice(std::initializer_list<std::pair<int64, int64>> extents)  //用 extents 初始化  starts_ 和  lengths_
void TensorSlice::SetFullSlice(int dim) //确保有  dim 个维度，每个维度都是所有元素
void TensorSlice::Extend(int dim) //扩展当前的维度到  dim, 新增的维度都包含所有元素 用字符串即为"-"
void TensorSlice::AsProto(TensorSliceProto* proto) //转为  TensorSliceProto
string TensorSlice::DebugString() // 转为形如 "1,2:-:3:" 的字符串
Status TensorSlice::Parse(const string& str, TensorSlice* slice) //解析 str 初始化 slice, str 形如 "1,2:-:3:" 等
void TensorSlice::Clear() //清空 starts_, lengths_
bool TensorSlice::IsFull() //只有所有维度都是 full 的时候才返回 true
bool TensorSlice::Intersect(const TensorSlice& other, TensorSlice* result) //求 this 和  other 的交集，保存在 result
void TensorSlice::ComputeRelative(const TensorSlice& sub, TensorSlice* relative) //start = starts_[d] - sub.start(d); length = sub.length(d)
void TensorSlice::UpdateToCover(const TensorSlice& other) //求  this 和  other 并集重置 this
Status TensorSlice::SliceTensorShape(const TensorShape& shape, TensorShape* result_shape) //TODO
static bool TensorSlice::HasExtentLength(const TensorSliceProto::Extent& extent) //return extent.has_length_case() == TensorSliceProto::Extent::kLength;
static int64 TensorSlice::GetExtentLength(const TensorSliceProto::Extent& extent) //rerurn extent.length();


Tensor DeepCopy(const Tensor& other)

用 other 初始化 tmp  的数据成员 buf_ shape，之后返回 tmp。

Status Concat(const gtl::ArraySlice<Tensor>& tensors, Tensor* result)

将 tensors 中的多个  Tensor 合并成一个 Tensor
前提是每个 tensor 的维度都不能为 0，每个 tensor 的类型必须相同

Status Split(const Tensor& tensor, const gtl::ArraySlice<int64>& sizes, std::vector<Tensor>* result)

将  tensor 分成 n 个  Tensor, 每个 Tensor 的维度保存在 sizes 中，分割后的
Tensor 保存在 result
