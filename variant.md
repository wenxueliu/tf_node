

This is an implementation of a type-erased container that can store an
object of any type. The implementation is very similar to std::any, but has
restrictions on the types of objects that can be stored, and eschews some of
the fancier constructors available for std::any. An object of
tensorflow::Variant is intended to be used as the value that will be stored
in a tensorflow::Tensor object when its type is DT_VARIANT.

tensorflow::Variant can store an object of a class that satisfies the
following constraints:

* The class is CopyConstructible.
* The class has a default constructor.
* It's either a protocol buffer, a tensorflow::Tensor, or defines the
following functions:

  string TypeName() const;
  void Encode(VariantTensorData* data) const;
  void Decode(const VariantTensorData& data);

Simple POD types can elide the Encode/Decode functions, they are provided by
helper methods.

1. 访问对象

get<T> 函数访问

2. 序列化和反序列化

存储在 Variant 的对象 必须实现 ValueInterface 接口

存储在 Variant 的对象如果不是 POD 对象，必须实现 Encode 和 Decode 方法确保
序列化和反序列化

## 数据结构

VariantTensorDataProto -> VariantTensorData -> Variant

message VariableDef
  string variable_name = 1;   // Name of the variable tensor.
  string initializer_name = 2;// Name of the initializer op.
  string snapshot_name = 3;   // Name of the snapshot tensor.
  SaveSliceInfoDef save_slice_info_def = 4; // Support for saving variables as slices of a larger variable.
  bool is_resource = 5;       // Whether to represent this as a ResourceVariable.

message SaveSliceInfoDef
  string full_name = 1;           // Name of the full variable of which this is a slice.
  repeated int64 full_shape = 2;  // Shape of the full variable.
  repeated int64 var_offset = 3;  // Offset of this variable into the full variable.
  repeated int64 var_shape = 4;   // Shape of this variable.

class Variant
  std::unique_ptr<ValueInterface> value_;

  struct Value : ValueInterface
    T value;

class VariantTensorData
  string type_name_;
  string metadata_;
  std::vector<Tensor> tensors_;

## 实例

```cpp
Variant x = 10;
EXPECT_EQ(*x.get<int>(), 10);

Variant x;
EXPECT_EQ(x.get<void>(), nullptr);

x = Int{42};
const Variant y = x;
EXPECT_EQ(y.get<Int>()->value, 42);
x.clear();
EXPECT_EQ(x.get<void>(), nullptr);


Tensor t(DT_FLOAT, TensorShape({}));
t.flat<float>()(0) = 42.0f;
Variant x = t;
EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 42.0f);
x.get<Tensor>()->flat<float>()(0) += 1.0f;
EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 43.0f);

TensorProto t;
t.set_dtype(DT_FLOAT);
t.mutable_tensor_shape()->set_unknown_rank(true);
x = t;

EXPECT_EQ(x.TypeName(), "tensorflow.TensorProto");
EXPECT_NE(x.get<TensorProto>(), nullptr);
EXPECT_EQ(x.get<TensorProto>()->dtype(), DT_FLOAT);
EXPECT_EQ(x.get<TensorProto>()->tensor_shape().unknown_rank(), true);

Variant x;
x = []() -> Variant {
  Variant y;
  y = Int{10};
  return y;
}();
EXPECT_EQ(x.get<Int>()->value, 10);

TensorList vec;
for (int i = 0; i < 4; ++i) {
  Tensor elem(DT_INT32, {1});
  elem.flat<int>()(0) = i;
  vec.vec.push_back(elem);
}

for (int i = 0; i < 4; ++i) {
  Tensor elem(DT_FLOAT, {1});
  elem.flat<float>()(0) = 2 * i;
  vec.vec.push_back(elem);
}

x = vec;

EXPECT_EQ(x.TypeName(), "TensorList");
const TensorList& stored_vec = *x.get<TensorList>();
for (int i = 0; i < 4; ++i) {
  EXPECT_EQ(stored_vec.vec[i].flat<int>()(0), i);
}
for (int i = 0; i < 4; ++i) {
  EXPECT_EQ(stored_vec.vec[i + 4].flat<float>()(0), 2 * i);
}

VariantTensorData serialized;
x.Encode(&serialized);

Variant y = TensorList();
y.Decode(serialized);

const TensorList& decoded_vec = *x.get<TensorList>();
for (int i = 0; i < 4; ++i) {
  EXPECT_EQ(decoded_vec.vec[i].flat<int>()(0), i);
}
for (int i = 0; i < 4; ++i) {
  EXPECT_EQ(decoded_vec.vec[i + 4].flat<float>()(0), 2 * i);
}

struct Pod {
  int x;
  float y;

  string TypeName() const { return "POD"; }
};

Variant x;
Pod p{10, 20.0f};
x = p;

VariantTensorData serialized;
x.Encode(&serialized);

Variant y;
y = Pod();
y.Decode(serialized);

EXPECT_EQ(p.x, y.get<Pod>()->x);
EXPECT_EQ(p.y, y.get<Pod>()->y);
```
