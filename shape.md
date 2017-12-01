

## 数据结构

PartialTensorShape -> ShapeHandle

TensorShape -> ShapeHandle

class Dimension
  const int64 value_;

class DimensionHandle
  const Dimension* ptr_ = nullptr;

struct DimensionOrConstant
  DimensionHandle dim;
  int64 val;

class Shape
  const int32 rank_; //dims_.size()
  const std::vector<DimensionHandle> dims_;

class ShapeHandle
  const Shape* ptr_ = nullptr;

struct ShapeAndType
  ShapeHandle shape;
  DataType dtype = DT_INVALID;

class InferenceContext
  static constexpr int64 kUnknownDim = -1;
  static constexpr int32 kUnknownRank = -1;
  ShapeManager shape_manager_;
  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_; // input_tensors_[idx] 是否被请求
  std::vector<ShapeHandle> outputs_; // 容量为  output_name_map_ 中  value 索引最大的
  std::vector<ShapeHandle> input_tensors_as_shapes_; // Can have fewer elements than inputs_.
  std::vector<bool> requested_input_tensor_as_partial_shape_;

  // input_handle_shapes_and_types_[i] is the list of shape/type pairs available
  // through the resource handle passed along input i of the node.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> input_handle_shapes_and_types_;

  // output_handle_shapes_and_types_[i] is the list of shape/type pairs
  // available through the resource handle passed along output i of the node.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> output_handle_shapes_and_types_; // 容量为  output_name_map_ 中  value 索引最大的

  const int graph_def_version_;
  const NodeDef& node_def_;
  NameRangeMap input_name_map_; //同一个 key 对应的  value 是一个数组
  NameRangeMap output_name_map_; //保存 ouput_name 与 pair(start,end) 对应，方便查询

  class ShapeManager
    std::vector<Shape*> all_shapes_;    // values are owned.
    std::vector<Dimension*> all_dims_;  // values are owned.

Shape 包含 Rank 和 Dim

提供

1. 对 DimensionHandle 的 Relex, Merge, Add, Divide, Substract, Multiply, Min,
   Max
2. 对 ShapeHandle 的 MergePrefix, Relex, Merge, Subshape, Concatenate, ReplaceDim,
3. 转为 Scalar, Vector, Matrix
4. Tensor, PartialTensorShape, TensorShape, TensorShapeProto,  到 ShapeHandle, 
5. input_handle_shapes_and_types_ 和  output_handle_shapes_and_types_ 的 Merge, Relex
6. Dim, Value, Rank 及基本属性的  set, get




支持的操作

#### Merge

1. 如果两个 shape 的 ShapeHandle 相同， 或待合并的 ShapeHandle 未知，什么也不做
2. 如果待合并的 ShapeHandle 已知， 但是被合并的 ShapeHandle 未知，用待合并替代被合并的 ShapeHandle
2. 两个 shape 都已知，那么 rank 必须相同
3. 如果某一 shape 每个 dim 都比另一个多，那么直接返回前者，否则将两者合并。

如:
    [2,?] and [?,2]  合并之后为 [2,2]
    [2,2] and [1,1]  合并之后为 [2,2]

#### Relex

1. 如果两个 shape 的 ShapeHandle 有一个未知，返回 UnknownShape
2. 如果两个 shape 的 ShapeHandle 都已知且相同，什么也不做
3. 如果两个 shape 的 ranks 都存在，但是不一样，返回 UnknownShape
4. 如果两个 shape 的某一个 dim 有一个未知，该维为  UnknownDim
5. 如果两个 shape 的某一个 dim 都有一个已知，但是不匹配，该维为  UnknownDim

如
  relaxing [2,?] and [?,2] results in [?,?]
  relaxing [2,2] and [3,2] results in [?,2]
  relaxing [2,2] with [1,2,3] results in ?

#### Concatenate

#### Divide

#### Add

#### Substract

#### Min

## 源码分析

bool MergeInput(int idx, ShapeHandle shape) //将 inputs_[idx] 和 shape 合并
bool RelaxInput(int idx, ShapeHandle shape) //将 inputs_[idx] 和 shape 合并
ShapeHandle input(int64 idx) // 返回 inputs_[idx]
int num_inputs() //返回 inputs_.size()

Tensor* input_tensor(int idx)

    requested_input_tensor_[idx] = true;
    return input_tensors_[idx];

bool requested_input_tensor(int idx) //返回 requested_input_tensor_[idx];
bool requested_input_tensor_as_partial_shape(int idx) //返回 requested_input_tensor_as_partial_shape_[idx];
void set_input_tensors(const std::vector<const Tensor*>& input_tensors) //input_tensors_ = input_tensors;
void set_input_tensors_as_shapes(const std::vector<ShapeHandle>& input_tensors_as_shapes) //input_tensors_as_shapes_ = input_tensors_as_shapes;
void set_output(int idx, ShapeHandle shape) // outputs_[idx] = shape;
int num_outputs() //返回 outputs_.size()
ShapeHandle output(int idx) const //返回 outputs_[idx];
AttrSlice attrs() const // AttrSlice(node_def_)
DimensionHandle Dim(ShapeHandle s, int64 idx) // 返回 s->dims_[idx] bug 溢出可能性
int32 Rank(ShapeHandle s) //返回 s->rank_
bool RankKnown(ShapeHandle s) // 返回 (s.IsSet() && (Rank(s) != kUnknownRank));
int64 Value(DimensionOrConstant d) //返回 d.dim.IsSet() ? d.dim->value_ : d.val;
bool ValueKnown(DimensionOrConstant d) //Value(d) != kUnknownDim;
DimensionHandle UnknownDim() //返回 MakeDim(kUnknownDim);
Status construction_status() //返回 return construction_status_;
std::vector<ShapeAndType>* output_handle_shapes_and_types(int idx) //output_handle_shapes_and_types_[idx].get();
std::vector<ShapeAndType>* input_handle_shapes_and_types(int idx)  //input_handle_shapes_and_types_[idx].get();
void set_output_handle_shapes_and_types(int idx, const std::vector<ShapeAndType>& shapes_and_types)// output_handle_shapes_and_types_[idx].reset(new std::vector<ShapeAndType>(shapes_and_types));
Status InferenceContext::Run( const std::function<Status(shape_inference::InferenceContext* c)>& fn) //fn(this)

Status set_output(StringPiece output_name, const std::vector<ShapeHandle>& shapes)

从  output_name_map_ 中找到 output_name 对应的  ShapeHandle 数组，用 shapes 重置

Status InferenceContext::input(StringPiece input_name, std::vector<ShapeHandle>* output)

从 input_name_map_ 中找到 input_name 对应的 ShapeHandle 数组加入 output

Status output(StringPiece output_name, std::vector<ShapeHandle>* output)

从  output_name_map_ 中找到 output_name 对应的  ShapeHandle 数组， 加入 output

bool FullyDefined(ShapeHandle s) // 确保 s 的每一 rank 和每一 dim 都是已知类型的。

DimensionHandle NumElements(ShapeHandle s) //构建一个与 s  大小一样的 DimensionHandle

string DebugString(ShapeHandle s);
string DebugString(DimensionHandle d);
string DebugString() const;

int32 Rank(ShapeHandle s) //return s.IsSet() ? s->rank_ : kUnknownRank;

Status WithRank(ShapeHandle shape, int64 rank, ShapeHandle* out)

1. 如果 shape 的 rank 存在，必须与 rank 一样，
2. 如果不存在，构造一个和 shape 的 rank 一样的 ShapeHandle 保存在 out 中

Status WithRankAtLeast(ShapeHandle shape, int64 rank, ShapeHandle* out)

shape.rank_ 的已知，且大于 rank， out 设置为 shape
shape.rank_ 的未知，out 设置为 ReturnUnknownShape(out)

Status WithRankAtMost(ShapeHandle shape, int64 rank, ShapeHandle* out)

shape.rank_ 的已知，且小于 rank， out 设置为 shape
shape.rank_ 的未知，out 设置为 ReturnUnknownShape(out)

int64 Value(DimensionOrConstant d) //return d.dim.IsSet() ? d.dim->value_ : d.val;

Status WithValue(DimensionHandle dim, int64 value, DimensionHandle* out)

dim 的 value 与  value 一样，设置 out 为 dim
dim 的 value 未知，构造一个和 value 一样的 DimensionHandle 赋值给  out

Status InferenceContext::Merge(DimensionHandle d0, DimensionHandle d1, DimensionHandle* out)

1. d0.SameHandle(d1) 或 !ValueKnown(d1) 或 Value(d0)== Value(d1) 那么 out = d0
2. !ValueKnown(d0) && ValueKnown(d1)，out = d1

Status InferenceContext::MergePrefix(ShapeHandle s, ShapeHandle prefix, ShapeHandle* s_out, ShapeHandle* prefix_out)

1. 如果 !RankKnown(prefix) || !RankKnown(s)， s_out = s; prefix_out = prefix;
2. 否则, prefix_out 保存  prefix 和 s 的共同部分的 Merge，s_out 为 prefix_out 加上 s 剩余部分

void InferenceContext::Relax(DimensionHandle d0, DimensionHandle d1, DimensionHandle* out)

d1 和 d0 必须完全一样， 否则 out 为 UnknownDim

void InferenceContext::Relax(ShapeHandle s0, ShapeHandle s1, ShapeHandle* out)

1. s0.SameHandle(s1) 或 !RankKnown(s0) || !RankKnown(s1) 或 Rank(s0) != Rank(s1) out = UnknownShape()
2. s0 与  s1 的 Dim 和 Value 都相同, 那么返回 s0
3. 否则 s0 和 s1 的 rank 进行 Relex 合并

Status InferenceContext::Merge(ShapeHandle s0, ShapeHandle s1, ShapeHandle* out)

1. s0.SameHandle(s1) 或 !RankKnown(s1)  out 为 s0
2. !RankKnown(s0) && RankKnown(s1)  out 为 s1
3. Rank(s0) == Rank(s1) 存在 Value(d0) != Value(d1) out 为  nullptr
4. Rank(s0) == Rank(s1) && Value(d0) 是 Value(d1) 的子集，out 为 d1
5. Rank(s0) == Rank(s1) && Value(d1) 是 Value(d0) 的子集，out 为 d0
 注：这里的 Value(d0) 是 Value(d1) 的子集是指所有 Value(d0) == UnknownDim, Value(d1) != UnknownDim

Status InferenceContext::Subshape(ShapeHandle s, int64 start, ShapeHandle* out)

  return Subshape(s, start, std::numeric_limits<int64>::max(), out);

Status InferenceContext::Subshape(ShapeHandle s, int64 start_in, int64 end_in, ShapeHandle* out)

1. start_in = 0, end_in 超过 Rank(s) 返回 out = s
2. out 等于  s 中 (start_in, end_in) 的 ShapeHandle

Status InferenceContext::Concatenate(ShapeHandle s1, ShapeHandle s2, ShapeHandle* out)

将 s1 和 s2 的  dim 合并

Status ReplaceDim(ShapeHandle s, int64 dim_index, DimensionHandle new_dim, ShapeHandle* out)

复制 s.dim_ 为 dims, dims[dim_index] = new_dim, out 来源于 dims

ShapeHandle MakeShape(const std::vector<DimensionHandle>& dims);

返回 shape_manager_.MakeShape(dims);

ShapeHandle MakeShape(std::initializer_list<DimensionOrConstant> dims);

返回 shape_manager_.MakeShape(dims);

ShapeHandle UnknownShape()

返回 shape_manager_.UnknownShape()

ShapeHandle UnknownShapeOfRank(int64 rank)

构造一个 rank 的 ShapeHandle

ShapeHandle Scalar(); //ruturn MakeShape({});
ShapeHandle Vector(DimensionOrConstant dim) // return MakeShape({dim});
ShapeHandle Matrix(DimensionOrConstant dim1, DimensionOrConstant dim2); //return MakeShape({dim1, dim2});

Status MakeShapeFromShapeTensor(int input_idx, ShapeHandle* out);

如果 input_tensors_as_shapes_[input_idx] 合法，out = input_tensors_as_shapes_[input_idx];
否则  MakeShapeFromTensor(input_tensor(input_idx), input_shape, out);

Status InferenceContext::MakeShapeFromTensor(const Tensor* t, ShapeHandle tensor_shape, ShapeHandle* out)

1. 如果 t == nullptr,  创建一个 dim 为 tensor_shape.dim_[0]->value 的 ShapeHandle, 每个 dim 为 UnknownDim.
2. t->shape().dims() == 1 并且 t->type() 为 DT_INT32 或 DT_INT64，那么构造一个 t->flat.size() 的 ShapeHandle 保存在 out


Status MakeShapeFromShapeProto(const TensorShapeProto& proto, ShapeHandle* out);

1. 校验  proto 合法
2. 将 proto 转化为 PartialTensorShape
3. 将 PartialTensorShape 转换为  ShapeHandle 保存在 out

Status MakeShapeFromPartialTensorShape(const PartialTensorShape& partial_shape, ShapeHandle* out);

将 partial_shape 转换为  ShapeHandle 保存在 out

Status MakeShapeFromTensorShape(const TensorShape& shape, ShapeHandle* out);

1. 将 shape 转化为 PartialTensorShape
2. 将 PartialTensorShape 转换为  ShapeHandle 保存在 out

DimensionHandle MakeDim(DimensionOrConstant d)

Status GetScalarFromTensor(const Tensor* t, int64* val);

将 t 转换为 val
即 如果 t.dims() == 0  并且  t->dtype() 为 DT_INT32 或 DT_INT64，设置 val = t->scalar<int64>()();

Status MakeDimForScalarInput(int idx, DimensionHandle* out)

把 input_tensors_[idx]->scalar()() 转换为 DimensionHandle 保存在  out

Status MakeDimForScalarInputWithNegativeIndexing(int idx, int input_rank, DimensionHandle* out);

把 input_tensors_[idx]->scalar()()  不超过 input_rank 的转换为 DimensionHandle 保存在  out

Status GetAttr(StringPiece attr_name, T* value)

Status Divide(DimensionHandle dividend, DimensionOrConstant divisor, bool evenly_divisible, DimensionHandle* out)

创建一个  DimensionHandle, 其中 dim.value 为 dividend.dim.val / divisor.dim.val

Status Add(DimensionHandle first, DimensionOrConstant second, DimensionHandle* out);

创建一个  DimensionHandle, 其中 dim.value 为 first_value.dim.value + second_value.dim.value

Status Subtract(DimensionHandle first, DimensionOrConstant second, DimensionHandle* out);

创建一个  DimensionHandle, 其中 dim.value 为 first_value.dim.value - second_value.dim.value

Status Multiply(DimensionHandle first, DimensionOrConstant second, DimensionHandle* out);

创建一个  DimensionHandle, 其中 dim.value 为 first_value.dim.value * second_value.dim.value

Status Min(DimensionHandle first, DimensionOrConstant second, DimensionHandle* out);

创建一个  DimensionHandle, 其中 dim.value 为 min(first_value.dim.value, second_value.dim.value)

Status Max(DimensionHandle first, DimensionOrConstant second, DimensionHandle* out);

创建一个  DimensionHandle, 其中 dim.value 为 max(first_value.dim.value, second_value.dim.value)

bool MergeInputHandleShapesAndTypes(int idx, const std::vector<ShapeAndType>& shapes_and_types)

bool MergeOutputHandleShapesAndTypes(int idx, const std::vector<ShapeAndType>& shapes_and_types)

bool RelaxInputHandleShapesAndMergeTypes(int idx, const std::vector<ShapeAndType>& shapes_and_types);

bool RelaxOutputHandleShapesAndMergeTypes(int idx, const std::vector<ShapeAndType>& shapes_and_types);

Status MakeShapeFromTensor(const Tensor* t, ShapeHandle tensor_shape, ShapeHandle* out);

int graph_def_version() const // 返回 graph_def_version_;

void PreInputInit(const OpDef& op_def, const std::vector<const Tensor*>& input_tensors, const std::vector<ShapeHandle>& input_tensors_as_shapes);

1. input_tensors 设置  input_tensors_
2. input_tensors_as_shapes 设置  input_tensors_as_shapes_
3. 用 op_def 的  input 和  output 初始化 input_name_map_, output_name_map_
4. 初始化 output_， 容量为  output_name_map_ 中  value 索引最大的
5. 初始化 output_handle_shapes_and_types_，容量为  output_name_map_ 中  value 索引最大的

void PostInputInit(std::vector<std::unique_ptr<std::vector<ShapeAndType>>> input_handle_data);

1. 确保  input_.size() 与 input_name_map_ 中一致
2. 用 input_handle_data 初始化 input_handle_shapes_and_types_ 
3. 重置 input_tensors_, requested_input_tensor_, requested_input_tensor_as_partial_shape_ 的容量为  input_.size()

DimensionHandle GetDimension(const DimensionOrConstant& d);

Status ReturnUnknownShape(ShapeHandle* out)

Status ReturnCreatedShape(const std::vector<DimensionHandle>& dims, ShapeHandle* out)

Status AttachContext(const Status& status);

TODO

bool MergeHandleShapesAndTypes(std::vector<ShapeAndType>& shapes_and_types, std::vector<ShapeAndType>* to_update)

合并  shapes_and_types 和  to_update 的 ShapeHandle 和 type,

其中 type 如果  shapes_and_types 与  to_update 对应类型不同以  shapes_and_types 为准
     ShapeHandle 依赖 Merge, 如果合并失败，以 shapes_and_types 为准

bool InferenceContext::MergeOutputHandleShapesAndTypes( int idx, const std::vector<ShapeAndType>& shapes_and_types)

合并 shapes_and_types 与 output_handle_shapes_and_types_[idx] 的  ShapeHandle 与 type

bool InferenceContext::MergeInputHandleShapesAndTypes( int idx, const std::vector<ShapeAndType>& shapes_and_types)

合并 shapes_and_types 与 input_handle_shapes_and_types_[idx] 的  ShapeHandle 与 type

bool RelaxHandleShapesAndMergeTypes(std::vector<ShapeAndType>& shapes_and_types, std::vector<ShapeAndType>* to_update);

对应 shapes_and_types 和 to_update 的  type 和  ShapeHandle 进行 relax 操作

bool InferenceContext::RelaxOutputHandleShapesAndMergeTypes( int idx, const std::vector<ShapeAndType>& shapes_and_types)

对应 shapes_and_types 和 output_handle_shapes_and_types_[idx] 的  type 和  ShapeHandle 进行 relax 操作

bool InferenceContext::RelaxInputHandleShapesAndMergeTypes( int idx, const std::vector<ShapeAndType>& shapes_and_types)

对应 shapes_and_types 和 input_handle_shapes_and_types_[idx] 的  type 和  ShapeHandle 进行 relax 操作


ShapeHandle InferenceContext::ShapeManager::MakeShape( const std::vector<DimensionHandle>& dims)

  all_shapes_.push_back(new Shape(dims));
  return all_shapes_.back();

ShapeHandle InferenceContext::ShapeManager::UnknownShape()

  all_shapes_.push_back(new Shape());
  return all_shapes_.back();


InferenceContext::InferenceContext(
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<TensorShapeProto>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<TensorShapeProto>& input_tensors_as_shapes,
    const std::vector<std::unique_ptr<std::vector<std::pair<TensorShapeProto, DataType>>>>& input_handle_shapes_and_types)

1. graph_def_version 初始化 graph_def_version_, node_def 初始化  node_def_
2. input_tensors 初始化 input_tensors_
3. input_tensors_as_shapes_ = input_tensors_as_shapes;
4. node_def, op_def 初始化 input_name_map_, output_name_map_
5. input_shapes 初始化  inputs_
6. input_handle_shapes_and_types 初始化 input_handle_shapes_and_types_
7. 重置 input_tensors_, requested_input_tensor_, requested_input_tensor_as_partial_shape_ 的容量


InferenceContext::InferenceContext(
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<PartialTensorShape>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<PartialTensorShape>& input_tensors_as_shapes,
    const std::vector<std::unique_ptr<std::vector<std::pair<PartialTensorShape, DataType>>>>& input_handle_shapes_and_types)

同上，不过类型变为了 PartialTensorShape
1. graph_def_version 初始化 graph_def_version_, node_def 初始化  node_def_
2. input_tensors 初始化 input_tensors_
3. input_tensors_as_shapes_ = input_tensors_as_shapes;
4. node_def, op_def 初始化 input_name_map_, output_name_map_
5. input_shapes 初始化  inputs_
6. input_handle_shapes_and_types 初始化 input_handle_shapes_and_types_
7. 重置 input_tensors_, requested_input_tensor_, requested_input_tensor_as_partial_shape_ 的容量

InferenceContext::InferenceContext(
    int graph_def_version, const NodeDef* node_def, const OpDef& op_def,
    const std::vector<ShapeHandle>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes,
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
        input_handle_shapes_and_types)

同上，不过类型变为了 ShapeHandle，不必要进行类型转换
1. graph_def_version 初始化 graph_def_version_, node_def 初始化  node_def_
2. input_tensors 初始化 input_tensors_
3. input_tensors_as_shapes_ = input_tensors_as_shapes;
4. node_def, op_def 初始化 input_name_map_, output_name_map_
5. input_shapes 初始化  inputs_
6. input_handle_shapes_and_types 初始化 input_handle_shapes_and_types_
7. 重置 input_tensors_, requested_input_tensor_, requested_input_tensor_as_partial_shape_ 的容量
