
通过 function 来创建 Operation，首先创建 FunctionDef，之后定义 Creator
之后通过 REGISTER_OP_NO_GRADIENT  或 REGISTER_OP_GRADIENT 注册到全局变量
factory 中，之后可以通过 GetOpGradientCreator 查询。

SymbolicGradientHelper::Compute 进行来的  Gradient 计算，具体是通过
Status SymbolicGradientBuilder::AddGradients()  来实现

目前系统以及注册的函数

* REGISTER_OP_NO_GRADIENT("Shape");
* REGISTER_OP_NO_GRADIENT("Rank");
* REGISTER_OP_NO_GRADIENT("Size");
* REGISTER_OP_NO_GRADIENT("ZerosLike");
* REGISTER_OP_NO_GRADIENT("OnesLike");
* REGISTER_OP_NO_GRADIENT("Const");
* REGISTER_OP_NO_GRADIENT("EditDistance");
* REGISTER_OP_NO_GRADIENT("StopGradient");
* REGISTER_OP_NO_GRADIENT("Less");
* REGISTER_OP_NO_GRADIENT("LessEqual");
* REGISTER_OP_NO_GRADIENT("Greater");
* REGISTER_OP_NO_GRADIENT("GreaterEqual");
* REGISTER_OP_NO_GRADIENT("Equal");
* REGISTER_OP_NO_GRADIENT("NotEqual");
* REGISTER_OP_NO_GRADIENT("LogicalAnd");
* REGISTER_OP_NO_GRADIENT("LogicalOr");
* REGISTER_OP_NO_GRADIENT("LogicalNot");
* REGISTER_OP_NO_GRADIENT("Range");
* REGISTER_OP_NO_GRADIENT("LinSpace");
* REGISTER_OP_NO_GRADIENT("Floor");
* REGISTER_OP_NO_GRADIENT("FloorDiv");
* REGISTER_OP_NO_GRADIENT("TruncateDiv");
* REGISTER_OP_NO_GRADIENT("RandomUniform");
* REGISTER_OP_GRADIENT("Reshape", ReshapeGrad);
* REGISTER_OP_GRADIENT("ExpandDims", ReshapeGrad);
* REGISTER_OP_GRADIENT("Squeeze", SqueezeGrad);
* REGISTER_OP_GRADIENT("Identity", IdentityGrad);
* REGISTER_OP_GRADIENT("Pack", PackGrad);
* REGISTER_OP_GRADIENT("Unpack", UnpackGrad);
* REGISTER_OP_GRADIENT("Concat", ConcatGrad);
* REGISTER_OP_GRADIENT("ConcatV2", ConcatGradV2);
* REGISTER_OP_GRADIENT("Split", SplitGrad);
* REGISTER_OP_GRADIENT("_ArrayToList", ArrayToListGrad);
* REGISTER_OP_GRADIENT("_ListToArray", ListToArrayGrad);
* REGISTER_OP_GRADIENT("Fill", FillGrad);
* REGISTER_OP_GRADIENT("Transpose", TransposeGrad);
* REGISTER_OP_GRADIENT("Reverse", ReverseGrad);
* REGISTER_OP_GRADIENT("ReverseV2", ReverseV2Grad);
* REGISTER_OP_GRADIENT("Slice", SliceGrad);
* REGISTER_OP_GRADIENT("StridedSlice", StridedSliceGrad);
* REGISTER_OP_GRADIENT("StridedSliceGrad", StridedSliceGradGrad);
* REGISTER_OP_GRADIENT("MapAccumulate", MapAccumulateGrad);
* REGISTER_OP_GRADIENT("Abs", AbsGrad);
* REGISTER_OP_GRADIENT("Neg", NegGrad);
* REGISTER_OP_GRADIENT("Inv", InvGrad);
* REGISTER_OP_GRADIENT("Reciprocal", InvGrad);
* REGISTER_OP_GRADIENT("Square", SquareGrad);
* REGISTER_OP_GRADIENT("Sqrt", SqrtGrad);
* REGISTER_OP_GRADIENT("Rsqrt", RsqrtGrad);
* REGISTER_OP_GRADIENT("Exp", ExpGrad);
* REGISTER_OP_GRADIENT("Expm1", Expm1Grad);
* REGISTER_OP_GRADIENT("Log", LogGrad);
* REGISTER_OP_GRADIENT("Log1p", Log1pGrad);
* REGISTER_OP_GRADIENT("Sinh", SinhGrad);
* REGISTER_OP_GRADIENT("Cosh", CoshGrad);
* REGISTER_OP_GRADIENT("Tanh", TanhGrad);
* REGISTER_OP_GRADIENT("Asinh", AsinhGrad);
* REGISTER_OP_GRADIENT("Acosh", AcoshGrad);
* REGISTER_OP_GRADIENT("Atanh", AtanhGrad);
* REGISTER_OP_GRADIENT("Sigmoid", SigmoidGrad);
* REGISTER_OP_GRADIENT("Sign", SignGrad);
* REGISTER_OP_GRADIENT("Sin", SinGrad);
* REGISTER_OP_GRADIENT("Cos", CosGrad);
* REGISTER_OP_GRADIENT("Acos", AcosGrad);
* REGISTER_OP_GRADIENT("Asin", AsinGrad);
* REGISTER_OP_GRADIENT("Atan", AtanGrad);
* REGISTER_OP_GRADIENT("Tan", TanGrad);
* REGISTER_OP_GRADIENT("Real", RealGrad);
* REGISTER_OP_GRADIENT("Imag", ImagGrad);
* REGISTER_OP_GRADIENT("Conj", ConjGrad);
* REGISTER_OP_GRADIENT("Add", AddGrad);
* REGISTER_OP_GRADIENT("Sub", SubGrad);
* REGISTER_OP_GRADIENT("Mul", MulGrad);
* REGISTER_OP_GRADIENT("Div", DivGrad);
* REGISTER_OP_GRADIENT("RealDiv", RealDivGrad);
* REGISTER_OP_GRADIENT("Pow", PowGrad);
* REGISTER_OP_GRADIENT("Maximum", MaximumGrad);
* REGISTER_OP_GRADIENT("Minimum", MinimumGrad);
* REGISTER_OP_GRADIENT("Complex", ComplexGrad);
* REGISTER_OP_GRADIENT("Select", SelectGrad);
* // REGISTER_OP_GRADIENT("AddN", AddNGrad);
* REGISTER_OP_GRADIENT("Sum", SumGrad);
* REGISTER_OP_GRADIENT("Mean", MeanGrad);
* // REGISTER_OP_GRADIENT("Prod", ProdGrad);
* // REGISTER_OP_GRADIENT("SegmentSum", SegmentSumGrad);
* // REGISTER_OP_GRADIENT("SegmentMean", SegmentMeanGrad);
* // REGISTER_OP_GRADIENT("SparseSegmentSum", SparseSegmentSumGrad);
* // REGISTER_OP_GRADIENT("SparseSegmentMean", SparseSegmentMeanGrad);
* // REGISTER_OP_GRADIENT("SparseSegmentSqrtN", SparseSegmentSqrtNGrad);
* // REGISTER_OP_GRADIENT("SegmentMin", SegmentMinGrad);
* // REGISTER_OP_GRADIENT("SegmentMax", SegmentMaxGrad);
* // REGISTER_OP_GRADIENT("UnsortedSegmentSum", UnsortedSegmentSumGrad);
* // REGISTER_OP_GRADIENT("UnsortedSegmentMax", UnsortedSegmentMaxGrad);
* REGISTER_OP_GRADIENT("Max", MaxGrad);
* REGISTER_OP_GRADIENT("Min", MinGrad);
* REGISTER_OP_GRADIENT("MatMul", MatMulGrad);
* REGISTER_OP_GRADIENT("BatchMatMul", BatchMatMulGrad);
* // REGISTER_OP_GRADIENT("SparseMatMul", SparseMatMulGrad);
* REGISTER_OP_GRADIENT("Softmax", SoftmaxGrad);
* REGISTER_OP_GRADIENT("Relu", ReluGrad);
* REGISTER_OP_GRADIENT("Relu6", Relu6Grad);
* REGISTER_OP_GRADIENT("CrossEntropy", CrossEntropyGrad);
* REGISTER_OP_GRADIENT("Conv2D", Conv2DGrad);
* REGISTER_OP_GRADIENT("MaxPool", MaxPoolGrad);
* REGISTER_OP_GRADIENT("AvgPool", AvgPoolGrad);
* REGISTER_OP_GRADIENT("MaxPoolGrad", MaxPoolGradGrad);
* REGISTER_OP_GRADIENT("BiasAdd", BiasAddGrad);



```

#define REGISTER_OP_GRADIENT(name, fn)
  static bool unused_grad_##ctr = SHOULD_REGISTER_OP_GRADIENT && ::tensorflow::gradient::RegisterOp(name, fn)

#define REGISTER_OP_NO_GRADIENT(name)
  static bool unused_grad___COUNTER__ = SHOULD_REGISTER_OP_GRADIENT && ::tensorflow::gradient::RegisterOp(name, nullptr)
```

通过  FunctionDefHelper::Create 构造 FunctionDefHelper 对象 fdef，之后通过
InstantiationResult 将 fdef 转换为 InstantiationResult, 及 Cpp 语法的函数对象

## 数据结构


FunctionDef -> FunctionDefLibrary -> FunctionLibraryDefinition
FunctionDef -> FunctionDefAndOpRegistration

FunctionDef -> FunctionBody
FunctionCallFrame

typedef std::function<Status(const AttrSlice& attrs, FunctionDef*)> Creator;
typedef std::unordered_map<string, Creator> OpGradFactory;
static OpGradFactory* factory = new OpGradFactory; 

op:function 的映射关系保存在全局变量 factory 中
1. 通过 REGISTER_OP_NO_GRADIENT 注册
2. GetOpGradientCreator 查询

message FunctionDefLibrary
  repeated FunctionDef function = 1;
  repeated GradientDef gradient = 2;

message FunctionDef
  OpDef signature = 1;
  map<string, AttrValue> attr = 5;
  repeated NodeDef node_def = 3;
  map<string, string> ret = 4;

message GradientDef
  string function_name = 1;  // The function name.
  string gradient_func = 2;  // The gradient function's name.

class SymbolicGradientHelper
  const FunctionBody* fbody_;
  FunctionBody* gbody_ = nullptr;

struct FunctionBody
  FunctionDef fdef;
  Graph* graph = nullptr;  // owned.
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  gtl::InlinedVector<Node*, 4> arg_nodes;
  gtl::InlinedVector<Node*, 4> ret_nodes;

if we have
   (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
then, g is
   (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                     dL/dy1, dL/dy2, ..., dL/dy_M),

class FunctionLibraryRuntime

class FunctionLibraryRuntimeImpl : public FunctionLibraryRuntime
  typedef FunctionLibraryRuntimeImpl ME;
  const DeviceMgr* const device_mgr_;
  Device* const device_;
  Env* const env_;
  const int graph_def_version_;
  const FunctionLibraryDefinition* const lib_def_;
  GraphOptimizer optimizer_;
  const CustomKernelCreator custom_kernel_creator_;
  std::function<Status(const string&, const OpDef**)> get_func_sig_; //lib_def_->LookUpOpDef(op, sig)
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel_; //CreateKernel(ndef, kernel);
  mutable mutex mu_;
  // Maps function instantiation to a handle. The key is a
  // canonicalized representation of the function name and
  // instantiation attrs. The handle is an index into the items_.
  std::unordered_map<string, Handle> table_ // key 为 Canonicalize(function_name, attrs)
  // func_graphs_ never shrinks or reorders its members.
  std::vector<FunctionBody*> func_graphs_ //
  std::vector<Item*> items_; //保存所有的 Item, 索引为 Handle

class FunctionDefHelper

typedef std::function<Status(const string&, const OpDef**)> GetFunctionSignature;

struct InstantiationResult
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  std::vector<NodeDef> nodes;

class FunctionCallFrame
  DataTypeVector arg_types_;
  DataTypeVector ret_types_;
  gtl::InlinedVector<Tensor, 4> args_;
  struct Retval {
    bool has_val = false;
    Tensor val;
  };
  gtl::InlinedVector<Retval, 4> rets_;


class FunctionLibraryDefinition : public OpRegistryInterface
  static constexpr const char* const kGradientOp = "SymbolicGradient";
  static constexpr const char* const kFuncAttr = "f";
  const OpRegistryInterface* const default_registry_;
  gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>> function_defs_; 保存 FunctionDefLibrary 中所有 function.signature().name() 与 function 的键值对
  gtl::FlatMap<string, string> func_grad_; 保存 FunctionDefLibrary 中所有 gradient


struct InstantiationResult
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  std::vector<NodeDef> nodes;

class FunctionInstantiationHelper
  GetFunctionSignature get_function_;
  InstantiationResult& result_;
  std::unordered_map<string, NameInfoItem> index_;
  // This contains information about a node in the new graph including the node names and input nodes' indexes.
  struct NodeInfo {
    string name;
    // Data inputs where <n, k> means arg k of node n.
    std::vector<std::pair<int, int>> data_inputs;
    // Control inputs (dependencies).
    std::vector<int> control_inputs;
  };
  std::vector<NodeInfo> nodes_; //保存 NodeDef 的 input, 后面加入 result_.nodes 的 input 中

注：首先阅读  InstantiateFunction 之后再看  FunctionInstantiationHelper 成员函数实现细节符合实现顺序

## 例子

``` cpp
  REGISTER_OP("One")
      .Output("y: T")
      .Attr("T: {float, double, int32, int64}")
      .Doc(R"doc(
  Returns a tensor with a single element (1) of type T.

  y: A scalar in type T.

  )doc");

  Status GetOpSig(const string& op, const OpDef** sig) {
    return OpRegistry::Global()->LookUpOpDef(op, sig);
  }
  auto fdef = FunctionDefHelper::Create(
      // Name
      "SquarePlusOne",
      // Inputs
      {"x: T"},
      // Outputs
      {"y: T"},
      // Attrs
      {"T: {float, double, int32, int64}"},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", "$T"}}},
       // o = One<T>()
       // NOTE: We can also have a Cast<Tin, Tout>(x) instead.
       {{"o"}, "One", {}, {{"T", "$T"}}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a:y", "o:y"}, {{"T", "$T"}}}},
      // Returns
      {{"y", "y:z:0"}};

  const char *e  == R"P(SquarePlusOne[T:{float, double, int32, int64}](x:T) -> (y:T) {
      a = Square[T=$T](x)
      o = One[T=$T]()
      y = Add[T=$T](a:y, o:y)
      return y = y:z:0
    )P"
  DebugString(fdef) == e
  InstantiationResult result;
  InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  DebugString(result.nodes)
    == R"P((x:float) -> (y:float) {
          a = Square[T=float](x)
          o = One[T=float]()
          y = Add[T=float](a, o)
        })P"

  REGISTER_OP("HasDefaultType")
    .Output("out: T")
    .Attr("T: {float, double, int32, int64} = DT_FLOAT");

  auto fdef = FunctionDefHelper::Create(
      // Name
      "ControlDep",
      // Inputs
      {"x: int32"},
      // Outputs
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Identity<int32>(x)
       {{"a"}, "Identity", {"x"}, {{"T", DT_INT32}}},
       // o = NoOp(^a)
       {{"o"}, "NoOp", {"^a"}, {}},
       // y = Identity<int32>(a, ^o)
       {{"y"}, "Identity", {"a:output:0", "^o"}, {{"T", DT_INT32}}}},
      // Returns
      {{"y", "y:output:0"}};

  const char* e = R"P(
ControlDep(x:int32) -> (y:int32) {
  a = Identity[T=int32](x)
  o = NoOp() @ a
  y = Identity[T=int32](a:output:0) @ o
  return y = y:output:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result);
  const char* e2 = R"P(
(x:int32) -> (y:int32) {
  a = Identity[T=int32](x)
  o = NoOp() @ a
  y = Identity[T=int32](a) @ o
}
)P";

  REGISTER_OP("HasDefaultType")
    .Output("out: T")
    .Attr("T: {float, double, int32, int64} = DT_FLOAT");

  auto fdef = FDH::Create(
      // Name
      "BackCompat",
      // Args
      {},
      // Return values
      {"y: float"},
      // Attrs
      {},
      // Nodes
      {// y = HasDefaultType(x), T missing, defaults to float
       {{"a"}, "HasDefaultType", {}, {}}},
      // Returns
      {{"y", "a:out:0"}};

  const char* e = R"P(
BackCompat() -> (y:float) {
  a = HasDefaultType()
  return y = a:out:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result);
  // Should get T=float from Op's default.
  const char* e2 = R"P(
() -> (a:float) {
  a = HasDefaultType[T=float]()
}
)P";

  auto fdef = FunctionDefHelper::Create(
      // Name
      "NTimesT",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = AddN<N=2>(x, y)
       {{"a"}, "AddN", {"x", "y"}, {{"T", DT_FLOAT}, {"N", 2}}}},
      // Returns
      {{"z", "a:sum:0"}};

  const char* e = R"P(
NTimesT(x:float, y:float) -> (z:float) {
  a = AddN[N=2, T=float](x, y)
  return z = a:sum:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result);
  const char* e2 = R"P(
(x:float, y:float) -> (a:float) {
  a = AddN[N=2, T=float](x, y)
}
)P";

  REGISTER_OP("Map")
    .Input("x: N * T")
    .Output("y: N * U")
    .Attr("T: type")
    .Attr("U: type")
    .Attr("N: int >= 1")
    // .Attr("func: func_name_with_attr")
    .Doc(R"doc(
Applies the 'func' on every input. I.e.,

y[i] = func<...>(x[i])

x: N tensors, each of type T;
y: N tensors, each of type U;

)doc");

  auto fdef = FunctionDefHelper::Create(
      // Name
      "AddSquared",
      // Args
      {"x: N*T"},
      // Return values
      {"y: T"},
      // Attrs
      {"N:int", "T:{float, double, int32, int64}"},
      // Nodes
      {// a = Map<func=Square<$T>,T=$T,U=$T,N=$N>(x)
       {{"a"},
        "Map",
        {"x"},
        {{"func", FDH::FunctionRef("Square", {{"T", "$T"}})},
         {"T", "$T"},
         {"U", "$T"},
         {"N", "$N"}}},
       // y = AddN<N=$N,T=$T>(a)
       {{"y"}, "AddN", {"a:y"}, {{"N", "$N"}, {"T", "$T"}}}},
      {{"y", "y:sum"}});

  const char* e = R"P(
AddSquared[N:int, T:{float, double, int32, int64}](x:N*T) -> (y:T) {
  a = Map[N=$N, T=$T, U=$T, func=Square[T=$T]](x)
  y = AddN[N=$N, T=$T](a:y)
  return y = y:sum
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  InstantiateFunction(fdef, Attrs({{"N", 3}, {"T", DT_FLOAT}}), GetOpSig, &result);
  const char* e2 = R"P(
(x_0:float, x_1:float, x_2:float) -> (y:float) {
  a = Map[N=3, T=float, U=float, func=Square[T=float]](x_0, x_1, x_2)
  y = AddN[N=3, T=float](a, a:1, a:2)
}
)P";

  const Tensor kZero = test::AsScalar<int32>(0);
  auto fdef = FunctionDefHelper::Create(
      // Name
      "Test",
      // Args
      {"i:float"},
      // Return values
      {"o:float"},
      // Attrs
      {},
      // Nodes
      {{{"zero"}, "Const", {}, {{"value", kZero}, {"dtype", DT_INT32}}},
       {{"s"},
        "Split",
        {"zero:output:0", "i"},
        {{"num_split", 4}, {"T", DT_FLOAT}}},
       {{"l"}, "Mul", {"s:output:0", "s:output:1"}, {{"T", DT_FLOAT}}},
       {{"r"}, "Mul", {"s:output:2", "s:output:3"}, {{"T", DT_FLOAT}}},
       {{"x"},
        "_ListToArray",
        {"l:z", "r:z"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       {{"o"}, "AddN", {"x:output"}, {{"N", 2}, {"T", DT_FLOAT}}}},
      {{"o", "o:sum:0"}});

  const char* e = R"P(
Test(i:float) -> (o:float) {
  zero = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  s = Split[T=float, num_split=4](zero:output:0, i)
  l = Mul[T=float](s:output:0, s:output:1)
  r = Mul[T=float](s:output:2, s:output:3)
  x = _ListToArray[N=2, T=float, Tin={float, float}](l:z, r:z)
  o = AddN[N=2, T=float](x:output)
  return o = o:sum:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result);
  const char* e2 = R"P(
(i:float) -> (o:float) {
  zero = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  s = Split[T=float, num_split=4](zero, i)
  l = Mul[T=float](s, s:1)
  r = Mul[T=float](s:2, s:3)
  x = _ListToArray[N=2, T=float, Tin={float, float}](l, r)
  o = AddN[N=2, T=float](x, x:1)
}
)P";

  REGISTER_OP("Cond")
    .Input("input: Tin")
    .Output("output: out_types")
    .Attr("Tin: list(type)")
    .Attr("out_types: list(type)")
    .Attr("cond: func")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Doc(R"doc(
output = Cond(input) ? then_branch(input) : else_branch(input)

cond: A function takes 'input' and returns a scalar.
then_branch: A function takes 'input' and returns 'output'.
else_branch: A function takes 'input' and returns 'output'.
)doc");

TEST(TFunc, Body_Array_List_Converter) {
  auto fdef = FDH::Define(
      // Name
      "MySelect",
      // Args
      {"x:float"},
      // Return values
      {"z:float"},
      // Attrs
      {},
      // Nodes
      {
          {{"y"},
           "Cond",
           {"x"},
           {{"Tin", DataTypeSlice{DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond")},
            {"then_branch", FDH::FunctionRef("MyThen")},
            {"else_branch", FDH::FunctionRef("MyElse")}}},
          {{"z"},
           "Cond",
           {"y", "y"},
           {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond2")},
            {"then_branch", FDH::FunctionRef("MyThen2")},
            {"else_branch", FDH::FunctionRef("MyElse2")}}},
      });

  const char* e = R"P(
MySelect(x:float) -> (z:float) {
  y = Cond[Tin={float}, cond=MyCond, else_branch=MyElse, out_types={float}, then_branch=MyThen](x)
  z = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](y:output:0, y:output:0)
  return z = z:output:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result);
  const char* e2 = R"P(
(x:float) -> (z:float) {
  y = Cond[Tin={float}, cond=MyCond, else_branch=MyElse, out_types={float}, then_branch=MyThen](x)
  z = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](y, y)
}
)P";
```


``` cpp
class FunctionTest {
 protected:
  FunctionTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")) {}

  void Create(const FunctionDef& fdef, Attrs attrs) {
    exec_ = nullptr;
    InstantiationResult result;
    TF_CHECK_OK(InstantiateFunction(fdef, attrs, GetOpSig, &result));

    arg_types_ = result.arg_types;
    ret_types_ = result.ret_types;

    Graph* g = new Graph(OpRegistry::Global());
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = false;
    TF_CHECK_OK(ConvertNodeDefsToGraph(opts, result.nodes, g));

    const int version = g->versions().producer();
    LocalExecutorParams params;
    params.device = device_.get();
    params.create_kernel = [this, version](const NodeDef& ndef,
                                           OpKernel** kernel) {
      return CreateNonCachedKernel(device_.get(), nullptr, ndef, version,
                                   kernel);
    };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    Executor* exec;
    TF_CHECK_OK(NewLocalExecutor(params, g, &exec));
    exec_.reset(exec);
  }

  void Run(const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionCallFrame frame(arg_types_, ret_types_);
    TF_CHECK_OK(frame.SetArgs(args));
    Executor::Args exec_args;
    exec_args.call_frame = &frame;
    exec_args.runner = FunctionTestSchedClosure;
    TF_CHECK_OK(exec_->Run(exec_args));
    std::vector<Tensor> computed;
    TF_CHECK_OK(frame.GetRetvals(&computed));
    CHECK_EQ(computed.size(), rets.size());
    for (int i = 0; i < rets.size(); ++i) {
      *(rets[i]) = computed[i];
    }
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<Executor> exec_;
  DataTypeVector arg_types_;
  DataTypeVector ret_types_;
};
```

## 源码分析

FunctionLibraryDefinition(const OpRegistryInterface* default_registry, const FunctionDefLibrary& lib_def)


Status ArgNumType(AttrSlice attrs, const OpDef::ArgDef& arg_def, bool* is_type_list, DataTypeVector* dtypes)

    if (arg_def 是 type_list 不为空，并且在  attrs 中找到了对应的值)
        is_type_list = true, dtypes 保存了每个元素的类型
    else
        is_type_list = false
        if (arg_def.type() != DT_INVALID)
            dtype = arg_def.type();
        else if (arg_def.type_attr().empty())
            dtype = DT_INVALID;
        else
            dtype = v->type();
        dtypes->resize(num, dtype);

Status ValidateSignatureWithAttrs(const OpDef& sig, AttrSlice attr_values)

确保 sig.attr() 每个属性在  attr_values 中都可以找到，并且是合法的类型

Status FunctionInstantiationHelper::AddItem(const string& name, const NameInfoItem& item)

将 {name, item} 插入 index_

NodeDef* FunctionInstantiationHelper::AddNode(const string& name)

将 {name, {}, {}} 的 NodeInfo 加入 node_

const NameInfoItem* FunctionInstantiationHelper::GetItemOrNull(const string& name)

从 index_ 中查找  name 对应的 NameInfoItem

void FunctionInstantiationHelper::AddInput(int node_index, int output_node, int output_index) {

nodes_[node_index].data_inputs 中加入 std::makie_pair(output_node, output_index)

void FunctionInstantiationHelper::AddDep(int node_index, int dep_index) {

nodes_[node_index].data_inputs 中加入 dep_index

void FunctionInstantiationHelper::AddNodeInputs()

将 nodes_[i].data_inputs 和 nodes_[i].control_inputs 中的依次加入 result_.nodes[i] 的 input

Status FunctionInstantiationHelper::BuildInputArgIndex(const OpDef::ArgDef& arg_def, AttrSlice attr_values)

1. 根据 attr_values, arg_def 找到  dtypes, is_type_list
2. index_ 中加入 { arg_def.name(), {true, result_.nodes.size(), 0, is_type_list, dtypes})
3. 遍历 dtypes
3.1 index_ 中加入 {arg_def.name()":"i, {true, result_.nodes.size(), 0, false, {dtypes[i]}}
3.2 增加 Node
    name: arg_def.name() + i
    op : `_Arg`
    attr:
       T: dtypes[i]
       index: arg_index
3.3 result_.arg_types 加入 dtypes[i]

Status FunctionInstantiationHelper::BuildNodeOutputIndex(const NodeDef& node, AttrSlice attrs, const int arg_index)

1. get_function_(node.op(), &node_sig) 获取的 OpDef node_sig
2. index_ 中加入 { node.name()":"node_sig->output_arg(i).name(), {false, arg_index, start, is_type_list, dtypes}}
3. 遍历 dtypes, index_ 中加入 { node.name()":"node_sig->output_arg(i).name() + j, {false, arg_index, start + j , false , {dtypes[k]}}}

Status FunctionInstantiationHelper::InstantiateNode(const NodeDef& fnode, AttrSlice attrs)

1. get_function_(fnode.op(), &fnode_sig) 获取的 OpDef fnode_sig
2. node_[nodes_.size() - 1].data_inputs 加入 index_ 中对应的元素 TODO
3. nodes_[nodes_.size() - 1].data_inputs 加入 index_ 中 fnode.input(i).substr(1) 对应的元素
4. 将 attr 中的元素依次加入节点属性

Status FunctionInstantiationHelper::AddReturnNode(const OpDef::ArgDef& ret_def, AttrSlice attrs,
      const ::tensorflow::protobuf::Map<string, string>& ret_map, int* ret_index)

1. 从 ret_map 中找到 ret_def.name() 对应的 value, 从  index_ 找到 value 对应的 item
2. 从 attrs 中找到  ret_def 对应的 dtypes
3. 添加 Node 其中
    name : ret_def.name() + "_RetVal" + i
    op : `_RetVal`
    attr :
        dtypes[i]: T
        ret_index: index
4. 将 dtypes 依次加入 result_.ret_types 中
5. nodes_[nodes_.size() - 1] 中加入 item->nid, item->idx + i

string Print(const OpDef::ArgDef& arg)
string Print(const NodeDef& n)
string Print(const FunctionDef& fdef)
string Print(gtl::ArraySlice<const NodeDef*> nodes)


Status AddDefaultAttrs(const string& op, const GetFunctionSignature& get_function, AttrValueMap* attrs)

1. get_function(op, &op_def) 初始化 op_def
2. 遍历  op_def 的属性，如果该属性有默认值，但是在 attrs 中找不到，就将该默认属性加入 attrs

Status InstantiateFunction(const FunctionDef& fdef, AttrSlice attr_values,
                           GetFunctionSignature get_function, InstantiationResult* result)

1. 确保 fdef.signature().attr() 中的属性在 attr_values 中都可以找到
2. 初始化输入参数
3. 初始化输出参数
4. 实例化节点
5. 增加返回值

string DebugString(const FunctionDef& func_def)
string DebugString(const GraphDef& instantiated_func_def)
string DebugString(gtl::ArraySlice<NodeDef> instantiated_func_nodes)
string DebugStringWhole(const GraphDef& gdef)

std::map<string, AttrValue> GetSetAttrs(const FunctionDef& fdef)

将 fdef.attr 中 value 如果已经设置，加入 map 返回

bool FunctionDefsEqual(const FunctionDef& f1, const FunctionDef& f2)

依次比较 signature(), attr(), node_def(), ret()

string Canonicalize(const string& funcname, AttrSlice attrs)

转换为 funcname[key=value1,key2=value2] 的形式

Status FunctionCallFrame::SetArgs(gtl::ArraySlice<Tensor> args)

用 args 初始化 args_

Status FunctionCallFrame::GetRetvals(std::vector<Tensor>* rets)

将 ret_ 加入 rets

Status FunctionCallFrame::ConsumeRetvals(std::vector<Tensor>* rets)

将 ret_ 移动到 rets

Status FunctionCallFrame::GetArg(int index, Tensor* val)

val = args_[index]

Status FunctionCallFrame::SetRetval(int index, const Tensor& val)

设置 rets_[index] 为 val

const FunctionDef* FunctionLibraryDefinition::Find(const string& name)

function_defs_ 中找到  name 对应的 FunctionDef

Status FunctionLibraryDefinition::AddFunctionDef(const FunctionDef& fdef)

如果已经存在于function_defs_ 或 default_registry_，返回
否则，加入 function_defs_[fdef.signature().name()]

Status FunctionLibraryDefinition::AddGradientDef(const GradientDef& grad)

如果已经存在于 func_grad_[grad.function_name()]，返回
否则, 将 grad.gradient_func() 加入 func_grad_[grad.function_name()]

Status FunctionLibraryDefinition::AddLibrary(const FunctionLibraryDefinition& other)


Status FunctionLibraryDefinition::AddLibrary( const FunctionDefLibrary& lib_def)

将 other.function() 加入 function_defs_
将 other.gradient() 加入 func_grad_

string FunctionLibraryDefinition::FindGradient(const string& func)

从 func_grad_ 中找到 func 对应的值

Status FunctionLibraryDefinition::LookUp( const string& op, const OpRegistrationData** op_reg_data)

首先在 function_defs_ 中查找，找不到再从 default_registry_ 中查找

const FunctionDef* FunctionLibraryDefinition::GetAttrImpl(const NodeDef& ndef)

依次从 function_defs_ 的 node.op(), func_grad_[node.attr()[f]], node.attr()[f] 中查找

FunctionDefLibrary FunctionLibraryDefinition::ToProto()

将 FunctionLibraryDefinition  转换为 FunctionDefLibrary

Status FunctionLibraryDefinition::GetAttr(const NodeDef& ndef, const string& attr, T* value)

从 ndef 对应的 FunctionDef 中查找 attr 对应的 value

Status FunctionLibraryDefinition::GetAttr(const Node& node, const string& attr, T* value)

从 node.def() 对应的 FunctionDef 中查找 attr 对应的 value

FunctionDefHelper::AttrValueWrapper FunctionDefHelper::FunctionRef(const string& name,
    gtl::ArraySlice<std::pair<string, AttrValueWrapper>> attrs)

将 ret.proto.func.name 设置为 name, attrs 加入 ret.proto.func.attr

FunctionDef FunctionDefHelper::Create(
    const string& function_name, gtl::ArraySlice<string> in_def,
    gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
    gtl::ArraySlice<Node> node_def,
    gtl::ArraySlice<std::pair<string, string>> ret_def)

1. in_def, out_def, attr_def 构造  OpDef 初始化 fdef.signature
2. node_def 初始化 fdef.node_def
3. ret_def 初始化  fdef.ret
返回 fdef

FunctionDef FunctionDefHelper::Define(const string& name, gtl::ArraySlice<string> arg_def,
    gtl::ArraySlice<string> ret_def, gtl::ArraySlice<string> attr_def, gtl::ArraySlice<Node> node_def)

1. name, arg_def, ret_def, attr_def 构造 OpDef 初始化 fdef.signature
2. 将 node_def 依次加入  fdef.node_def
3. 将 fdef.signature.output_arg() 加入 fdef.ret (约束条件：TODO)

bool RegisterOp(const string& op, Creator func)

初始化 factory, 将 {op, func} 加入

Status GetOpGradientCreator(const string& op, Creator* creator)

从 factory 中查找 op 对应的 Creator, 如果找到返回，找不到错误


### FunctionLibraryRuntimeImpl

const FunctionBody* FunctionLibraryRuntimeImpl::GetFunctionBody(Handle h) // 返回 func_graphs_[h]

Status FunctionLibraryRuntimeImpl::CreateKernel(const NodeDef& ndef, OpKernel** kernel)

1. 如果 custom_kernel_creator_ 不为空，custom_kernel_creator_(this, ndef, &ret); kernel = ret
2. 如果 lib_def_ 中找不到  ndef.op() 对应的 FunctionDef，调用 CreateNonCachedKernel(device_, this, ndef, graph_def_version_, kernel) 并返回; 否则，继续
3. 创建  OpKernelConstruction  对象，调用 CallOp(handle, &construction) 构造函数初始化 kernel

Status FunctionLibraryRuntimeImpl::FunctionDefToBody(const FunctionDef& fdef, AttrSlice attrs, FunctionBody** fbody)

用  fdef, attrs, lib_def_ 初始化 fbody

Status FunctionLibraryRuntimeImpl::InstantiateSymbolicGradient(const NameAttrList& func, FunctionBody** g_body)

1. 如果在 lib_def_ 中找到 func 对应的 fdef, 调用 g_body = SymbolicGradient(f_body);
2. 否则获取 func.name 和 attr 对应的 Creator , 调用 creator(AttrSlice(&func.attr()), &grad_fdef), 之后将  grad_fdef 转为  g_body

Status FunctionLibraryRuntimeImpl::Instantiate(const string& function_name, AttrSlice attrs, Handle* handle)

1. 获取  fbody
2. 如果  table_ 中已经存在 function_name, attrs 对应的 fbody 什么也不做
3. 如果  table_ 中不存在 function_name, attrs 对应的  将其加入 table_, func_graphs_

Status FunctionLibraryRuntimeImpl::Instantiate(const string& function_name, AttrSlice attrs, Handle* handle)

从  table_ 中查找  function_name 和 attrs 对应的 Handle 初始化  handle


void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g)

   调用 optimizer.Optimize(lib, lib->env(), lib->device(), g, /*shape_map=*/nullptr); 包括常量展开，子表达式消除，函数内联

Status FunctionLibraryRuntimeImpl::CreateItem(Handle handle, Item** item)

1. 找到函数 body，创建图
2. 进行优化，包括常量展开，子表达式消除，函数内联
3. 创建执行器
4. 初始化 item

Status FunctionLibraryRuntimeImpl::GetOrCreateItem(Handle handle, Item** item)

如果  items_ 中已经存在 handle 对应的 item, 返回
如果  items_ 中不存在 handle 对应的 item,  创建之后，加入 items_

void FunctionLibraryRuntimeImpl::Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets, DoneCallback done)

1. 获取 fbody
2. 获取 frame
3. 创建 item
4. 初始化 executor 参数
5. 运行执行器

bool FunctionLibraryRuntimeImpl::IsStateful(const string& func)

func 对应的 OpDef 是否是  stateful 的

Status FunctionDefToBodyHelper(const FunctionDef& fdef, const AttrSlice& attrs,
    const FunctionLibraryDefinition* const lib_def,
    const std::function<Status(const string&, const OpDef**)>& get_func_sig,
    FunctionBody** fbody)

用  fdef, attrs, lib_def, 初始化 fbody

bool RemoveIdentityNodes(Graph* g)

将 g 中的 Identity 节点删除，并将其输入节点与输出节点重新建立连接

bool RemoveListArrayConverter(Graph* g)

 将 g 中的 `_ListToArray` `_ArrayToList` 节点删除

static bool ValidateInlining(const Node* node, const FunctionBody* fbody)

当 node 的输入输出节点的数量和类型与  fbody 的 arg_type 和 ret_type 分别相同时，表明可以内联.

void InlineFunctionBody(const FunctionLibraryDefinition& flib_def, Graph* g, Node* caller, const FunctionBody* fbody)

将  caller 和  flib_def 构建图 g

bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph)

将 graph 中的的可以内联的节点进行内联

void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty)

反向广度遍历 g 的所有节点， 如果节点是 Op 节点修改节点的名称(增加 pretty)，
如果某个节点的前一个节点也是 Op 节点，就该 input endge 名称(增加  pretty)。

### FunctionBody

FunctionBody::FunctionBody(const FunctionDef& f, DataTypeSlice arg_t, DataTypeSlice ret_t, Graph* g)

0. 用 f 初始化 f
1. 用 arg_t 初始化 arg_types
2. 用 ret_t 初始化 ret_types
3. 遍历 this.graph 的所有节点，将 op 为 `_Retval` 的节点加入 ret_nodes 中，将 op 为 `_Arg` 的节点加入 arg_nodes

### SymbolicGradientHelper

void SymbolicGradientHelper::Copy()

将 fbody_ 拷贝到 gbody_

FunctionBody* SymbolicGradientHelper::Compute()

计算  Gradient，此函数重要

FunctionBody* SymbolicGradient(const FunctionBody& f)

对 f  进行 gradient 计算



## 附录

### 每个 Op 的说明

1. ReshapeGrad

```cpp
  /*
   * [T: type](x: T, shape: int32, dy: T) -> (dx:T, dshape: int32) {
   *     x_shape = Shape[T:$T](x)
   *     dx = Reshape[T:$T](dy, x_shape)
   *     dshape = ZerosLike[T:DT_INT32](shape)
   * }
   */

  *g = FDH::Define(
      // Arg defs
      {"x: T", "shape: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dshape: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
        {{"dshape"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
      });
```

2. SqueezeGrad
``` cpp

  /*
   * [T:type](x: T, dy: T) -> (dx: T) {
   *    x_shape = Shape[T: $T](x)
   *    dx = Reshape[T: $T](dy, x_shape)
   * }
   */

  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
      });
```
3. IdentityGrad

```cpp
  /*
   * [T: type](x: T, dy: T) -> (dx: T) {
   *   dx = Identity[T: $T](dy)
   * }
   */

  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"dx"}, "Identity", {"dy"}, {{"T", "$T"}}},
      });
```

PackGrad


```
  /*
   * _[T: type, N: int, axis: int](x: N*T, dy: T) -> (dx: N*T) {
   *   dx = Unpack[T:$T, num:$N, axis: $axis](dy)
   *   return dx = dx:output
   * }
   */
  *g = FDH::Create(
      "_",
      // Arg defs
      {"x: N*T", "dy: T"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int", "axis: int"},
      // Nodes
      {
        {
          {"dx"},
          "Unpack",
          {"dy"},
          {{"T", "$T"}, {"num", "$N"}, {"axis", "$axis"}}
        },
      },
      {{"dx", "dx:output"}});
```

ConcatGrad

```
    *g = FDH::Create(
        "_",
        // Arg defs
        {"dim: int32", "x: N*T", "dy: T"},
        // Return signature
        {"d_dim: int32", "dx: N*T"},
        // Attr defs
        {"T: type", "N: int"},
        // Nodes
        nodes,
        // Return values
        {{"dx", "dx:output"}, {"d_dim", "d_dim:y:0"}});
```

ConcatGradV2

```
    *g = FDH::Create(
        "_",
        // Arg defs
        {"x: N*T", "dim: int32", "dy: T"},
        // Return signature
        {"dx: N*T", "d_dim: int32"},
        // Attr defs
        {"T: type", "N: int"},
        // Nodes
        nodes,
        // Return values
        {{"dx", "dx:output"}, {"d_dim", "d_dim:y:0"}});
```

SplitGrad

```
  *g = FDH::Define(
      // Arg defs
      {"dim: int32", "x: T", "dy: num_split*T"},
      // Ret val defs
      {"d_dim: int32", "dx: T"},
      // Attr defs
      {"T: type", "num_split: int"},
      // Nodes
      {
        {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
        {{"dx"}, "Concat", {"dim", "dy"}, {{"T", "$T"}, {"N", "$num_split"}}}
      });
```

ArrayToListGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: N*T", "dy: out_types"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int", "out_types: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ListToArray", dys,
         {{"T", "$T"}, {"N", "$N"}, {"Tin", "$out_types"}}}
      });
```

ListToArrayGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: Tin", "dy: N*T"},
      // Ret val defs
      {"dx: Tin"},
      // Attr defs
      {"T: type", "N: int", "Tin: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ArrayToList", {"dy"},
         {{"T", "$T"}, {"N", "$N"}, {"out_types", "$Tin"}}}
      });
```

FillGrad

```
  *g = FDH::Define(
      // Arg defs
      {"dims: int32", "x: T", "dy: T"},
      // Ret val defs
      {"d_dims: int32", "dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"d_dims"}, "ZerosLike", {"dims"}, {{"T", DT_INT32}}},
          FDH::Const("zero", 0),
          {{"rank"}, "Rank", {"dy"}, {{"T", "$T"}}},
          FDH::Const("one", 1),
          {{"r"}, "Range", {"zero", "rank", "one"}, {}},
          // dx = sum(dy)
          {{"dx"}, "Sum", {"dy", "r"}, {{"T", "$T"}}},
      });
```

TransposeGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: T", "p: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dp: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"q"}, "InvertPermutation", {"p"}, {}},
          {{"dx"}, "Transpose", {"dy", "q"}, {{"T", "$T"}}},
          {{"dp"}, "ZerosLike", {"p"}, {{"T", DT_INT32}}},
      });
```

ReverseGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: T", "d: bool", "dy: T"},
      // Ret val defs
      {"dx: T", "dd: bool"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"dx"}, "Reverse", {"dy", "d"}, {{"T", "$T"}}},
          {{"dd"}, "ZerosLike", {"d"}, {{"T", DT_BOOL}}},
      });
```

ReverseV2Grad

```
  *g = FDH::Define(
      // Arg defs
      {"x: T", "d: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dd: int32"},
      // Attr defs
      {"T: type", "Tidx: {int32, int64}"},
      // Nodes
      {
          {{"dx"}, "ReverseV2", {"dy", "d"}, {{"T", "$T"}}},
          {{"dd"}, "ZerosLike", {"d"}, {{"T", "$Tidx"}}},
      });
```

SliceGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: T", "begin: int32", "size: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "begin_grad: int32", "size_grad: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {// paddings = concat(1, [begin, shape(x) - begin - size])
       FDH::Const("one", 1),
       {{"b1"}, "ExpandDims", {"begin", "one"}, {{"T", DT_INT32}}},
       {{"xs"}, "Shape", {"x"}, {{"T", "$T"}}},
       {{"xs_b"}, "Sub", {"xs", "begin"}, {{"T", DT_INT32}}},
       {{"xs_b_s"}, "Sub", {"xs_b", "size"}, {{"T", DT_INT32}}},
       {{"a1"}, "ExpandDims", {"xs_b_s", "one"}, {{"T", DT_INT32}}},
       {{"paddings"},
        "Concat",
        {"one", "b1", "a1"},
        {{"N", 2}, {"T", DT_INT32}}},
       // dx = Pad(dy, paddings)
       {{"dx"}, "Pad", {"dy", "paddings"}, {{"T", "$T"}}},
       {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
       {{"size_grad"}, "ZerosLike", {"size"}, {{"T", DT_INT32}}}});
```
StridedSliceGrad

```
  *g = FDH::Define(
      // Arg defs
      {"x: T", "begin: int32", "end: int32", "stride: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "begin_grad: int32", "end_grad: int32", "stride_grad: int32"},
      // Attr defs
      {"T: type", "Index: {int32, int64}", "begin_mask: int", "end_mask: int",
       "ellipsis_mask: int", "new_axis_mask: int", "shrink_axis_mask: int"},
      {// Nodes
       {{{"xs"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"},
         "StridedSliceGrad",
         {"xs", "begin", "end", "stride", "dy"},
         {{"T", "$T"},
          {"Index", "$Index"},
          {"begin_mask", "$begin_mask"},
          {"end_mask", "$end_mask"},
          {"ellipsis_mask", "$ellipsis_mask"},
          {"new_axis_mask", "$new_axis_mask"},
          {"shrink_axis_mask", "$shrink_axis_mask"}}},
        {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
        {{"end_grad"}, "ZerosLike", {"end"}, {{"T", DT_INT32}}},
        {{"stride_grad"}, "ZerosLike", {"stride"}, {{"T", DT_INT32}}}}});
```

StridedSliceGradGrad

```
  *g = FDH::Define(
      // Arg defs
      {"shape: int32", "begin: int32", "end: int32", "stride: int32", "dy: T",
       "grad: T"},
      // Ret val defs
      {"shape_grad: int32", "begin_grad: int32", "end_grad: int32",
       "stride_grad: int32", "dy_grad: T"},
      // Attr defs
      {"T: type", "Index: {int32, int64}", "begin_mask: int", "end_mask: int",
       "ellipsis_mask: int", "new_axis_mask: int", "shrink_axis_mask: int"},
      {// Nodes
       {{{"shape_grad"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
        {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
        {{"end_grad"}, "ZerosLike", {"end"}, {{"T", DT_INT32}}},
        {{"stride_grad"}, "ZerosLike", {"stride"}, {{"T", DT_INT32}}},
        {{"dy_grad"},
         "StridedSlice",
         {"grad", "begin", "end", "stride"},
         {{"T", "$T"},
          {"Index", "$Index"},
          {"begin_mask", "$begin_mask"},
          {"end_mask", "$end_mask"},
          {"ellipsis_mask", "$ellipsis_mask"},
          {"new_axis_mask", "$new_axis_mask"},
          {"shrink_axis_mask", "$shrink_axis_mask"}}}}});
```

