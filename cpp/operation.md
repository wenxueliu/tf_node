

Op: Compute -> Function:Run -> Execotur:Run -> Device:Compute


 1. 根据 NodeDef 可以找到 OpDef 的 name
 2. 根据 OpDef 的 name, 通过 global_op_registry 的 OpRegistry::Global()->LookUp 找到 OpRegistrationData
 3. 通过 OpRegistrationData 可以找到 OpDef 和 OpShapeInferenceFn

 1. 根据 DeviceType 和 NodeDef  通过 global_kernel_registry 的 GlobalKernelRegistryTyped() 可以找到 KernelRegistration
 2. 通过 KernelRegistration 可以找到 KernelDef, kernel_class_name, OpKernelRegistrar::Factory
 3. 根据 DeviceType 和 NodeDef  通过 FindKernelDef 可以找到 KernelDef

 1. 根据 NodeDef 可以通过  SupportedDeviceTypesForNode 找到 NodeDef 支持的设备类型

主要用于 FunctionLibraryDefinition library(OpRegistry::Global(), {})  和 Graph graph(OpRegistry::Global())

每个 Operation 通过 REGISTER_OP 注册到全局的 global_op_registry 中， 通过
LookUpOpDef 就可以查找到该 Operation. 然后定义该 Operation 的 input, output,
attr，定义该操作的回到函数，该函数解析 input, output 进行处理。


struct XlaDeviceOpRegistrations
    std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>> op_kernel_registrars;

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels()
        XlaOpRegistry::RegisterCompilationKernels()
            new kernel_factory::OpKernelRegistrar(new KernelDef(*kdef), "XlaJitOp", op.second->factory));
        new kernel_factory::OpKernelRegistrar(def, "XlaDeviceDummyOp", dummy_factory));

CompileGraph()
    ConvertGraphToXla()
        XlaOpRegistry::RegisterCompilationKernels()

static bool register_me = RegisterLaunchOpCreator();
        RegisterLaunchOpCreator()
            CreateXlaLaunchOp()

Status MarkForCompilationPass::Run()
    Status MarkForCompilationPass::RunImpl()
        XlaOpRegistry::RegisterCompilationKernels()

ConvertTfGraphToXlaSessionModule()
    SetupXlaCpuClient()
        XlaOpRegistry::RegisterCompilationKernels()

## 使用流程

1. 注册一个 Operation
2. 注册一个 OpKernel
3. 创建一个 OpKernel

```cpp
class DummyKernel : public tensorflow::OpKernel {
 public:
  explicit DummyKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(tensorflow::OpKernelContext* context) override {}
};

REGISTER_OP("Test1").Input("a: Ti").Input("b: Ti").Output("o: To").Attr("Ti: type").Attr("To: type");;
/*
 * "op: 'Test1' device_type: 'CPU'
 * constraint { name: 'Ti' allowed_values { list { type: [ DT_INT8 ] } } }
 * constraint { name: 'To' allowed_values { list { type: [ DT_INT8 ] } } }
 * host_memory_arg: ['a', 'b']
 */
REGISTER_KERNEL_BUILDER(Name("Test1").Device(tensorflow::DEVICE_CPU)
                        .HostMemory("a").HostMemory("b")
                        .TypeConstraint<int8>("Ti").TypeConstraint<int8>("To"),
                        DummyKernel);
REGISTER_KERNEL_BUILDER(Name("Test1").Device(tensorflow::DEVICE_GPU)
                        .HostMemory("a").HostMemory("b")
                        .TypeConstraint<float>("Ti").TypeConstraint<float>("To"),
                        DummyKernel);

NodeDef node_def
NodeDefBuilder builder("Test1" + "-op", "Test1").Input(DT_INT8).Input(DT_INT8).Finalize(&node_def)
NodeDefBuilder builder("Test1" + "-op", "Test1").Input(DT_FLOAT).Input(DT_FLOAT).Finalize(&node_def)
DeviceBase device_(Env::Default())
Status status
std::unique_ptr<OpKernel> op(CreateOpKernel(
    std::move(DEVICE_CPU), &device_, cpu_allocator(),
    node_def, TF_GRAPH_DEF_VERSION, &status));

REGISTER_OP("BuildTypeAttr").Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("BuildTypeAttr")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        DummyKernel);

NodeDef def
def.set_name("BuildTypeAttr-op")
def.set_op("BuildTypeAttr")
AttrValue attr_value;
ParseAttrValue("type", "DT_FLOAT", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("T", attr_value))
DeviceBase device(Env::Default());
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));

REGISTER_OP("BuildTypeListAttr").Attr("T: list(type)");
REGISTER_KERNEL_BUILDER(Name("BuildTypeListAttr")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bool>("T"),
                        DummyKernel);
NodeDef def
def.set_name("BuildTypeListAttr-op")
def.set_op("BuildTypeListAttr")
AttrValue attr_value;
//ParseAttrValue("list(type)", "[DT_BOOL]", &attr_value)
ParseAttrValue("list(type)", "[DT_BOOL DT_BOOL]", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("T", attr_value))
DeviceBase device(Env::Default());
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));

REGISTER_OP("ListOut").Output("a: int32").Output("b: T").Attr("T: list(type)");
REGISTER_KERNEL_BUILDER(Name("ListOut").Device(tensorflow::DEVICE_CPU),
                        DummyKernel);

Env* env = Env::Default();
OpKernelContext::Params params;
params.record_tensor_accesses = false;
std::unique_ptr<DummyDevice> device(new DummyDevice(env, params.record_tensor_accesses));
params.device = device.get();
Status status;
NodeDef def
def.set_name("ListOut-op")
def.set_op("ListOut")
AttrValue attr_value;
//ParseAttrValue("list(type)", "[DT_BOOL]", &attr_value)
ParseAttrValue("list(type)", "[DT_FLOAT, DT_INT32]", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("T", attr_value))
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
    def, TF_GRAPH_DEF_VERSION, &status));
params.op_kernel = op.get();
gtl::InlinedVector<TensorValue, 4> inputs{};
params.inputs = &inputs;
std::unique_ptr<OpKernelContext> ctx(new OpKernelContext(&params));

REGISTER_OP("GetAttrStringList")
    .Attr("attr_name: string")
    .Attr("a: list(string)");
REGISTER_KERNEL_BUILDER(Name("GetAttrStringList").Device(DEVICE_CPU), GetAttrKernel);
NodeDef def
def.set_name("GetAttrStringList-op")
def.set_op("GetAttrStringList")

AttrValue attr_value;
ParseAttrValue("string", "a", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("attr_name", attr_value))
AttrValue attr_value;
ParseAttrValue("list(string)", "['foo', 'bar']", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("a", attr_value))
DeviceBase device(Env::Default());
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));

REGISTER_OP("GetAttrInt")
    .Attr("attr_name: string")
    .Attr("a: int")
    .Attr("b: list(int)");
REGISTER_KERNEL_BUILDER(Name("GetAttrInt").Device(DEVICE_CPU), GetAttrKernel);
NodeDef def
def.set_name("GetAttrInt-op")
def.set_op("GetAttrInt")

AttrValue attr_value;
ParseAttrValue("string", "a", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("attr_name", attr_value))

AttrValue attr_value;
ParseAttrValue("int", "35", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("a", attr_value))

AttrValue attr_value;
ParseAttrValue("list(int)", "[-1, 2, 4]", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("b", attr_value))
DeviceBase device(Env::Default());
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));

REGISTER_OP("GetAttrShape")
    .Attr("attr_name: string")
    .Attr("a: shape")
    .Attr("b: list(shape)");
REGISTER_KERNEL_BUILDER(Name("GetAttrShape").Device(DEVICE_CPU), GetAttrKernel);

NodeDef def
def.set_name("GetAttrShape-op")
def.set_op("GetAttrShape")

AttrValue attr_value;
ParseAttrValue("string", "a", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("attr_name", attr_value))

AttrValue attr_value;
ParseAttrValue("int", "35", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("a", attr_value))

AttrValue attr_value;
ParseAttrValue("list(int)", "[-1, 2, 4]", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("b", attr_value))
DeviceBase device(Env::Default());
std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));

AttrValue attr_value;
ParseAttrValue("string", "a", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("attr_name", attr_value))

AttrValue attr_value;
ParseAttrValue("shape", "{ dim { size: 3 } }", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("a", attr_value))

AttrValue attr_value;
ParseAttrValue("list(shape)", "[{ dim { size:2 } }, { dim { size: 4 } }]", &attr_value)
def.mutable_attr()->insert(AttrValueMap::value_type("a", attr_value))

REGISTER_OP("LabeledKernel");
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU),
                        LabeledKernel<0>);
REGISTER_KERNEL_BUILDER(Name("LabeledKernel").Device(DEVICE_CPU).Label("one"),
                        LabeledKernel<1>);

  std::unique_ptr<OpKernel> op_kernel =
      ExpectSuccess("LabeledKernel", DEVICE_CPU, {"_kernel|string|'one'"});
  auto* get_labeled_kernel = static_cast<BaseKernel*>(op_kernel.get());
  EXPECT_EQ("LabeledKernel<1>", GetKernelClassName("LabeledKernel", DEVICE_CPU,
                                                   {"_kernel|string|'one'"}));

REGISTER_OP("Same")
    .Input("a: int32")
    .Input("b: T")
    .Input("c: N * int32")
    .Input("d: N * T")
    .Input("e: TList")
    .Output("ndef: string")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("TList: list(type)");
REGISTER_KERNEL_BUILDER(Name("Same").Device(DEVICE_CPU), TestKernel);
                                                ```



注册一个 Operation Test1, 输入包含 float 类型的 a 和 int32 类型的 b, 输出 uint8 类型的 o 以及属性 T
注册一个 OpKernel Test1, 类型是 CPU,  并指定创建 OpKernel 的回调函数为 DummyKernel, 并且输入 a 保存在 host memory, 限制 Ti 的类型为 int8, To 类型为 int8
创建一个 OpKernel，该 OpKernel 由 DummyKernel(context) 生成

注意点:
1. REGISTER_OP 和 REGISTER_KERNEL_BUILDER 中 Test1 必须一致
2. 通过 Device 指定运行在 CPU 还是 GPU
3. 通过 HostMemory 指定某些属性运行在 host memory, 默认运行在 device memory
4. DummyKernel 这个函数对象的格式，已经必须返回 OpKernel, 因此，继承 OpKernel 类是一个不错的选择

### 注册一个 Operation 流程

REGISTER_OP(name)

  static ::tensorflow::register_op::OpDefBuilderReceiver register_op__COUNTER__ = ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP(name)>(name)

REGISTER_SYSTEM_OP(name)

  static ::tensorflow::register_op::OpDefBuilderReceiver register_op__COUNTER__ = ::tensorflow::register_op::OpDefBuilderWrapper<true>(name)

1. 调用 OpDefBuilderReceiver 的构造函数初始化 register_op__COUNTER__
2. 构造 OpDefBuilderWrapper 对象 wrapper
3. 创建函数对象 wrapper.builder().Finalize，接收 OpRegistrationData 类型参数，返回 Status
4. 创建全局变量 global_op_registry = new OpRegistry;
5. 调用 global_op_registry->Register()，
6. 调用 global_op_registry->RegisterAlreadyLocked()
6.1 构造 OpRegistrationData 对象 op_reg_data
6.2 调用 OpDefBuilder.Finalize(op_reg_data)
6.3 ValidateOpDef(op_reg_data->op_def);
6.4 将 op_reg_data->op_def.name(): op_reg_data 加入 global_op_registry.registry_

至此，在 global_op_registry.registry_ 中保存了对应的 op_def, 可以通过 LookUpOpDef, LookUp 查找,
通过 GetRegisteredOps, Export 导出对应的 op_def

```
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
}
```

### 注册 OpKernel 的过程

```
REGISTER_KERNEL_BUILDER(kernel_builder, ...)
  constexpr bool should_register___COUNTER__flag = SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__);
  static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body____COUNTER____object(
          should_register___COUNTER__flag ? ::tensorflow::register_kernel::kernel_builder.Build() : nullptr,
          #__VA_ARGS__,
          [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new __VA_ARGS__(context); });


REGISTER_KERNEL_BUILDER(kernel_builder, ...)
  static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body____COUNTER____object(
          ::tensorflow::register_kernel::system::kernel_builder.Build()
          #__VA_ARGS__,
          [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new __VA_ARGS__(context); });
```

实例

REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp)

实现

1. 首先创建并初始化 kernel_builder, 将 kernel_builder 转换为 kernel_def
2. 定义函数对象，入参 OpKernelConstruction, 返回 OpKernel
3. 调用 OpKernelRegistrar 构造函数
3.1 调用 OpKernelRegistrar::InitInternal 函数
3.1.1 构造 key = "kernel_def->op():DeviceType(kernel_def->device_type()):kernel_def->label()"
3.1.2 初始化全局变量 global_kernel_registry = new KernelRegistry;
3.1.3 将 key, KernelRegistration(*kernel_def, kernel_class_name, factory) 加入 global_kernel_registry

```
typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry;

void* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
  return reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
}
```
FindKernelRegistration(DeviceType& device_type, NodeDef& node_def, KernelRegistration** reg, bool* was_attr_mismatch)

从 global_op_registry 中查找 device_type, node_def 对应的 KernelRegistration

FindKernelDef(DeviceType& device_type, NodeDef& node_def, KernelDef** def, string* kernel_class_name)

从 global_op_registry 中查找 device_type, node_def 对应的 KernelRegistration, 并设置 def，kernel_class_name

### 创建 OpKernel

所有操作起于 CreateOpKernel("CPU", device_.get(), device_->GetAllocator({}), node, TF_GRAPH_DEF_VERSION, &status);

已知参数

* DeviceType device_type
* DeviceBase* device,
* Allocator* allocator
* FunctionLibraryRuntime* flib,
* const NodeDef& node_def
* int graph_def_version,
* OpKernel** kernel

CreateOpKernel()
  OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
  ValidateNodeDef(node_def, *op_def);
  FindKernelRegistration(device_type, node_def, &registration, &was_attr_mismatch);
    GlobalKernelRegistryTyped()->equal_range(key)
    AttrsMatch(node_def, iter->second.def, &match)
  InOutTypesForNode(node_def, *op_def, &inputs, &outputs)
  MemoryTypesForNode(OpRegistry::Global(), device_type, node_def, &input_memory_types, &output_memory_types)
  OpKernelConstruction context(device_type, device, allocator, &node_def, op_def, flib, inputs, input_memory_types, outputs, output_memory_types, graph_def_version, &s);
  *kernel = (*registration->factory)(&context);

1. 从 global_op_registry.registry_ 中查找 node_def.op() 对应的 op_def
2. 校验 node_def 与 op_def 的有效性。
2.1 node_def 的 input 命名合法
2.2 node_def 的 attr 与 op_def 所有的 attr 相同
2.3 node_def 的 input 合法, 数量与 op_def 相同，且
3. 从 global_kernel_registry 中查找 "node_def.op():device_type:node_def.attr(kKernelAttr]" 对应的 KernelRegistration
4. 遍历 op_def 的 input_arg, output_arg 从 node_def 中找到对应的类型，加入 inputs, outputs
5. TODO
6. 构造 OpKernelConstruction
7. 调用 REGISTER_KERNEL_BUILDER 时，第二个参数指定的函数

例如，对于 REGISTER_KERNEL_BUILDER(Name("Variable").Device(DEVICE_CPU), VariableOp), 会调用 VariableOp(context)

### 如何判断一个 NodeDef 是合法的

TODO
OpDef::ArgDef number_attr(), type_attr(), type()


## 源文件

op.cc
op.h
op_def_builder.cc
op_def_builder.h
kernel_def_builder.h
kernel_def_builder.cc

## 数据结构

核心数据结构

### 核心变量

global_op_registry
global_kernel_registry

### 核心类





```
message AttrValue {
  message ListValue {
    repeated bytes s = 2;                        // "list(string)"
    repeated int64 i = 3 [packed = true];        // "list(int)"
    repeated float f = 4 [packed = true];        // "list(float)"
    repeated bool b = 5 [packed = true];         // "list(bool)"
    repeated DataType type = 6 [packed = true];  // "list(type)"
    repeated TensorShapeProto shape = 7;         // "list(shape)"
    repeated TensorProto tensor = 8;             // "list(tensor)"
    repeated NameAttrList func = 9;              // "list(attr)"
  }
  oneof value {
    bytes s = 2;                 // "string"
    int64 i = 3;                 // "int"
    float f = 4;                 // "float"
    bool b = 5;                  // "bool"
    DataType type = 6;           // "type"
    TensorShapeProto shape = 7;  // "shape"
    TensorProto tensor = 8;      // "tensor"
    ListValue list = 1;          // any "list(...)"
    NameAttrList func = 10;
    string placeholder = 9;
  }
}

message OpDef {
  string name = 1;

  //number_attr  不为 “” 时，必须是数字，指定了属性的个数, 具体属性可以为type_attr 或 type
  //当 number_attr 为 “”  时，属性是 type_attr 或 type 或 type_list_attr
  message ArgDef {
    string name = 1; //input, output 或 attr 的名称
    string description = 2;
    //type, type_attr, number_attr, type_list_attr 必须有一个不为空
    DataType type = 3;
    string type_attr = 4;    // if specified, attr must have type "type"
    string number_attr = 5;  // if specified, attr must have type "int"
    string type_list_attr = 6;
    bool is_ref = 16; //表明所有的类型是否是引用
  };
  repeated ArgDef input_arg = 2;
  repeated ArgDef output_arg = 3;
  message AttrDef {
    string name = 1;
    string type = 2;
    AttrValue default_value = 3;
    string description = 4;
    bool has_minimum = 5; //如果为 true, minimum 不能为 0
    int64 minimum = 6;
    AttrValue allowed_values = 7;
  }
  repeated AttrDef attr = 4;
  OpDeprecation deprecation = 8;
  string summary = 5;
  string description = 6;
  bool is_commutative = 18;
  bool is_aggregate = 16;  // for things like add
  bool is_stateful = 17;  // for things like variables, queue
  bool allows_uninitialized_input = 19;  // for Assign, etc.
}

例子参考 op_def_builder_test.cc op_def_util_test.cc op_kernel_test.cc op_compatibility_test.cc

message KernelDef {
  string op = 1; // OpDef 的名称, 必须与 OpDef 的 name 相同
  string device_type = 2; //DEVICE_CPU 或 DEVICE_GPU
  message AttrConstraint {
    string name = 1; //OpDef 的属性名
    AttrValue allowed_values = 2; //OpDef 中属性 name 允许的类型
  }
  repeated AttrConstraint constraint = 3; //对 Op 的属性的类型进行限制
  repeated string host_memory_arg = 4; //默认 OpDef 的属性是保存在 device_memory 的，当指定该参数时，对应的属性不管是 input 或 output 保存在 host memory，保存的内存是 input 和 output 的 index(即该节点的第几个输入或输出)
  // This allows experimental kernels to be registered for an op that
  // won't be used unless the user specifies a "_kernel" attr with
  // value matching this.
  string label = 5; //op_def 或 node_def 的 `_kernal` 属性对应的 value
}

例子参考 kernel_def_builder_test.cc op_kernel_test.cc op_compatibility_test.cc

message OpDeprecation {
  int32 version = 1;
  string explanation = 2;
};

message OpList {
  repeated OpDef op = 1;
};
```

class KernelDefBuilder
  KernelDef* kernel_def_;

class OpRegistry
  mutable mutex mu_;
  mutable std::vector<OpRegistrationDataFactory> deferred_
  mutable std::unordered_map<string, const OpRegistrationData*> registry_ //op_reg_data->op_def.name():op_reg_data
  mutable bool initialized_ //是否已经初始化, 也就是 deferred_ 是否被处理
  mutable Watcher watcher_

struct OpRegistrationData
  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
  typedef std::function<Status(shape_inference::InferenceContext* c)> OpShapeInferenceFn;

class OpListOpRegistry : public OpRegistryInterface
  std::unordered_map<string, const OpRegistrationData*> index_; //op_reg_data->op_def.name():op_reg_data

class OpDefBuilderWrapper<true>
  mutable ::tensorflow::OpDefBuilder builder_;

class OpDefBuilder
  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  string doc_;
  std::vector<string> errors_;

  op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);

struct OpRegistrationData
  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;

实现代理给 op_reg_data_.op_def, Attr, Input, Output 目的都是简化 op_def 的操作

struct KernelRegistration
  const KernelDef def;
  const string kernel_class_name;
  const kernel_factory::OpKernelRegistrar::Factory factory;

class OpKernelConstruction
  const DeviceType device_type_;
  DeviceBase* const device_;
  Allocator* allocator_;
  const NodeDef* def_;
  const OpDef* op_def_;
  FunctionLibraryRuntime* flib_;
  DataTypeSlice input_types_;
  MemoryTypeSlice input_memory_types_;
  DataTypeSlice output_types_;
  MemoryTypeSlice output_memory_types_;
  const int graph_def_version_;
  Status* status_;

class OpKernel
  const std::unique_ptr<const NodeDef> def_;
  const DataTypeVector input_types_;
  const MemoryTypeVector input_memory_types_;
  const DataTypeVector output_types_;
  const MemoryTypeVector output_memory_types_;
  const int graph_def_version_;
  const bool is_internal_;  // True if this is an internal operation
  NameRangeMap input_name_map_; //由 OpKernelConstruction 的 op_def_->input_arg() 初始化
  NameRangeMap output_name_map_; //OpKernelConstruction 的 op_def_->ouput_arg() 初始化
  bool expensive_;

class OpKernelContext
  typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;
  Status status_;
  Params* params_;    // not owned
  mutable mutex mu_;  // mutable so const accessors can acquire the lock
  gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators_ //保存 Allocator:TrackingAlloctor 的 pair 对
  gtl::InlinedVector<TensorValue, 4> outputs_;
  // Constructed only if <params->record_tensor_accesses>.
  ManualConstructor<UniqueTensorReferences> referenced_tensors_; //
  bool is_output_dead_ = false;
  int64 host_temp_memory_size_; //记录CPU 分配内存大小 参见 allocate_temp
  int64 device_temp_memory_size_;//记录GPU 分配内存大小 参见 allocate_temp
  gtl::InlinedVector<int64, 2> host_persistent_alloc_ids_; //record_host_persistent_memory_allocation
  gtl::InlinedVector<int64, 2> device_persistent_alloc_ids_; //record_device_persistent_memory_allocation
  int64 host_persistent_memory_allocated_;
  int64 device_persistent_memory_allocated_;

  struct Params
    int64 step_id = 0;
    OpKernel* op_kernel = nullptr;
    DeviceBase* device = nullptr;
    PerOpGpuDevice* eigen_gpu_device = nullptr; //避免重新初始化
    bool track_allocations = false;
    bool log_memory = false;
    bool record_tensor_accesses = false;
    // Array indexed by output number for this node
    const AllocatorAttributes* output_attr_array = nullptr;
    // Shared resources accessible by this op kernel invocation.
    ResourceMgr* resource_manager = nullptr;
    // Per-step resources accessible by this op kernel invocation should be
    // stored in this container..
    ScopedStepContainer* step_container = nullptr;
    // Mechanism used by this op kernel invocation to communicate with
    // computations running on other devices.
    Rendezvous* rendezvous = nullptr;
    // The session state for this op.
    SessionState* session_state = nullptr;
    // The tensor store for this op.
    TensorStore* tensor_store = nullptr;
    // Mechanism used by this op kernel invocation to register a callback
    // for its cancellation.
    CancellationManager* cancellation_manager = nullptr;
    // Inputs to this op kernel.
    const gtl::InlinedVector<TensorValue, 4>* inputs = nullptr;
    bool is_input_dead = false;
    const gtl::InlinedVector<AllocatorAttributes, 4>* input_alloc_attrs = nullptr;
    const gtl::InlinedVector<DeviceContext*, 4>* input_device_contexts = nullptr;
    DeviceContext* op_device_context = nullptr;
    FrameAndIter frame_iter; // Control-flow op supports.
    FunctionCallFrame* call_frame = nullptr; // Function call supports.
    FunctionLibraryRuntime* function_library = nullptr;
    std::function<void(std::function<void()>)>* runner = nullptr;
    StepStatsCollector* stats_collector = nullptr;
    checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache = nullptr; // TensorSliceReaderCache support.

struct TensorValue
  mutex* mutex_if_ref;  // nullptr if not a ref, != nullptr if a ref
  Tensor* tensor;

class OpOutputList
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;

class OpInputList
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;

class OpMutableInputList
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;

class PersistentTensor
  Tensor tensor_;

class OpSegment
  typedef std::unordered_map<string, OpKernel*> KernelMap;
  struct Item {
    int num_holds = 1;      //持有 session 的个数
    KernelMap name_kernel;  // op name -> kernel.
    ~Item();
  };
  // session handle -> item.
  // Session handles are produced by strings::FpToString()
  typedef std::unordered_map<string, Item*> SessionMap;
  mutable mutex mu_;
  SessionMap sessions_ GUARDED_BY(mu_);

OpDefBuilderWrapper -> OpDefBuilder -> OpRegistrationData->OpDef -> OpRegistrationData -> OpRegistry

OpKernelConstruction -> OpKernel

KernelDefBuilder -> KernelDef 
KernelRegistry -> KernelRegistration -> KernelDef
                                        kernel_factory::OpKernelRegistrar::Factory

typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry //key 为  KernelDef 的 op() + device_type() + lable() 组成


typedef std::function<Status(shape_inference::InferenceContext* c)> OpShapeInferenceFn;
typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;


```
REGISTER_OP("my_op_name")
    .Attr("<name>:<type>")
    .Attr("<name>:<type>=<default>")
    .Input("<name>:<type-expr>")
    .Input("<name>:Ref(<type-expr>)")
    .Output("<name>:<type-expr>")
    .Doc(R"(
<1-line summary>
<rest of the description (potentially many lines)>
<name-of-attr-input-or-output>: <description of name>
<name-of-attr-input-or-output>: <description of name;
  if long, indent the description on subsequent lines>
)");
```

### 变量

全局变量 \_registered_ops 保存 op_def.name: op_def 键值对，其中 op_def 类型为 op_def_pb2.OpDef

\_InitOpDefLibrary.op_list_ascii 中的所有 op_def 都加入 \_op_def_lib, 注册到 \_registered_ops

Operation 为 TensorFlow Graph 的节点, 必须属于一个 Graph, 每个 Operation 包含输入和输出，
每个 Operation 可以有零个或多个 Tensor 作为输入也可以有零个或多个 Tensor 作为输出

通过 tf.matmul 或 tf.Graph.create_op 创建一个 Operation

会将 Operation 的 control_flow_context 加入所属 Grapth 的 control_flow_context

**关键属性**

nodedef : Operation 的属性, 包括 name, device
nodedef.input : 为 [t._as_node_def_input() for t in self._inputs] + ["^%s" % op.name for op in self._control_inputs]
inputs : 每个 inputs 中 Tensor 的 consumer
graph  : 该 Operation 所属的 Graph
output : [Tensor(self, i, output_type) for i, output_type in enumerate(output_types)]
input_types : 如果为 None, 被设置为 [i.dtype.base_dtype for i in self._inputs]
control_inputs : 里面的元素必须是 Operation 或 (Tensor, IndexedSlices)
c_op : 对应到 C 的 Operation, 是否使用 c Operation 依赖于 graph._c_op

关键操作

* inputs(self)
* outputs(self)
* input_dtypes(self)
* input_types(self)
* control_inputs(self)
* type(self)
* graph(self)
* node_def(self)
* op_def(self)
* traceback(self):
* traceback_with_start_lines(self):

* tf_output(self, output_idx)
* tf_input(self, input_idx):
* add_input(self, tensor, dtype=None):
* update_input(self, index, tensor, dtype=None):
* add_control_inputs(self, ops):
* add_control_input(self, op):
* run(self, feed_dict=None, session=None): 调用之前 graph 必须在 session 中启动

问题: node_def.input 与 input 的关系？

### OpRegistryInterface

class OpRegistryInterface
class OpRegistry : public OpRegistryInterface
class OpListOpRegistry : public OpRegistryInterface

### OpRegistry
``` cpp
   OpRegistry::Global()->Register(
     [](OpRegistrationData* op_reg_data)->Status {
       // Populate op_reg_data here.
       return Status::OK();
   });
```
OpRegistry* OpRegistry::Global()

  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;

  主要用于 FunctionLibraryDefinition library(OpRegistry::Global(), {})  和 Graph graph(OpRegistry::Global())

void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory)

如果 initialized_ 为 true, 待注册的 Operation 加入 registry_
如果 initialized_ 为 false, 待注册的 Operation 加入 deferred_

Status OpRegistry::LookUp(const string& op_type_name, const OpRegistrationData** op_reg_data)

从 registry_ 中找 op_type_name 对应的 OpRegistrationData
如果找到，op_reg_data 指向该 OpRegistrationData
如果没有找到，返回错误信息

void OpRegistry::Export(bool include_internal, OpList* ops)

将 registry_ 排序之后加入 ops

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs)

将 registry_ 中的元素加入 op_defs

Status OpRegistry::SetWatcher(const Watcher& watcher)

设置 watcher_ = watcher

void OpRegistry::DeferRegistrations()

设置 initialized_ 为 false

Status OpRegistry::ProcessRegistrations()

  return CallDeferred();

void OpRegistry::ClearDeferredRegistrations()

清空 deferred_

bool OpRegistry::MustCallDeferred()

如果 initialized_ 为 true, 直接返回
如果 initialized_ 为 false, 将 deferred_ 中的 Operation 加入 registry_, 清空 deferred_

Status OpRegistry::CallDeferred()

如果 initialized_ 为 true, 直接返回
如果 initialized_ 为 false, 将 deferred_ 中的 Operation 加入 registry_, 清空 deferred_

Status OpRegistry::RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)

初始化 OpRegistrationData op_reg_data， 调用 op_data_factory(op_reg_data)，并将其加入 registry_

### OpListOpRegistry

class OpListOpRegistry : public OpRegistryInterface
  std::unordered_map<string, const OpRegistrationData*> index_; //OpDef.name() : new OpRegistrationData()

OpListOpRegistry::OpListOpRegistry(const OpList* op_list)

将 op_list 中的元素依次加入 index_

Status OpListOpRegistry::LookUp(const string& op_type_name, const OpRegistrationData** op_reg_data)

从 index_ 找到 op_type_name 对应的 op_reg_data


### OpDefBuilder

### 用法

  OpDefBuilder("PolymorphicDefaultOut").Output("out: T").Attr("T: type = DT_STRING")
  OpDefBuilder("Binary").Input("a: T").Input("b: T").Output("out: T").Attr("T: type")
  OpDefBuilder("OutTypeList").Output("out: T").Attr("T: list(type) >= 0")
  OpDefBuilder("TypeListRestrict").Input("a: T").Attr("T: list({string, bool}) >= 0")

### Attr

"<name>:<type>", "<name>:<type>=<default>"

name : \[a-zA-Z][a-zA-Z0-9_]*
type :
    "string", "int", "float", "bool", "type", "shape", or "tensor" // "func"
    "numbertype", "realnumbertype", "quantizedtype", "{int32,int64}"
        (meaning "type" with a restriction on valid values)
    "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
        (meaning "string" with a restriction on valid values)
    "list(string)", ..., "list(tensor)", "list(numbertype)", ...
        (meaning lists of the above types)
    "int >= 2" (meaning "int" with a restriction on valid values)
    "list(string) >= 2", "list(int) >= 2"
        (meaning "list(string)" / "list(int)" with length at least 2)

### Input or Output

"<name>:<type-expr>" or "<name>:Ref(<type-expr>)"

name: \[a-zA-Z][a-zA-Z0-9_]*

type-expr

   <type> is either one of "float", "int32", "string", ...
                 or the name of an attr (see above) with type "type".
   <number> is the name of an attr with type "int".
   <type-list> is the name of an attr with type "list(type)".

* For a single tensor: <type>
* For a sequence of tensors with the same type: <number>*<type>
* For a sequence of tensors with different types: <type-list>


## OpKernel

### OpKernelConstruction

class OpKernelConstruction
  const DeviceType device_type_;
  DeviceBase* const device_;
  Allocator* allocator_;
  const NodeDef* def_;
  const OpDef* op_def_;
  FunctionLibraryRuntime* flib_;
  DataTypeSlice input_types_;
  MemoryTypeSlice input_memory_types_;
  DataTypeSlice output_types_;
  MemoryTypeSlice output_memory_types_;
  const int graph_def_version_;
  Status* status_;

class OpKernel
  const std::unique_ptr<const NodeDef> def_;
  const DataTypeVector input_types_;
  const MemoryTypeVector input_memory_types_;
  const DataTypeVector output_types_;
  const MemoryTypeVector output_memory_types_;
  const int graph_def_version_;
  const bool is_internal_;  // True if this is an internal operation
  NameRangeMap input_name_map_; //由 OpKernelConstruction 的 op_def_->input_arg() 初始化
  NameRangeMap output_name_map_; //OpKernelConstruction 的 op_def_->ouput_arg() 初始化
  bool expensive_;

class AsyncOpKernel : public OpKernel
class WhereGPUOp : public AsyncOpKernel //整个核心都在 ComputeAsync
class CallOp : public AsyncOpKernel //整个核心都在 ComputeAsync
  FunctionLibraryRuntime::Handle handle_;

// Register your OpKernel by specifying the Op's name, the device the
// kernel runs on, any type attr constraints for this kernel, any
// host-memory args, and the class to instantiate.  Examples:
//
//  // A kernel that supports all types.
//  REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);
//
//  // The following are equivalent ways of specifying that the kernel only
//  // works if the "T" type attr is set to DT_FLOAT.
//  REGISTER_KERNEL_BUILDER(
//      Name("Sub").Device(DEVICE_CPU).TypeConstraint<float>("T"),
//      SubOp<float>);
//  // (You would then repeat this for every type supported by "Sub".)
//
//  // This form allows you to specify a list of types as the constraint.
//  REGISTER_KERNEL_BUILDER(Name("Sub")
//                              .Device(DEVICE_CPU)
//                              .TypeConstraint("T", {DT_FLOAT}),
//                          SubOp<float>);
//
//  // A kernel that expects one of the input tensors in host memory.
//  REGISTER_KERNEL_BUILDER(
//      Name("Reshape").Device(DEVICE_GPU).HostMemory("shape"), ReshapeOp);
//
// See kernel_def_builder for details.


Status OpKernel::InputRange(StringPiece input_name, int* start, int* stop)

从 input_name_map_ 中查找 input_name 对应的 start, stop

Status OpKernel::OutputRange(StringPiece output_name, int* start, int* stop)

从 output_name_map_ 中查找 output_name 对应的 start, stop

Status OpKernel::MakeShape(const Tensor& shape, TensorShape* out)

创建以 shape 保存在 out


### OpKernelContext

根据 name 找到 input 或 output 的索引, 之后根据索引找到具体的 Tensor

为什么 name 对于 start , stop 的 Operation, 这里可能是一个，也可能是多个

### 源码注解

  // There are three methods to allocate Tensors when an Op kernel
  // executes.
  //
  // 1) allocate_persistent. This is only needed for Tensors that will
  // be stored by the Op between invocations, and it *must* be used
  // for those Tensors. The call returns a PersistentTensor, and that
  // is the only object the Op is allowed to hold on to between
  // invocations. When the Tensor is needed in a subsequent
  // invocation, it can be retrieved from the PersistentTensor using
  // the AccessTensor method. This ensures that the system is made
  // aware of any use of the tensor's allocated memory, which is
  // needed for correctness on asynchronous devices such as GPUs.
  //
  // 2) allocate_output. This should be used to allocate any tensor
  // that is going to be used as an output from the Op at the end of
  // the current execution. The caller indicates which output the
  // Tensor will be assigned to, and the call returns the
  // newly-allocated Tensor. The Tensor can subsequently be assigned
  // to during kernel execution, and will be used as the designated
  // output when the kernel execution completes.
  //
  // 3) allocate_temp. This should be used to allocate any scratch
  // storage that is needed while the kernel is executing, and will
  // not be retained by the Op.
  //
  // In some cases a Tensor needs to be used as an output even though
  // it was previously allocated elsewhere. The Tensor may have been
  // passed as an input, or stored in a PersistentTensor during a
  // previous kernel execution, or allocated earlier in the kernel
  // execution at a time when it was not known which output it would
  // be assigned to. In this case the kernel can use set_output or
  // set_output_ref to indicate that the tensor should be used as the
  // designated output. It is legal to use any previously-allocated
  // Tensor as an argument to set_output or set_output_ref, including
  // Tensors allocated via allocate_temp. There may be a performance
  // penalty to using a Tensor that was not allocated using
  // allocate_output. This is because allocate_output uses the
  // AllocatorAttributes stored in output_attr_array for the
  // designated output. In some cases, using the wrong attributes may
  // cause an extra copy of the Tensor's buffer.

void OpKernelContext::record_tensor_reference(const Tensor& tensor)

  if (params_->record_tensor_accesses)
    really_record_tensor_reference(tensor);

void OpKernelContext::retrieve_accessed_tensors(TensorReferenceVector* out_vector)

    referenced_tensors_->FreezeAndReturnReferences(out_vector);

bool OpKernelContext::has_input(int index)

    inputs[index].tensor != nullptr;

mutex* OpKernelContext::input_ref_mutex(int index)

    inputs[index].mutex_if_ref;

inline DeviceContext* OpKernelContext::input_device_context(int index) {

    input_device_contexts[index];

Allocator* get_allocator(AllocatorAttributes attr);

1. 创建 device->GetStepAllocator(attr, resource_manager())
2. 从 wrapped_allocators_ 中查找对应的 alloctor, 如果找不到，创建新的 TrackingAllocator，并加入 wrapped_allocators_

void really_record_tensor_reference(const Tensor& tensor)

将 tensor 加入 referenced_tensors_

Status input(StringPiece name, const Tensor** tensor);

从 op_kernel 中找到 name 对应的 Tensor

Status input_dtype(StringPiece name, DataType* dtype);

从 op_kernel 中找到 name 对于的 Tensor 的 dtype

mutex* input_ref_mutex(int index)

找到 inputs[index].mutex_if_ref 的锁

Status input_ref_mutex(StringPiece name, mutex** out_mutex);

1. 从 op_kernel 中找到 name 对应的 start 索引
2. inputs[start].mutex_if_ref;

const Tensor& OpKernelContext::input(int index)

返回 inputs[index].tensor

Tensor mutable_input(int index, bool lock_held) {

如果 lock_held 为 true, 不加锁，否则加锁, 返回 inputs[index].tensor

void replace_ref_input(int index, const Tensor& tensor, bool lock_held)

用 tensor 替代 inputs[index].tensor

void forward_ref_input_to_ref_output(int input_index, int output_index)

outputs_[index] = TensorValue(mu, tensor_for_ref);

bool forward_input_to_output_with_shape(int input_index, int output_index,
    const TensorShape& output_shape, Tensor** output)

将 inputs_[input_index] 转移到 output_[output_index]

Status forward_input_to_output_with_shape(StringPiece input_name, StringPiece output_name,
    const TensorShape& output_shape, Tensor** output)

用 input_name 对应的 input Tensor 替代 output_name 对应的 output Tensor

std::unique_ptr<Tensor> forward_input(int input_index, DataType output_dtype,
    const TensorShape& output_shape,
    MemoryType output_memory_type, const AllocatorAttributes& output_attr)

将 inputs[input_index] 的拷贝为 output_ 的 tensor 并返回，前提是 output 与 input
dtype，shape, memory_type 都相同

Status forward_input_or_allocate_temp(gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr, Tensor* out_temp)

从 candidate_input_indices 中找到匹配的 input 替代 out_temp,
如果找不到就分配一个

void delete_ref_input(int index, bool lock_held)

销毁 inputs[index].tensor;

Status mutable_input(StringPiece name, Tensor* tensor, bool lock_held)

找到 name 对应的 Tensor 保存在 tensor

Status replace_ref_input(StringPiece name, const Tensor& tensor, bool lock_held)

用 tensor 替换 name 对于的 input Tensor

Status OpKernelContext::input_list(StringPiece name, OpInputList* list)

将 name 对应的 input 保存在 list 中

Status OpKernelContext::mutable_input_list(StringPiece name, OpMutableInputList* list)

将 name 对应的 input 保存在 list 中, 并加锁

Status OpKernelContext::output_list(StringPiece name, OpOutputList* list)

将 name 对应的 output 保存在 list 中

Status allocate_output(int index, const TensorShape& shape, Tensor** output)
Status allocate_output(int index, TensorShape& shape, Tensor** output)
Status allocate_output(StringPiece name, TensorShape& shape, Tensor** tensor)
Status allocate_output(StringPiece name, TensorShape& shape, Tensor** tensor, AllocatorAttributes attr)
Status OpKernelContext::allocate_output(int index, const TensorShape& shape, Tensor** output, AllocatorAttributes attr)

分配一个 Tensor, 替换 name 或 index 对应的 output Tensor, 将分配的 Tensor 保存在 output

Status allocate_tensor(DataType type, TensorShape& shape, Tensor* out_tensor,
    AllocatorAttributes attr, AllocationAttributes& allocation_attr)

创建一个 Tensor, out_tensor 指向该 Tensor

Status allocate_temp(DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr, const AllocationAttributes& allocation_attr)

分配 Tensor 保存在 out_temp, 并根据 allocator_attr 决定是否进行记录

Status allocate_persistent(DataType type, const TensorShape& shape, PersistentTensor* out_persistent,
                        Tensor** out_tensor, AllocatorAttributes attr)

分配一个 tensor, out_tensor 指向分配的 Tensor

Status set_output(StringPiece name, const Tensor& tensor)
void set_output(int index, const Tensor& tensor)
void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref)
Status set_output_ref(StringPiece name, mutex* mu, Tensor* tensor_for_ref)

用 tensor 替换 name 或 index 对应的 output Tensor

Status OpKernelContext::mutable_output(StringPiece name, Tensor** tensor)
inline Tensor* OpKernelContext::mutable_output(int index)

返回 name 对应的 output Tensor, 保存在 Tensor

Status OpKernelContext::release_output(StringPiece name, TensorValue* value)
inline TensorValue OpKernelContext::release_output(int index)

将 output_[index] 置为 TensorValue()

bool OpKernelContext::ValidateInputsAreSameShape(OpKernel* op)

inputs 中其他元素的类型必须与 inputs[0] 的类型相同

Status MatchSignature(const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs)

检查 expected_inputs 与 input_  的类似是否兼容
检查 expected_outputs 与 output_  的类似是否兼容

bool OpKernelContext::allocate_on_host(AllocatorAttributes alloc_attr)

  return alloc_attr.on_host() || device()->attributes().device_type() == "CPU";

void OpKernelContext::record_host_persistent_memory_allocation(int64 size, int64 alloc_id)

  host_persistent_memory_allocated_ += size;
  host_persistent_alloc_ids_.push_back(alloc_id);

void OpKernelContext::record_device_persistent_memory_allocation(int64 size, int64 alloc_id)

  device_persistent_memory_allocated_ += size;
  device_persistent_alloc_ids_.push_back(alloc_id);

std::vector<int64> OpKernelContext::host_persistent_alloc_ids()

获取 host_persistent_alloc_ids_

std::vector<int64> OpKernelContext::device_persistent_alloc_ids()

获取 device_persistent_alloc_ids

bool InTypeList(DataType dt, const AttrValue& type_list)

返回 dt 是否在 type_list 中

Status AttrsMatch(AttrSlice attrs, const KernelDef& kernel_def, bool* match)

对于 kernel_def.constrainta() 中的元素 constraint，
当 constraint.name():constraint.allowed_values() 都存在于 attrs, match 设置为 true
当 constraint.name():constraint.allowed_values() 与 attrs 不完全匹配时

Status FindKernelRegistration(const DeviceType& device_type, const NodeDef& node_def, const KernelRegistration** reg, bool* was_attr_mismatch)

从 global_op_registry 中存在与 device_type, node_def 对应的 KernelRegistration 列表,
将与 node_def 的 attr 匹配的 KernelRegistration 保存在 reg，如果有不对应的设置 was_attr_mismatch 为 true

Status FindKernelRegistration(const DeviceType& device_type, const Node& node, const KernelRegistration** reg, bool* was_attr_mismatch)

  return FindKernelRegistration(device_type, node.def(), reg, was_attr_mismatch);

Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def, const KernelDef** def, string* kernel_class_name)

  FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch)
  if (def != nullptr) def = &reg->def;
  if (kernel_class_name != nullptr) kernel_class_name = reg->kernel_class_name;

Status SupportedDeviceTypesForNode(const std::vector<DeviceType>& prioritized_types, const NodeDef& def, DeviceTypeVector* device_types)

如果 def.op() 和 def 已经注册，从 prioritized_types 中找到已经注册的 DeviceType 类型加入 device_types。
如果 def.op() 没有注册，将 prioritized_types 加入 device_types

1. 从 registry_ 中找 def.op() 对应的 OpRegistrationData
2. 如果没有找到, 将 prioritized_types 中都加入 device_types, 返回
3. 如果找到，遍历 prioritized_types, 如果 KernelRegistry 中存在与 def 对应的 KernelRegistration, 加入 device_types

string KernelsRegisteredForOp(StringPiece op_name)

遍历 KernelRegistry，将其 value 转化为字符串。格式为
device='kernel_def.device_type()';[label='kernel_def.label()]; (kernel_def.constraint(i).name() in "类型列表")+\n ...

std::unique_ptr<OpKernel> CreateOpKernel(DeviceType device_type, DeviceBase* device, Allocator* allocator, const NodeDef& node_def, int graph_def_version, Status* status)

  CreateOpKernel(std::move(device_type), device, allocator, nullptr, node_def, graph_def_version, &kernel);
  return std::unique_ptr<OpKernel>(kernel);

bool FindArgInOp(StringPiece arg_name, const protobuf::RepeatedPtrField<OpDef::ArgDef>& args)

args 中是否存在  arg_name

Status ValidateKernelRegistrations(const OpRegistryInterface& op_registry)

  for (const auto& key_registration : GlobalKernelRegistryTyped())
    const KernelDef& kernel_def(key_registration.second.def);
    const Status status = op_registry.LookUp(kernel_def.op(), &op_reg_data);
    const OpDef& op_def = op_reg_data->op_def;
    for (const auto& host_memory_arg : kernel_def.host_memory_arg())
      如果 host_memory_arg 既不存在于 op_def.input_arg(), 也不存在于 op_def.output_arg() 返回错误


Status AddArgToSig(const NodeDef& node_def, const OpDef::ArgDef& arg_def, DataTypeVector* sig)


### 工具类

bool HasAttrStyleType(const OpDef::ArgDef& arg)

  return arg.type() != DT_INVALID || !arg.type_attr().empty() || !arg.type_list_attr().empty();

Status AllowedTypeValue(DataType dt, const OpDef::AttrDef& attr)

  dt 是否在 attr.allowed_values().list().type() 中

Status AllowedStringValue(const string& str, const OpDef::AttrDef& attr)

  str 是否在 attr.allowed_values().list().s() 中

Status ValidateAttrValue(const AttrValue& attr_value, const OpDef::AttrDef& attr)

1. 确认 attr_value 包含类型 attr.type()
2. 如果 attr.has_minimum() 并且 attr.type() == int,  必须满足attr_value.i() >= attr.minimum()
3. 如果 attr.has_minimum() 并且 attr.type() == list(x),  必须满足 attr_value.x_size() >= attr.minimum()
4. attr_value 是否在 attr.type() 中

Status AttrValueHasType(const AttrValue& attr_value, StringPiece type)

确认 attr_value 包含类型 type

const OpDef::AttrDef* FindAttr(StringPiece name, const OpDef& op_def)

op_def.attr() 中是否包含 name 的属性

OpDef::AttrDef* FindAttrMutable(StringPiece name, OpDef* op_def)

op_def.attr() 中是否包含 name 的属性, 并且返回的属性是可以修改的

static Status ValidateArg(const OpDef::ArgDef& arg, const OpDef& op_def, bool output, std::set<string>* names)

arg 是 op_def 的属性，并且是合法的属性, 将 arg.name() 加入 names

arg.type() != DT_INVALID || !arg.type_attr().empty() || !arg.type_list_attr().empty();

如果 !arg.number_attr().empty()
1. op_def 中存在 arg.name() 的属性 attr
2. attr->type() == "int" && attr->has_minimum() && attr->minimum() >= 0 && arg.type_list_attr().empty() 并且 arg.type() != DT_INVALID 和 !arg.type_attr().empty() 只能有一个为 null

如果 !arg.type_attr().empty() : op_def 中 arg 对应的 attr.type() == "type"
如果 !arg.type_list_attr().empty() : op_def 中 arg 对应的 attr.type() == "list(type)"

Status ValidateOpDef(const OpDef& op_def)

1. 检查 op_def.name() 的合法性
2. 检查每个属性的合法性
2.1 属性的格式合法
2.2 attr.type 在 attr.allowed_values() 中
2.3 attr.default_value()
3. 检查 input 参数的合法性
4. 检查 output 参数的合法性

Status CheckOpDeprecation(const OpDef& op_def, int graph_def_version)

如果 op_def.has_deprecation() 为 true, 显示错误信息

string SummarizeArgs(const protobuf::RepeatedPtrField<OpDef::ArgDef>& args)

将所有参数转换为字符串, `arg.name():[Ref(][arg.number_attr()*][arg.type()|arg.type_attr()][)]`

string SummarizeOpDef(const OpDef& op_def)

op_def 转换为字符

bool IsSubsetOf(const T& sub, const T& super)

sub 的每个元素是否在 super 中

bool MoreRestrictive(const OpDef::AttrDef& old_attr, const OpDef::AttrDef& new_attr)

new_attr 是否是  old_attr 的子集

bool HigherMinimum(const OpDef::AttrDef& old_attr, const OpDef::AttrDef& new_attr)

  return new_attr.minimum() > old_attr.minimum();

string MinStr(const OpDef::AttrDef& attr)

  return strings::StrCat(attr.minimum());

void FillAttrMap(const OpDef& op_def, AttrMap* attr_map)

将 op_def.attr() 加入 attr_map

string ComputeArgSignature(protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
    const AttrMap& old_attrs, const AttrMap& new_attrs, std::vector<bool>* ref, bool names)

将 args 中的原生转换为 string, ref 保存了每个属性是否为 ref 类型。

Status OpDefCompatible(const OpDef& old_op, const OpDef& new_op)

检查 old_op 和  new_op 的兼容性
1. name 相同
2. old_op.attr() 的元素在 new_op 中都能找到，并且 type 相同，没有更严格, 没有更高的 minimum
3. new_op.attr() 的原生在 old_op 中都能找到
4. old_op 和  new_op 的 input 和 ouput 的 signature 相同

Status OpDefAddedDefaultsUnchanged(const OpDef& old_op, const OpDef& penultimate_op, const OpDef& new_op)

1. penultimate_op 中属于 new_op 不属于 old_op 的 OpDef::AttrDef
2. new_attr 和 penultimate_op 的元素都有默认值, 并且相同

void RemoveNonDeprecationDescriptionsFromOpDef(OpDef* op_def)

删除  op_def 中的描述，包括  input, output, attr, summary, desciption

void RemoveDescriptionsFromOpDef(OpDef* op_def)

删除 op_def 中的描述

void RemoveDescriptionsFromOpList(OpList* op_list)

删除 op_list 中的描述

### OpSegment

维护了 OpKernel 运行设备的 session，每一个 session 包含多个设备, 每个设备一对 node_name:kernel
通过引用计数来记录 session 中的每个设备被多少客户端同时持有, 当没有任何客户端持有该 node_name:kernel,
就从 session_ 中删除

例子
```cpp
  OpSegment::CreateKernelFn GetFn(const NodeDef* ndef) {
    return [this, ndef](OpKernel** kernel) {
      Status s;
      auto created = CreateOpKernel(DEVICE_CPU, &device_, cpu_allocator(),
                                    *ndef, TF_GRAPH_DEF_VERSION, &s);
      if (s.ok()) {
        *kernel = created.release();
      }
      return s;
    };
  }

  std::vector<NodeDef> int32_nodedefs_;
  std::vector<NodeDef> float_nodedefs_;
  NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_INT32)
                      .Input("y", 0, DT_INT32)
                      .Finalize(&def);
  int32_nodedefs_.push_back(def);
  NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_FLOAT)
                      .Input("y", 0, DT_FLOAT)
                      .Finalize(&def);
  float_nodedefs_.push_back(def);

  OpSegment opseg;
  OpKernel* op;

  opseg.AddHold("A");
  opseg.AddHold("B");

  for (int i = 0; i < 10; ++i) {
    // Register in session A.
    auto* ndef = &float_nodedefs_[i];
    opseg.FindOrCreate("A", ndef->name(), &op, GetFn(ndef));

    // Register in session B.
    ndef = &int32_nodedefs_[i];
    opseg.FindOrCreate("B", ndef->name(), &op, GetFn(ndef));
  }

  opseg.RemoveHold("A");
  opseg.RemoveHold("B");
```


Status OpSegment::FindOrCreate(const string& session_handle, const string& node_name, OpKernel** kernel, CreateKernelFn create_fn)

如果 在 session_ 中找到  session_handle, node_name 对应的 OpKernel, 设置 kernel 返回
如果 在 session_ 中没有找到 session_handle, node_name 对应的 OpKernel, 调用 create_fn 创建 OpKernel, 加入 sessions_


### OpGenOverrideMap

string WordWrap(StringPiece prefix, StringPiece str, int width)

str 以空格分割，每个元素一行，增加前缀 prefix

static bool SplitAt(char split_ch, StringPiece* orig, StringPiece* before_split)

orig 中找到 split_ch 对应的之前的子字符串

static bool StartsWithFieldName(StringPiece line, const std::vector<string>& multi_line_fields)

line 格式为 "<spaces><field>:", 其中 field 是否存在于 multi_line_fields

static bool ConvertLine(StringPiece line, const std::vector<string>& multi_line_fields, string* ml)

如果 line 中的 field 在 multi_line_fields 中(其中  line 格式为 "<spaces><field>: otherthing")
1. 从 line 中找到冒号之前的部分 up_to_colon
2. 找到双引号之内的部分为 uescaped 和双引号之后的部分 suffix，

如果 uescaped 中找到 "END[num]" 的字符串，之后组合字符串 `up_to_colon: <<END[num]\n uescaped \n END[num] suffix \n`
如果 uescaped 中没有找到 "END[num]", 之后组合字符串 `up_to_colon: <<END\n uescaped \n END[num] suffix \n`

string PBTxtToMultiline(StringPiece pbtxt, const std::vector<string>& multi_line_fields)

将 pbtxt 以换行符进行分割, 组成数组，每个行为 line(其中  line 格式为 "<spaces><field>: otherthing")
如果 line 中的 field 不在 multi_line_fields 中, 直接将 line 转换为 "line\n"
如果 line 中的 field 在 multi_line_fields 中
1. 从 line 中找到冒号之前的部分 up_to_colon
2. 找到双引号之内的部分为 uescaped 和双引号之后的部分 suffix，

例如

```text
  foo: "ghi\njkl\n"
```
转换为
```text
  foo: <<END
ghi
jkl

更多参考 op_gen_lib_test.cc

static bool FindMultiline(StringPiece line, size_t colon, string* end)

将 line  索引 conlon+1 之后的部分并且包含 `<<` 之后的部分保持在 end, 如 `<<END[num]`  将 END[num] 保存在 end

string PBTxtFromMultiline(StringPiece multiline_pbtxt)

把形如 `something :<<END[num]\n uescaped \n END[num] suffix \n` 转为
`something:"uescaped" suffix"`

更多参考 op_gen_lib_test.cc

Status OpGenOverrideMap::LoadFileList(Env* env, const string& filenames)

依次读取 filenames 中每个 filename 的内容，之后解析为 OpDef，保存在 map_ 里面

Status OpGenOverrideMap::LoadFile(Env* env, const string& filename)

从 filename 中读内容，之后解析为 OpDef，保存在 Map 里面

static void StringReplace(const string& from, const string& to, string* s)

将 s 中，用 to 替换 from

static void RenameInDocs(const string& from, const string& to, OpDef* op_def)

将 op_def  的 input, output, attr, summary, description，用 to 替换 from

const OpGenOverride* OpGenOverrideMap::ApplyOverride(OpDef* op_def)

从  map_ 中找到 op_def.name() 对应的 OpGenOverride 替代 op_def 中的相关值。

name, atrr, input, output


### MemoryType

Status MemoryTypesForNode(const OpRegistryInterface* op_registry, const DeviceType& device_type, const NodeDef& ndef,
                          MemoryTypeVector* inp_mtypes, MemoryTypeVector* out_mtypes)

初始化 ndef 中 opdef 的每个 input 和 output 是保存在 HOST_MEMORY 还是 DEVICE_MEMORY，inp_mtypes 和  out_mtypes 分别记录
了哪些索引应该保存在 HOST_MEMORY，哪些保存在 DEVICE_MEMORY

1. 查找 ndef.op() 对应的已经注册的 op_def(OpDef)
2. 根据 device_type, ndef 找到 kdef(KernelDef)
3. 找到 ndef 的输入参数和输出参数的类型 inp_dtypes, out_dtypes
4. 如果 ndef.op() == "SymbolicGradient", inp_mtypes 保存 inp_dtypes, out_mtypes 保存 out_dtypes 返回
5. 用 kdef->host_memory_arg() 和 ndef 的 "_input_hostmem" 和 "_output_hostmem" 初始化 inp_mtypes 和  out_mtypes

注: DT_INT32 保存在 HOST_MEMORY, 非 DT_INT32 保存在  DEVICE_MEMORY

static Status ProcessMemoryTypes(const DeviceType& device_type, const Graph* g, const std::function<Status(const Edge*, MemoryType, MemoryType)>& fn)
```
对 g 中的所有 Edge 的 MemoryType 进行处理

1. 找到 g 中每个节点 Op 的每个属性的 MemoryType
2. 建立 src + src_output 保存的 MemoryType 的映射关系
3. 遍历 g 的所有 Edge，找到每个 Edge e 的 src 对应的 MemoryType 记录在 sm, dst 对应的 MemoryType 记录在 dm，调用  fn(e, sm, dm)

Status ValidateMemoryTypes(const DeviceType& device_type, const Graph* g)

检查 g 的所有  Edge 确保 src+src_ouput 与  dst+dst_input 的 MemoryType 是一致的.
j
static Node* Send(Graph* g, const string& tensor_name, const string& device_name, bool host, const Edge* edge)

创建一个 Node，并返回

name : "n/_${NUM}"
op : `_HostSend` 或 `_Send`
input : edge->src(),edge->src_output()
attr :  "tensor_name", tensor_name
        "send_device", device_name
        "send_device_incarnation", 0
        "recv_device", device_name
        "_hostmem_sendrecv", true

static Node* Recv(Graph* g, const string& tensor_name, const string& device_name, bool host, const Edge* edge)

创建一个 Node，并返回

name : "n/_${NUM}"
op : `_HostRecv` 或 `_Recv`
attr :  "tensor_type", edge->src()->output_type(edge->src_output())
        "tensor_name", tensor_name
        "send_device", device_name
        "send_device_incarnation", 0
        "recv_device", device_name
        "_hostmem_sendrecv", true

Status EnsureMemoryTypes(const DeviceType& device_type, const string& device_name, Graph* g)

1. 确保 g 中每个 Edge 的 MemoryType 是一样的
2. 对于不能兼容的 Edge，将 Edge 修改为 "e->src+e->src_ouput -> `_Send` -> `_Recv` -> e->dst+e->dst_input"
3. 确保此时所有的 Edge 的 MemoryType  是一样的

Status MemoryTypeForOutput(const DeviceType& device_type, const Graph* g, const Node* n, int index, MemoryType* memory_type)

获取 node 索引为  index 的输出类型

```cpp
  REGISTER_OP("HostMemoryTest")
      .Input("a: float")
      .Input("b: T")
      .Input("c: N * string")
      .Input("d: Tlist")
      .Output("o: N * T")
      .Output("p: Tlist")
      .Attr("T: type")
      .Attr("N: int")
      .Attr("Tlist: list(type)");
  REGISTER_KERNEL_BUILDER(Name("HostMemoryTest").Device(DEVICE_CPU), DummyKernel);
  REGISTER_KERNEL_BUILDER(Name("HostMemoryTest")
                            .Device(DEVICE_GPU)
                            .HostMemory("a")
                            .HostMemory("c")
                            .HostMemory("d")
                            .HostMemory("o"),
                        DummyKernel);
  NodeDef node_def;
  NodeDefBuilder("test", "HostMemoryTest")
                   .Input(FakeInput())
                   .Input(FakeInput(DT_BOOL))
                   .Input(FakeInput(3))
                   .Input(FakeInput({DT_INT32, DT_FLOAT, DT_INT32}))
                   .Finalize(&node_def));
  MemoryTypeVector input, output;
  MemoryTypesForNode(OpRegistry::Global(), DEVICE_CPU, node_def, &input, &output);
  MemoryTypesForNode(OpRegistry::Global(), DEVICE_GPU, node_def, &input, &output);
```

### CallOp

void ComputeAsync(OpKernelContext* ctx, DoneCallback done)

1. 用 ctx 初始化 ctx->params_->function_library->Run() 的参数， 之后调用该函数
2. 用返回值 重置 ctx->ouput

## 附录

```cpp
  Tensor x0(DT_FLOAT, {2, 3});
  x0.flat<float>().setZero();
  Tensor x1(DT_FLOAT, {2, 3});
  x1.flat<float>().setZero();
  Tensor dy(DT_FLOAT, {2, 2, 3});
  test::FillIota<float>(&dy, 0);
  auto gdef = test::function::GDef(
      {f::NDef("x0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("x1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("axis", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
               {{"f", FDH::FunctionRef("Pack",
                                       {{"N", 2}, {"T", T}, {"axis", axis}})},
                {"Tin", DataTypeSlice{T, T, T}},
                {"Tout", DataTypeSlice{T, T}}})});
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x0:0", x0},
                         {"x1:0", x1},
                         {"axis:0", test::AsScalar(axis)},
                         {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  TF_CHECK_OK(sess->Close());


  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(op.name, &op_reg_data));
  shape_inference::InferenceContext c(
      op.graph_def_version, &op.node_def, op_reg_data->op_def, in_shapes,
      op.input_tensors, {}, std::move(input_resource_handle_shapes_and_types));
  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));
  const int num_outputs = c.num_outputs();

