
## Node

一个节点要满足的条件
0. node.op 必须与  op 的 name 一致
1. 输入中，control input 在最后面，即任何非 control input  前面不能有 control input
2. 节点中 op 的 attr 不能有重复, node  中除了 internal 属性(_开头的属性), 其他属性与 op 的属性必须一致
3. 节点中的  attr 与  op 的 attr 在类型上必须匹配
4. 在创建节点时，对应的  op 必须先注册到全局变量 global_op_registry

### 数据结构

NodeBuilder -> NodeDefBuilder -> NodeDef

class AttrSlice
  struct Scratch
    string a;
    string b;
  const NodeDef* ndef_;
  const AttrValueMap* attrs_;

class NodeBuilder
  struct NodeOut
    bool error;
    string name;
    int32 index;
    DataType dt;

  NodeDefBuilder def_builder_;
  std::vector<NodeOut> inputs_; //当前节点的输入，在 graph 增加 Edge 的时候使用
  std::vector<Node*> control_inputs_; //保存所有 control input, 在 graph 增加 Edge 的时候使用
  std::vector<string> errors_;

class NodeDefBuilder
  const OpDef* op_def_;
  NodeDef node_def_;
  int inputs_specified_; //输入的索引
  std::vector<string> control_inputs_; //因为 control input 需要最后加入，所以，所有的  control input 先临时保存在这里
  std::vector<string> errors_;

class NodeDef
  string name : 节点名称, 格式 \[A-Za-z0-9.]\[A-Za-z0-9_./]*
  string op   : node_class 的 string 描述。下划线开头的，限是内部使用。Switch, Merge, Enter 等等, 参见 kNodeClassTable
  string input[] : 输入, 可以是多个
  string device : 节点所属的设备
  map<string, AttrValue> attr : 节点属性,  除了默认属性，对应 OpDef 的所有属性都是必须的

1. 当 input 中类型不是通用类型或者是后期需要指定的类型时，将该类型加入 attr
2. control_input 是在其他 input 之后添加
3. NodeDef 可能有内部属性(以"_" 开头)
4. 控制节点，不是输入节点且没有输入 edge，或和不是输出节点且没有输出 edge 的节点

其中
  input 每个元素, 包括一般 input(格式 "名称:输出节点的index", index 为 0 可以忽略) 和 control input(格式 "^node").
  device 的格式
  ```
  PARTIAL_SPEC ::= ("/" CONSTRAINT) *
  CONSTRAINT ::= ("job:" JOB_NAME)
               | ("replica:" [1-9][0-9]*)
               | ("task:" [1-9][0-9]*)
               | ( ("gpu" | "cpu") ":" ([1-9][0-9]* | "*") )
  Valid values for this string include:
  * "/job:worker/replica:0/task:1/gpu:3"  (full specification)
  * "/job:worker/gpu:3"                   (partial specification)
  * ""                                    (no specification)
  ```
  attr 的 key 格式 "\[a-z][a-z0-9_]+" 来自 OpDef 对应的属性名; value 必须匹配 OpDef 对应属性的类型

### NodeBuilder

Status NodeBuilder::Finalize(Graph* graph, Node** created_node)

1. 用 def_builder_ 构建 node_def
2. node_def 增加到 graph
3. 增加 inputs_  作为 node_def 到 graph 的 edge
4. 增加 control_inputs_  作为 node_def 到 graph 的 edge

bool NodeBuilder::GetOutputType(Node* node, int i, DataType* dt)

获取  node 第 i 个输出节点的类型

### NodeDefBuilder

typedef std::function<Status(const OpDef&, int, const NodeDef&, NodeDefBuilder*)> FakeInputFunctor;

根据 OpDef 创建并初始化 NodeDef

void NodeDefBuilder::AddInput(StringPiece src_node, int src_index)

当  src_index 大于 0 时，增加 "src_node:src_index" 的输入
当  src_index 小于 0 时，增加 "src_node" 的输入

void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg, StringPiece src_node, int src_index, DataType dt)

1. 将 src_node:src_index 加入 NodeDef
2. 当 input_arg->type() != DT_INVALID，校验 input_arg.type() 与 dt 的类型兼容
2. 当 input_arg->type() == DT_INVALID，增加属性 "input_arg->type_attr() BaseType(dt)"

NodeDefBuilder& NodeDefBuilder::Input(StringPiece src_node, int src_index, DataType dt)

增加 "src_node:src_index" 的输入， 并检查 OpDef 的类型与 dt 相同

void NodeDefBuilder::ListInput(const OpDef::ArgDef* input_arg, gtl::ArraySlice<NodeOut> src_list)

1. 将 src_list 中的元素依次加入 NodeDef
2. 校验类型

void NodeDefBuilder::VerifyInputType(const OpDef::ArgDef* input_arg, DataType expected, DataType dt)

expected 与 dt 的类型兼容

NodeDefBuilder& NodeDefBuilder::ControlInput(StringPiece src_node)

src_node 加入  control_inputs_

NodeDefBuilder& NodeDefBuilder::Device(StringPiece device_spec)

设置 NodeDef 的 device

Status NodeDefBuilder::Finalize(NodeDef* node_def)

将 control_input 加入 NodeDef, 在 OpDef 中但不在 NodeDef
中的属性，如果设置了默认值，增加属性。 初始化 node_def

NodeDefBuilder& NodeDefBuilder::Attr(StringPiece name, const AttrValue& value) {

增加 name, value 属性


### NodeDef Utils

static string SummarizeAttrsHelper(AttrSlice attrs, StringPiece device)

将  attrs 中的元素和 device 转换为 string。形如 a=b,c=d,_device=cpu

string SummarizeNodeDef(const NodeDef& node_def)

将  node_def 转换为 string, 形如 `add-op=add[c=d,e=f,_device=cpu](a,b,c)`

Status AddArgToSig(const NodeDef& node_def, const OpDef::ArgDef& arg_def, DataTypeVector* sig)

从  node_def.attr() 中查找 arg_def 对应的属性加入 sig

Status InOutTypesForNode(const NodeDef& node_def, const OpDef& op_def, DataTypeVector* inputs, DataTypeVector* outputs)

从  node_def 中查找  op_def 对应的  input, output 的类型

Status ValidateNodeDef(const NodeDef& node_def, const OpDef& op_def)

node_def 和 op_def 的 name, attr, input 数量 必须相同

Status ValidateExternalNodeDefSyntax(const NodeDef& node_def)

node_def 的 name 必须符合规则，control input 必须在最后
