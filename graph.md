
问题:

1. Node, Graph, Edge, Operation 之间到底是什么关系
2. Operation 如何初始化，如何使用，如何执行
3. Node 的 Attr 有什么作用, 如何使用的

An output is in the same frame as the node except for Exit nodes.
The output of Exit is in the parent frame of the Exit node.

An input is in the same frame as the node except for Enter nodes.
The input of Enter is in the parent frame of the Enter node.

Each participating device needs to decide a) if there is a next iteration,
and b) if the loop terminates. We take the approach to encode this control
flow logic in the dataflow graph. There are at least two possible encodings.
In a completely decentralized encoding, the participants communicate peer
to peer. The other encoding uses a frame leader (the participant who owns
the pivot termination predicate) to broadcast the termination condition to
all the participants. For now we take the latter because it is simpler.

#图

DAG (directed acyclic graph)
internal node : 中间节点
source : 表示输入, 它不依赖任何
sink : 表示输出, 没有其他依赖它

operation : 一个 operation 依赖另外一个 operation, 一个 operation 可以有多个输入，多个输出,
每个输入有 input index, 每个输出有 output index
由于有些优化依赖一个 edge 的输入和另外一个 edge 的输出，因此需要 index idex, 和 output index

### 概述

#### 最简版本

一个 Node 包括 input Edge 和 output Edge，Node 之间通过 Edge 连接，多个 Node
构成了 Graph。

Node 上可以进行 Operation，整个过程:

1. 数据从一些 input Edge 进入 Node
2. 在 Node 进行 Operation
3. 从 Node 的 output Edge 出去
4. 进入下一个 Node，进行 Operation, 再出去

#### 稍微详细版本

一个 Graph 包括 Node, Edge(分为 ControlEdge和非 ControlEdge), FunctionDefLibrary, 支持操作包括:
1. 根据 Node 的 id 查找 Node, 其中 id 为 0 为 Source Node, id 为 1 为 Sink Node
2. Edge 的 id 查找 Edge.
3. 增，删，遍历 Node, Edge
4. Device 和Node 的映射关系
6. graph 的序列化

Node 分为 Source, Sink, Op 三类，id 为 0 的是 Source 节点，
id 为 1 的是 Sink 节点，其他节点为 Op 节，只有 Op 节点参与 Operation
Grap 中必须有一个 Source 节点和一个  Sink 节点, source + 0 -> dink + 0

Node 表达图中一个节点，包括节点本身的属性与节点与其他节点的关系。一个 Node 操作包括:
0. 由 id, cost_id 和 NodeProperties 构造一个 Node
1. 获取 id, cost_id
2. type , Node, Edge 的输入输出
3. 希望部署在某个设备
4. 属性
5. 通过 Node 的 input Edge 和 output Edge 获取邻居 Node
6. Node 的类型，source, sink 或 op
7. 根据 名字找到 Node 的 class
8. Node 包含 Operation, 根据  Operation 的不同分为三大类, Source(id 为 0), Sink(id 为 1),  Op(id 大于 1)

Edge 连接了两个 Node, 由 src Node + src_output + dst_input + dst Node, 一个 Edge 的操作包括
对于 Control 节点，src_output 和 dst_input 都是 kControlSlot
1. Edge 的 id
2. 获取所属的 src Node 和 dst Node
3. input 索引 ，ouput 索引
4. 是否是 ControlEdge

## 数据结构

Grap
GraphDef -> Graph

NodeDef -> Graph

GraphConstructorOptions -> Options

ImportGraphDefOptions -> Options

OpRegistryInterface -> Graph -> Options -> GraphDefBuilder

Options -> NodeBuilder -> Node

GraphDefBuilder -> Graph

GraphDefBuilder -> GraphDef                   -> Graph
                   GraphConstructorOptions

struct BuildGraphOptions
  std::vector<string> feed_endpoints;
  std::vector<string> fetch_endpoints;
  std::vector<string> target_nodes; //Remove this when we unify target_nodes and fetch_endpoint, the former via "ref" fetch_endpoints.
  // If `true`, uses Arg/Retval to implement feeds/fetches; otherwise
  // uses Recv/Send to implement feeds/fetches.
  // TODO(mrry): Remove this when the distributed runtime supports Arg/Retval.
  bool use_function_convention = false;
  DebugOptions debug_options;
  string DebugString() const;

class GraphDefBuilder
  class Options
    Graph* const graph_ : 所属的 graph
    Status* const status_ : 操作状态
    string name_ :
    string device_ : Node 的设备名
    std::vector<Node*> control_inputs_ : 保存 Node 的 control_input 节点
    std::vector<std::pair<string, AttrValue>> attrs_;  保存 Node attr
  Graph graph_; //graph_(OpRegistry::Global())
  Status status_;
  Options opts_; //opts_(graph_)

Options 调用 FinalizeBuilder 转换为 NodeNode
最后调用 ToGraph 或 ToGraphDef 完成 Budiler 到 Graph 的转换

class GraphRunner
  std::unique_ptr<Device> cpu_device_;

class GraphMgr
  typedef GraphMgr ME;
  typedef std::map<string, Tensor> NamedTensors;
  typedef std::function<void(const Status&)> StatusCallback;
  struct ExecutionUnit
    Graph* graph = nullptr;
    Device* device = nullptr;
    Executor* root = nullptr;
    FunctionLibraryRuntime* lib = nullptr;
    // Build the cost model if this value is strictly positive.
    int64 build_cost_model = 0;
  struct Item
    string session;
    string handle;
    FunctionLibraryDefinition* lib_def = nullptr;
    std::vector<ExecutionUnit> units;
    GraphMgr* graph_mgr;

  const WorkerEnv* worker_env_;             // Not owned.
  DeviceMgr* device_mgr_;
  CostModelManager cost_model_manager_;
  mutex mu_;
  int64 next_id_ GUARDED_BY(mu_) = 0;
  bool sync_on_finish_ = true; //  从环境变量中读取 TF_SYNC_ON_FINISH 初始化
  // Table mapping graph handles to registered graphs.
  std::unordered_map<string, Item*> table_; //
  // Don't attempt to process cost models unless explicitly requested for at
  // least one of the items.
  bool skip_cost_models_ = true;




### CostGrap

包含多个 Node
每个 Node 包含

message CostGraphDef
  message Node
    string name = 1;//名称，全局唯一
    string device = 2;
    int32 id = 3; //在 partition 中唯一
    message InputInfo
      int32 preceding_node = 1;
      int32 preceding_port = 2;
    repeated InputInfo input_info = 4; //输入列表
    message OutputInfo
      int64 size = 1;
      int64 alias_input_port = 2;
      TensorShapeProto shape = 3;
      DataType dtype = 4;
    repeated OutputInfo output_info = 5; //输出列表
    int64 temporary_memory_size = 6;
    int64 host_temp_memory_size = 10;
    int64 device_temp_memory_size = 11;
    int64 host_persistent_memory_size = 12;
    int64 device_persistent_memory_size = 16;
    int64 compute_cost = 9;     // 估计计算耗时 ms
    int64 compute_time = 14;    // 分析的计算耗时 ms
    int64 memory_time = 15;     //内存访问时间，ms
    bool is_final = 7;          //该节点是否可以被销毁
    repeated int32 control_input = 8; //控制输入
  repeated Node node = 1;

### Operation

OpDef
    string name : 操作名称, 格式 \[A-Za-z0-9.]\[A-Za-z0-9_./]*
    ArgDef input_arg[] :
    ArgDef output_arg[] :
    AttrDef attr[] : 与 NodeDef attr 相关联
    OpDeprecation deprecation :
    string summary : 操作做什么事的总结
    string description : 操作的描述
    //以下为优化选项
    bool is_commutative : op(a,b) == op(b,a) 时为 true
    bool is_aggregate : 多个输入，一个输出且输入与输出的类型相同
    bool is_stateful : 是否是有状态的
    bool allows_uninitialized_input : 默认为 false

ArgDef
    string name : 输入或输出的名字
    string description : 描述
    //单个 tensor 设置 type 或 type_attr, 多个 tensor 设置 number_attr, type_list_attr
    DataType type :
    string type_attr :
    string number_attr :
    string type_list_attr :
    bool is_ref :

AttrDef
    string name : 名字 格式 \[a-z][a-z0-9_]+
    string type : 类型 ("string", "list(string)",
    AttrValue default_value : 默认值
    string description : 描述
    bool has_minimum
    int64 minimum : 对于 int 是最小值，对于 list(___) 是最小长度
    AttrValue allowed_values :

### Node

NodeDef
  string name : 节点名称, 格式 \[A-Za-z0-9.]\[A-Za-z0-9_./]*
  string op   : node_class 的 string 描述。下划线开头的，限是内部使用。Switch, Merge, Enter 等等, 参见 kNodeClassTable
  string input[] : 输入, 可以是多个
  string device : 节点所属的设备
  map<string, AttrValue> attr : 节点属性, 来源于 OpDef

  `_input_hostmem` : 哪些输入保存在 HOST_MEMORY
  `_output_hostmem` : 哪些输出保存在 HOST_MEMORY

其中
  input 每个元素, 包括一般 input(格式 名称:输出节点的index, index 为 0 可以忽略) 和 control input(格式 ^node).
  device 的格式
  attr 的 key 格式 "\[a-z][a-z0-9_]+" value 为 ....


class NodeProperties
  const OpDef* op_def :
  NodeDef node_def :
  const DataTypeVector input_types :
  const DataTypeVector output_types :

class Node
  int id_;       // -1 until Initialize() is called
  int cost_id_;  // -1 if there is no corresponding cost accounting node
  NodeClass class_;

  EdgeSet in_edges_; //输入边界
  EdgeSet out_edges_; //输出边界
  static const std::unordered_map<string, NodeClass>& kNodeClassTable : string:node_class 的对应关系
  std::shared_ptr<NodeProperties> props_; //保存所有的属性
  int assigned_device_name_index_; // Graph 的 device_names_ key
  Graph* graph_; //所属的 Graph

class Edge
  Node* src_;
  Node* dst_;
  int id_;
  int src_output_;
  int dst_input_;

class EdgeSet
  typedef const Edge* key_type;
  typedef const Edge* value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  class const_iterator;
  typedef const_iterator iterator;

  static const int kInline = 2;  // Must be >= 2.
  const void* ptrs_[kInline]; //If ptrs_[0] == this then ptrs_[1] points to a set<const Edge*>.
  uint32 mutations_ = 0;

  每次删除元素，会将最后的元素移动到被删的索引。

enum NodeClass //一个节点支持的 op 类型
  NC_UNINITIALIZED, //Node node_class 的默认值
  NC_SWITCH,
  NC_MERGE,
  NC_ENTER,
  NC_EXIT,
  NC_NEXT_ITERATION,
  NC_LOOP_COND,
  NC_CONTROL_TRIGGER,
  NC_SEND,
  NC_HOST_SEND,
  NC_RECV,
  NC_HOST_RECV,
  NC_CONSTANT,
  NC_VARIABLE,
  NC_IDENTITY,
  NC_GET_SESSION_HANDLE,
  NC_GET_SESSION_TENSOR,
  NC_DELETE_SESSION_TENSOR,
  NC_METADATA,
  NC_OTHER  // Not a special kind of node

typedef std::unordered_set<int32> NodeSet;

struct Node
  int32 rank;    // rank number assigned by Pearce-Kelly algorithm
  bool visited;  // Temporary marker used by depth-first-search
  void* data;    // User-supplied data
  NodeSet in;    // List of immediate predecessor nodes in graph
  NodeSet out;   // List of immediate successor nodes in graph

### Graph

GraphDef
    NodeDef node[] : 所有节点列表
    VersionDef versions : 版本, 每个 TensorFlow 支持某一范围的版本
    FunctionDefLibrary library : 实验性的功能, 暂不介绍

NodeDef
  string name : 节点名称, 格式 \[A-Za-z0-9.]\[A-Za-z0-9_./]*
  string op   : node_class 的 string 描述。下划线开头的，限是内部使用。Switch, Merge, Enter 等等, 参见 kNodeClassTable
  string input[] : 输入, 可以是多个
  string device : 节点所属的设备
  map<string, AttrValue> attr : 节点属性, 来源于 OpDef

VersionDef
    int32 producer : 当前版本
    int32 min_consumer : 最小版本号
    int32 bad_consumers[] : 哪些版本是不允许的。

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


class Graph
    static const int kControlSlot : 当 Edge.src_output 等于 kControlSlot 时，该 Edge 为 ControlEdge
    FunctionLibraryDefinition ops_;
    const std::unique_ptr<VersionDef> versions_;
    core::Arena arena_ : 内存分配器, 优化性能用
    std::vector<Node*> nodes_ : 所有的 Node, 索引是 node.id
    int64 num_nodes_ : Node 数量
    std::vector<Edge*> edges_ : 所有的 Edge, 索引是 Edge id
    int num_edges_ : Edge 的数量
    std::vector<Node*> free_nodes_ : 已经分配了空间，但是还没有使用的 Node
    std::vector<Edge*> free_edges_ : 已经分配了空间，但是还没有使用的 Edge
    int name_counter_ : 用于生产唯一的名字
    std::vector<string> device_names_ : 保存 Node 对应的 Device 名称, 索引是 node.assigned_device_name_index_
    std::unordered_map<string, int> device_names_map_ : // Maps unique device names to indices within device_names_[i].

## 源码分析


### graph

Graph::Graph(const OpRegistryInterface* ops)

1. 基本类成员初始化
2. 创建一个 \_SOURCE 和 \_SINK 的 Node
3. 增加一个 Edge, Edge 的 src_ 为 SOURCE Node, dst_ 为 SINK Node，src_output 和 dst_intput 都为 kControlSlot

Node* AddNode(const NodeDef& node_def, Status* status)

1. node_def.op() 查找对于的 op_def
2. 通过 node_def 创建并初始化 Node

Node* CopyNode(Node* node)

拷贝 node

void Graph::RemoveNode(Node* node)

1. 如果 node->in_edges_ 不为空，删除 node 的所有 in_edges_
2. 如果 node->out_edges_ 不为空，删除 node 的所有 out_edges_
3. 回收该 node 到 free_nodes_

Edge* AddEdge(Node* source, int x, Node* dest, int y) {

给 source 的 x 和 dest 的 y 两个方向加入一个 edge

1. 初始化 edge
2. 给 source 的 x 和 dest 的 y 两个方向加入一个 edge

void RemoveEdge(const Edge* e)

将 e  从它的  src->out_edges_ 和 dst->in_edges_ 中删除，并回收到 free_edges_

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {

给 dst 增加一个 input

void ToGraphDef(GraphDef* graph_def)

遍历所有的 Node, 用 node 的 assigned_device_name 和 in_edge 初始化 node_def,
将  node_def 加入 graph_def

Node* AllocateNode(std::shared_ptr<NodeProperties> props, const Node* cost_node)

构造一个节点，并初始化。优先从 free_nodes_ 中取

void ReleaseNode(Node* node) {

释放内存， 回收节点到 node

int InternDeviceName(const string& device_name)

从 device_names_map_ 中找到 device_names_ 对应的 index

void Graph::ToGraphDefSubRange(GraphDef* graph_def, int from_node_id)

graph_def 保存  from_node_id 开始的所有 Node

bool Graph::IsValidNode(Node* node)

1. node.id() < nodes_.size()
2. nodes_[node.id()] == node


## 实现分析

void SetDefaultDevice(const string& device, GraphDef* graph_def)

设置 graph_def 所有节点的 device 为 device

Status BuildControlFlowInfo(Graph* g, std::vector<ControlFlowInfo>* info)

采用广度优先算法，从 g 的 source node 开始，初始化 info 中 info[node->id()] 相关的信息

### CostModel

is_global : 如果是 true, Id() 为 Node:cost_id, 如果为 false, Id() 为 Node:id

int32 min_count_
int32 update_times_
std::vector<int32> count_ : 每个节点执行的次数, 其中索引为节点 id
std::vector<Microseconds> time_ : 每个节点累计执行时间, 其中索引为节点 id
std::vector<gtl::InlinedVector<Bytes, 2> > slot_bytes_ 每个 channel 累计输出 bytes, 其中索引为节点 id
std::vector<Microseconds> max_exec_time_ : 其中索引为节点 id
std::vector<MemUsage> max_mem_usage_ : 其中索引为节点 id
std::vector<gtl::InlinedVector<int64, 2> > output_port_alloc_ids_ : 其中索引为节点 id
std::set<int64> host_persistent_alloc_ids_
std::map<string, std::set<int64>> persistent_alloc_ids_by_devices_
TensorShapeProto unknown_shape_

CostModel(bool is_global) : is_global_(is_global)

unknown_shape_.set_unknown_rank(true);

void SuppressInfrequent()

1. 如果 count_ 为空，返回, 否则继续步骤2
2. 如果 count_ 中存在非零值，找到 count_ 中的中间大小值，并设置 min_count_ 为中间值的一半 否则，设置 min_count_ 为 1

void MergeFromLocal(const Graph& g, const CostModel& cm)

将 cm 中属于 g 中节点的统计信息合并到当前 CostModel 中
其中统计信息包括 count_, time_, slot_bytes_

void MergeFromGlobal(const CostModel& cm)

将 cm 的统计信息依次合并到当前 CostModel
其中统计信息包括 count_, time_, slot_bytes_

void MergeFromStats(const NodeNameToCostIdMap& map, const StepStats& ss)

将 ss 中所有设备，所有节点的统计信息更新到当前 CostModel
其中统计信息包括 count_, time_, slot_bytes_

void Ensure(int id)

确保当前节点能够容纳 id 的信息

void SetNumOutputs(const Node* node, int num_outputs)

设置

slot_bytes_
max_mem_usage_->output_port_mem
max_mem_usage->output_port_shape
max_mem_usage->output_port_type
output_port_alloc_ids_ 的

的容量最少为 num_outputs

void RecordCount(const Node* node, int count)

count_ 中将 node 的执行次数增加 count

int TotalCount(const Node* node)

从 count_ 中获取 node 的执行次数

void RecordSize(const Node* node, int slot, Bytes bytes)

将 node 对应的 slot_bytes_[slot] 对应的值加 bytes

Bytes TotalBytes(const Node* node, int slot)

设置 node 对应 slot_bytes_[slot] 的值

void SizeEstimate(const Node* node, int slot)

如果 node 的执行次数少于 min_count_ 返回 Bytes(0)
否则 返回 TotalBytes(node, slot) / std::max(1, TotalCount(node))

void RecordTime(const Node* node, Microseconds time)

time_ 中将 node 的执行时间增加 count

Microseconds TotalTime(const Node* node

从 time_ 中获取 node 的执行次数

Microseconds TimeEstimate(const Node* node)

如果 node 的执行次数少于 min_count_ 返回 kMinTimeEstimate
否则 返回 std::max(kMinTimeEstimate, TotalTime(node) / std::max(1, count))

void CheckInitialized(const Graph& graph)

检查 graph 中的节点是否已经存在于当前的 CostModel

void RecordMaxMemorySize(const Node* node, int output_slot, Bytes bytes, const TensorShapeProto& tensor_shape, const DataType& dtype)

用 bytes 对比 max_mem_usage_[Id(node)].output_port_mem[output_slot]
如果 bytes 大，更新 max_mem_usage_[Id(node)]

Bytes MaxMemorySize(const Node* node, int slot)

返回 max_mem_usage_[Id(node)].output_port_mem[slot]

TensorShapeProto& MaxMemoryShape(const Node* node,int slot)

返回 max_mem_usage_[Id(node)].output_port_shape[slot]

DataType MaxMemoryType(const Node* node, int slot)

返回 max_mem_usage_[Id(node)].output_port_type[slot]

Bytes CostModel::TempMemorySize(const Node* node)

返回 max_mem_usage_[Id(node)].temp_memory_size

Bytes CostModel::HostTempMemorySize(const Node* node)

返回 max_mem_usage_[Id(node)].host_temp_memory_size

Bytes CostModel::DeviceTempMemorySize(const Node* node)

返回 max_mem_usage_[Id(node)].host_temp_memory_size

Bytes CostModel::HostPersistentMemorySize(const Node* node)

返回 max_mem_usage_[Id(node)].host_persistent_memory_size;

Bytes CostModel::DevicePersistentMemorySize(const Node* node)

返回 max_mem_usage_[Id(node)].device_persistent_memory_size;

void RecordMemoryStats(const Node* node, MemoryStats& memory_stats)

用 memory_stats 更新 max_mem_usage_, host_persistent_alloc_ids_, persistent_alloc_ids_by_devices_

void RecordMaxExecutionTime(const Node* node, Microseconds time)

设置 node 的最大长执行时间： max_exec_time_[Id(node)] = std::max(max_exec_time_[Id(node)], time)

Microseconds MaxExecutionTime(const Node* node)

获取 max_exec_time_[Id(node)]

void RecordAllocationId(const Node* node, int output_slot, int64 alloc_id)

    output_port_alloc_ids_[Id(node)][output_slot] = alloc_id;

int64 AllocationId(const Node* node, int slot) const {

    output_port_alloc_ids_[id][slot];

bool CostModel::IsPersistentTensor(const Node* node, int64 alloc_id) const {

  if (host_persistent_alloc_ids_.count(alloc_id) > 0)  return true;
  if (persistent_alloc_ids_by_devices_.find(node->assigned_device_name()) ==
      persistent_alloc_ids_by_devices_.end())
    return false;
  return persistent_alloc_ids_by_devices_.at(node->assigned_device_name())
      .count(alloc_id);

Microseconds CopyTimeEstimate(Bytes b, double network_latency_millis, double estimated_gbps)

根据网络延迟(network_latency_millis)和传输速率(estimated_gbps)估计传输 b 大小的数据需要的时间

Microseconds ComputationTimeEstimate(int64 math_ops) {

  return Microseconds(math_ops / 1000);

void IncrementUpdateTimes()

    update_times_++

int32 GetUpdateTimes()

    return update_times_;

static void AddNodesToCostModel(const Graph& g, CostModel* cost_model)

将 g 中的所有节点加入 cost_model

static void AssignSizes(const Graph& g, CostModel* cost_model)

for(e : g.edges()):
    if e->IsControlEdge() continue
    cost_model.slot_bytes_[Id(e->src())][e->src_output_()] += size

static Microseconds TimeEstimateForNode(CostModel* cost_model, Node* n) {

  CHECK(n->IsOp());
  if (IsConstant(n) || IsVariable(n))
    return Microseconds(0);
  return kDefaultTimeEstimate;

static void EstimateComputationCosts(const Graph& g, CostModel* cost_model) {

  for (Node* n : g.nodes())
    if (!n->IsOp()) continue;
    cost_model->RecordTime(n, TimeEstimateForNode(cost_model, n));

void InitFromGraph(const Graph& g)

为当前 cost_model 预留空间，将 g 的所有节点加入当前 cost_model

void AddToCostGraphDef(const Graph* graph, CostGraphDef* cost_graph)

将 graph 转换为 cost_graph

1. 遍历 g 的每个 node 的每个 edge
2. 将 control edge 加入 control_inputs, 其他加入 inputs
3. 对 control_inputs 根据 id 进行排序
4. 对于 control edge 增加控制输入, input 增加 InputInfo, 对于 output 增加 OutputInfo

Bytes MinTensorMemoryUsage(const TensorShapeProto& tensor_shape, const DataType& dtype) {

遍历 tensor_shape 的所有 dim 计算 Tensor 需要空间

### EdgeSet

EdgeSet()

设置 ptrs_ 中元素都为 null

### NodeOut

struct NodeOut
  Node* node;
  int index;

NodeOutHash : 根据 node, sizeof(node), index, 的 NodeOut 计算Hash
NodeOutEq : 判断两个 NodeOut 是相同


Node* AddZerosLike(Graph* g, NodeOut input)

1. g 中增加一个节点 node
2. g 中增加一个 Edge, 其中 input + input.index + node + 0

其中该节点
1. name : Func/_[NUM] (NUM 为一个数字)
1. input : input.name
2. Attr : T : input.type
3. Operation : ZerosLike

Node* AddSymGrad(Graph* g, Node* n, gtl::ArraySlice<NodeOut> grads) {

构建节点，操作为 SymbolicGradient, 输入为 n 的输入节点和 n 的 grads 节点与输出节点进行 AddN 操作之后的节点

其中:
1. name : Func/_[NUM] (NUM 为一个数字)
2. Operation : SymbolicGradient
3. input : node-> input 和 grads
4. Attr : Tin : 所有输入的类型
          Tout : n 的输入类型(gradient 中 node 的 out 与 node 的 input 的类型相同)
          f : n->attrs()

class SymbolicGradientBuilder
  gtl::ArraySlice<NodeOut> y_node_outputs_;
  gtl::ArraySlice<NodeOut> x_node_outputs_;
  gtl::ArraySlice<NodeOut> y_grad_node_outputs_;
  std::vector<NodeOut>* x_grad_node_outputs_;
  Graph* graph_;  // Not owned.
  //key 是 y_node_outputs_ 对应的所有节点, value 是 y_grad_node_outputs_ 中的节点
  //y_node_outputs_[i]  的每个输入节点都会保存 y_grad_node_outputs_[i] 到 backprops_ 中, 
  //也可以理解为 y_node_outputs_ 的输入节点有多少个输出， 每个输出对应的 y_grad_node_outputs_ 都会加入 backprops_
  typedef std::vector<NodeOut> BackpropedGradients;
  std::unordered_map<NodeOut, BackpropedGradients, NodeOutHash, NodeOutEq> backprops_;
  std::vector<int> pending_; 每个节点对应的访问的节点
  std::deque<Node*> ready_;
  std::unordered_set<int> stop_nodes_;


void BackpropAlongEdge(const NodeOut& dst_grad, const NodeOut& src)

1. 找到 src 对应的 grads
2. grads->push_back(dst_grad);
3. 如果 --pending_[src.node->id()] == 0, ready_.push_back(src.node);

void BackpropZerosAlongEdge(const NodeOut& src)

  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }

void SymbolicGradientBuilder::InitBackprop()

1. 将 x_node_outputs_ 中连接的每个节点(除去控制节点) node 加入 visited
2. 清空 backprops_
3. 设置 pending_ 保存每个节点直连的所有节点数
4. 将与  y_node_outputs_  直连的所有节点加入 ready_

NodeOut SumGradients(const NodeOut& src)

构建反向传播图, 以 src 的对应的 Grad 为输入构建 AddN 操作的 Graph
1. 找到 src 对应的 grads(类型为 BackpropedGradients)
2. 如果 grads.isEmpty()，graph_ 中增加一个节点 node, 与 src 连接, 返回。
3. 如果 grads.size() == 1，返回 grads[0]
4. 将 src 所有的输出节点作为输出节点进行 AddN 操作, 将 src 对应的 grad 节点与
   Add 作为 Edge 加入 grad


node :
1. name : Func/_[NUM] (NUM 为一个数字)
2. Operation : AddN
3. input : grads
4. Attr : N : grads.size()
          T : src.type()

Status Compute()

TODO

## graph_constructor

通过 GraphConstructor.Construct

1. 控制和非控制 edge 不能连接
2. 控制必须在最后

struct ImportGraphDefOptions
  string prefix; //不能为已经存在的某个 Node 的前缀
  std::map<TensorId, TensorId> input_map; //不能重复
  std::vector<string> control_dependencies; //不能重复
  std::vector<TensorId> return_tensors;

struct GraphConstructorOptions
  bool allow_internal_ops = false; //Node 中的 op 是否允许是内部的，即以 "_" 开头的 Operation
  bool expect_device_spec = false;

class GraphConstructor
  const Options opts_;
  const NodeDefSlice node_defs_;
  const VersionDef* versions_;
  const FunctionDefLibrary* library_;
  Graph* g_;
  const VersionDef original_versions_;
  ShapeRefiner* refiner_;
  std::vector<std::pair<Node*, int>>* return_tensors_; //
  std::unordered_map<StringPiece, NodeInfo, StringPiece::Hasher> gdef_nodes_; //所有节点, key = node->name() value= 节点在 node_def_ 中的索引
  std::unordered_map<StringPiece, Node*, StringPiece::Hasher> existing_nodes_; //所有节点, key = node->name() value = node
  std::vector<int> ready_; //保存node_defs_ 中输入节点的索引
  std::vector<int> pending_count_;
  std::vector<gtl::InlinedVector<int, 4>> outputs_; 索引为该节点在 node_defs_ 中的索引，保存的值为该节点所有输出节点在  node_defs_ 中的索引的集合
  std::vector<EdgeInfo> back_edges_;

  struct NodeInfo
    int gdef_index;
    Node* node;  // nullptr until the NodeDef is converted to a Node.

  struct InputInfo
    string name;
    Node* node;
    int index;

  struct EdgeInfo
    string src_name;
    int src_index;
    Node* dst_node;
    int dst_index;

  struct Options
    bool allow_internal_ops;
    bool expect_device_spec;
    string prefix;
    std::map<TensorId, TensorId> input_map;
    std::vector<string> control_dependencies;
    std::vector<TensorId> return_tensors;
    bool importing;

bool NodeNameInValues(const std::map<TensorId, TensorId>& input_map, const StringPiece& node_name)

从 input_map 中查找 node_name 是否存在

bool NodeNameInValues(const std::vector<string>& control_dependencies, const StringPiece& node_name)

从 control_dependencies 中查找 node_name 是否存在

Status EnsureNoNameCollisions()

1. 所有节点加入  existing_nodes_
2. prefix  要符合命名规范，且不能与某个节点名的前缀冲突

Status ValidateInputMapAndControlDependencies()

如果 opts_.input_map 和 opts_.control_dependencies 不存在于 existing_nodes_, 返回错误.
控制节点和非控制节点连接，返回错误

Status BuildNodeIndex()

1. 确保 node_defs_ 中每个节点的名字是有效的
2. 将 node_defs_ 中每个元素加入 gdef_nodes_
3. 确保 node_def.input 中控制节点在最后


void GraphConstructor::RemapNodeDefInputs(NodeDef* node_def, std::vector<bool>* input_already_exists)

 如果 node_def 的 input 在 ops_.input_map 中，


Status GraphConstructor::Convert()

1. 从输入节点开始遍历，一层一层遍历所有节点, 建立节点之间的 edge。
TODO

问题： back edge 是什么？

Status GraphConstructor::PopulateReturnTensors()

如果在 ops_.input_map 和 gdef_nodes_ 都没有找到，就返回错误
如果在 ops_.input_map  但在 gdef_nodes_ 中找到了， 加入 return_tensors_
如果在 ops_.input_map 中找到了， 加入  return_tensors_


std::unordered_set<string> GetNextIterationNodes(const GraphConstructor::NodeDefSlice& node_defs)

找到 op 是 NextIteration 或 RefNextInteration 的节点

Status InitFromEdges()

遍历所有的节点，
1. 初始化 pending_count_
1.1 如果 op 是 Merge 或 RefMerge 的节点，且该节点的输入节点中有节点的 op 是 NextIteration 或  RefNextInteration  那么，将该节点控制节点数加 1  加入 pending_count_
1.2 否则 该节点的输入节点个数 pending_count_
2. 初始化 ready_ : 如果节点没有输入节点，那就将该节点在  node_defs_ 中的索引加入 ready_
3. 初始化 outputs_ key 为保存该节点在 node_defs_ 中的索引，该节点所有输出节点的在 node_defs_ 中的索引集合为 value

Status ValidateColocationConstraints(const NodeDef& node_def)

1. node_def.attr() 中查找 kColocationAttrName 对应的 
TODO

Status GraphConstructor::MakeNode(const NodeDef& node_def, Node** node) {

1. g_ 中增加 node,
  if (opts_.expect_device_spec) (*node)->set_assigned_device_name(node_def.device());

其他方法

  TODO

void RemoveInputs(const std::vector<int>& inputs_to_remove, NodeDef* node_def, std::vector<bool>* input_already_exists)
void GraphConstructor::AddControlDependencies( NodeDef* node_def, std::vector<bool>* input_already_exists)
void GraphConstructor::AddPrefixToNodeDef( const std::vector<bool>& input_already_exists, NodeDef* node_def)
Status GraphConstructor::AddBackEdges()
Status GraphConstructor::UpdateVersionDef()
Status GraphConstructor::PopulateReturnTensors()
Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst, int input_index)

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts, const GraphDef& gdef, Graph* g)
Status ConvertNodeDefsToGraph(const GraphConstructorOptions& opts, gtl::ArraySlice<NodeDef> nodes, Graph* g)
Status ImportGraphDef(const ImportGraphDefOptions& opts, const GraphDef& gdef, Graph* g, ShapeRefiner* refiner, std::vector<std::pair<Node*, int>>* return_tensors)
void CopyGraph(const Graph& src, Graph* dest) {

### GraphDefBuilder

实现非常简单，关注类型变化

Node* SourceOp(const string& op_name, const GraphDefBuilder::Options& opts)

ops -> NodeBuilder -> Node

Node* UnaryOp(const string& op_name, NodeOut input, const GraphDefBuilder::Options& opts);

在 SourceOp 基础上增加了 input

Node* BinaryOp(const string& op_name, NodeOut a, NodeOut b, const GraphDefBuilder::Options& opts);

在 SourceOp 基础上增加了两个 input

### Graph_partition

struct ControlLoop
  Node* enter = nullptr;
  Node* merge = nullptr;
  Node* switch_node = nullptr;

bool NeedSameDeviceSendRecv(const Edge* edge, const GraphInfo& info)

edge 不是 ControlEdge, 并且 edge->src 和 edge->dst 的 MemoryType 不同

bool IsDstInputOnHost(const Edge* edge, const GraphInfo& info)

edge->dst 的类型是否为 HOST_MEMORY

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot)

src_name 和 src_slot 作为 input 加入 dst

void AddReadControl(const std::vector<NodeDef*>& recvs, const std::vector<string>& inputs)

将 inputs 中每个元素加入 recvs 的每个节点中

void SetSendRecvAttrs(const PartitionOptions& opts, const Edge* edge, NodeDefBuilder* builder)

给 builder 增加属性 tensor_name, send_device, send_device_incarnation, recv_device, client_terminated

NodeDef* AddSend(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge,
                 NodeDefBuilder::NodeOut send_from, int64 start_time, Status* status)

gdef 增加两个节点:
    NodeBuilder :
        Name: src->name()
        Operation: "_HostCast" or "Cast"
        Device : src->assigned_device_name(),
        Input: send_from
        Attr :  "_start_time", start_time
                "DstT", cast_dtype
    NodeBuilder :
        Name: src->name()
        Operation: "_HostSend" or "_Send"
        Device : src->assigned_device_name(),
        Input: send_from
        Attr :  "_start_time", start_time
    其中 Node* src = edge->src();

NodeDef* AddRecv(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge, NodeDef** real_recv, Status* status)

gdef 增加一个节点:
    if dtype != cast_dtype
    NodeBuilder :
        Name: src->name()
        Operation: "_HostCast" or "Cast"
        Device : dst->assigned_device_name(),
        Attr :  "_start_time", start_time
                "DstT", cast_dtype
    else if edge->IsControlEdge()
    NodeBuilder :
        Name: src->name()
        Operation: "Identity"
        Device : dst->assigned_device_name(),
        Attr :  "_start_time", start_time
                "DstT", cast_dtype
    else
    NodeBuilder :
        Name: src->name()
        Operation: "_HostRecv" : "_Recv"
        Device : dst->assigned_device_name(),
        Attr :  "tensor_type", cast_dtype
                "_start_time", start_time
                "DstT", cast_dtype
    其中
        Node* src = edge->src();
        Node* dst = edge->dst();

NodeDef* AddDummyConst(const PartitionOptions& opts, GraphDef* gdef, const Edge* edge, Status* status)

gdef 增加一个节点:
    NodeBuilder :
        Name: src->name()
        Operation: "Const"
        Device : src->assigned_device_name(),
        Attr :  "dtype", DT_FLOAT
                "value", tensor(DT_FLOAT, TensorShape({0}))
                "DstT", cast_dtype

NodeDef* AddControlTrigger(const PartitionOptions& opts, GraphDef* gdef,
                           const string& assigned_device_name, int64 epoch,
                           int64 starttime, Status* status)

gdef 增加一个节点:
    NodeBuilder :
        Name: synch_epoch
        Operation: "ControlTrigger"
        Device : assigned_device_name
        Input: send_from
        Attr :  "_start_time", starttime

void OptimizeControlFlowColocation(Graph* graph)
  DFS(*graph, visit, {});

Node* AddControlEnter(Graph* g, const string& node_name,
                      const string& device_name, const string& frame_name,
                      const int parallel_iterations, Status* status)

创建一个 Node 并返回
    NodeBuilder :
        Name: node_name
        Operation: "Enter"
        Input: {"dummy", 0, DT_FLOAT}
        Attr :  "frame_name", frame_name
                "parallel_iterations", parallel_iterations

Node* AddControlMerge(const string& in_name1, const string& in_name2, Graph* g,
                      const string& node_name, const string& device_name,
                      Status* status)

创建一个 Node 并返回
    NodeBuilder :
        Name: node_name
        Operation: "Merge"
        Input: {in_name1, 0, DT_FLOAT}, {in_name2, 0, DT_FLOAT}}

Node* AddControlSwitch(NodeBuilder::NodeOut input1, NodeBuilder::NodeOut input2,
                       const string& device_name,
                       const GraphDefBuilder::Options& bopts)

创建一个有两个输入的 Node 并返回
    NodeBuilder :
        Name: opts.GetNameForOp("Switch")
        Operation: "Switch"
        Input: input1, input2

Node* AddControlNext(NodeBuilder::NodeOut input, const string& device_name,
                     const GraphDefBuilder::Options& bopts)

创建一个有一个输入的 Node 并返回
    NodeBuilder :
        Name: node_name
        Operation: "NextIteration"
        Input: input

Node* EmptyConst(const GraphDefBuilder::Options& options)

创建一个有一个输入的 Node 并返回
    NodeBuilder :
        Name: Const
        Operation: "Const"
        Attr:   "value", proto
                "dtype", DataTypeToEnum<float>::v()

Node* AddControlConst(const string& device_name,
                      const GraphDefBuilder::Options& bopts)

  Node* res_node = EmptyConst(bopts);
  res_node->set_assigned_device_name(device_name);

void AddControlFlowInfo(const Node* node, const Node* src, std::vector<ControlFlowInfo>* cf_info)

用 cf_info[src->id()] 初始化 cf_info[node->id()]

Status AddControlLoop(const PartitionOptions& opts, Graph* g, const Node* src,
                      const Edge* edge, Node* loop_cond, std::vector<ControlFlowInfo>* cf_info, ControlLoop* loop)

    const string& next_name = ControlLoopName(opts.new_name(edge->dst()->name()));

    enter:
        Name: enter_name
        Operation: "Enter"
        Input: {"dummy", 0, DT_FLOAT}
        Attr :  "frame_name", frame_name
                "parallel_iterations", parallel_iterations

    merge:
        Name: merge_name
        Operation: "Merge"
        Input: {enter_name, 0, DT_FLOAT}, {next_name, 0, DT_FLOAT}}

    switch_node:
        Name: opts.GetNameForOp("Switch") Operation: "Switch"
        Input: merge, loop_cond

    next:
        Name: opts.GetNameForOp("NextIteration")
        Operation: "NextIteration"
        Input: {switch_node, 1}

    cf_info[src->id()] 初始化 cf_info[enter->id()]
    cf_info[src->id()] 初始化 cf_info[meger->id()]
    cf_info[src->id()] 初始化 cf_info[switch_node->id()]
    cf_info[src->id()] 初始化 cf_info[node->id()]

    enter->0 ---> 0->merge
    next->0 ---> 1->merge
    loop->enter = enter;
    loop->merge = merge;
    loop->switch_node = switch_node;

Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info)

    MemoryTypesForNode(g.op_registry(), DeviceType(parsed.type), node->def(), &input_memory_types, &output_memory_types)
    用 input_memory_types 初始化 info->input_types
    用 output_memory_types 初始化 info->output_types

Status AddControlFlow(const PartitionOptions& opts, Graph* g, GraphInfo* g_info)

TODO, 关键函数
1. 遍历 g 的所有节点, 对于是 NC_LOOP_COND 的节点，建立 frame_name 到 Node 的映射关系
2. 遍历g 的所有 Edge 关联的节点, 去掉 Source, Sink Node, 源和目的设备不同，但是
   src_frame_name 和 dst_frame_name 相同的节点，增加 ControlEdge
3. 遍历g 的所有 Edge 关联的节点, 去掉 Source, Sink Node, 源和目的设备不同，但是
   src_frame_name 和 dst_frame_name 相同的节点，增加 ControlEdge


Status TopologicalSortNodesWithTimePriority(
    const GraphDef* gdef, std::vector<std::pair<const NodeDef*, int64>>* nodes,
    std::unordered_map<const NodeDef*, int64>* node_to_start_time_out)

2. node_to_output_nodes, node_to_start_time, inputs_needed

TODO

### NodeBuilder

  NodeDefBuilder def_builder_;
  std::vector<NodeOut> inputs_;
  std::vector<Node*> control_inputs_;
  std::vector<string> errors_;

NodeBuilder -> NodeDefBuilder -> g->AddNode(g->AddEdge, graph->AddControlEdge)

### optimizer_cse

子表达式消除

### TensorId

pair<StringPiece, int>

将 ^name, name:digits 或 name 转换为 TensorId 其中

* ^name : TensorId(name, Graph::kControlSlot)
* name:digits : TensorId(name, index)
* name : TensorId(name, 0)

### testlib

Node* Send(Graph* g, Node* input, const string& tensor, const string& sender,
           const uint64 sender_incarnation, const string& receiver);

构造如下节点，并加入 g

    NodeBuilder :
        Name: g->NewName("n")
        Operation: "_Send"
        Input: input, 0
        Attr :
             "tensor_name", tensor
             "send_device", sender
             "send_device_incarnation", static_cast<int64>(sender_incarnation)
             "recv_device", receiver

Node* Recv(Graph* g, const string& tensor, const string& type,
           const string& sender, const uint64 sender_incarnation,
           const string& receiver);

    NodeBuilder :
        Name: g->NewName("n")
        Operation: "_Recv"
        Input: input, 0
        Attr :
            "tensor_type", dtype
            "tensor_name", tensor
            "send_device", sender
            "send_device_incarnation", static_cast<int64>(sender_incarnation)
            "recv_device", receiver

Node* Constant(Graph* g, const Tensor& tensor)

    NodeBuilder :
        Name: g->NewName("n")
        Operation: "Const"
        Attr :
            "dtype", tensor.dtype()
            "value", tensor

Node* Constant(Graph* g, const Tensor& tensor, const string& name)

    NodeBuilder :
        Name: name
        Operation: "Const"
        Attr :
            "dtype", tensor.dtype()
            "value", tensor

Node* HostConstant(Graph* g, const Tensor& tensor)

    NodeBuilder :
        Name: g->NewName("n")
        Operation: "HostCost"
        Attr :
            "dtype", tensor.dtype()
            "value", tensor

Node* HostConstant(Graph* g, const Tensor& tensor, const string& name)

    NodeBuilder :
        Name: name
        Operation: "HostCost"
        Attr :
            "dtype", tensor.dtype()
            "value", tensor

Node* Var(Graph* g, const DataType dtype, const TensorShape& shape)

    NodeBuilder :
        Name: g->NewName("n")
        Operation: "Variable"
        Attr :
            "dtype", dtype
            "shape", shape

Node* Var(Graph* g, const DataType dtype, const TensorShape& shape, const string& name)

    NodeBuilder :
        Name: name
        Operation: "Variable"
        Attr :
            "dtype", dtype
            "shape", shape

Node* Assign(Graph* g, Node* var, Node* val)

    NodeBuilder :
        Name:  g->NewName("n")
        Operation: "Assign"
        Input: var, val
        Attr :
            "use_locking", true

Node* Reduce(Graph* g, const string& reduce, Node* data, Node* axes, bool keep_dims)

    NodeBuilder :
        Name:  g->NewName("n")
        Operation: reduce
        Input: data, axes
        Attr :
            "keep_dims", keep_dims

Node* QuantizeToUINT8(Graph* g, Node* data)

    NodeBuilder :
        Name:  g->NewName("n")
        Operation: Quantize
        Input: data
        Attr :
            "T", DT_QUINT8
            "max_range", 1.0f
            "min_range", -1.0f

Node* Matmul(Graph* g, Node* in0, Node* in1, bool transpose_a, bool transpose_b)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: MatMul
        Input: in0, in1
        Attr :
            "transpose_a", transpose_a
            "transpose_b", transpose_b

Node* BatchMatmul(Graph* g, Node* in0, Node* in1, bool adj_x, bool adj_y)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: BatchMatMul
        Input: in0, in1
        Attr :
            "adj_x", adj_x
            "adj_y", adj_y

Node* RandomNumberGenerator(const string& op, Graph* g, Node* input, DataType dtype)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: op
        Input: input
        Attr :
            "dtype", dtype
            "seed", 0

Node* RandomUniform(Graph* g, Node* input, DataType dtype)
    RandomNumberGenerator("RandomUniform", g, input, dtype);

Node* RandomGaussian(Graph* g, Node* input, DataType dtype)
    RandomNumberGenerator("RandomStandardNormal", g, input, dtype);

Node* TruncatedNormal(Graph* g, Node* input, DataType dtype)
    RandomNumberGenerator("TruncatedNormal", g, input, dtype);

Node* RandomGamma(Graph* g, Node* shape, Node* alpha)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: RandomGamma
        Input: shape, alpha
        Attr :
            "seed", 0

Node* RandomPoisson(Graph* g, Node* shape, Node* lam)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: RandomPoisson
        Input: shape, lam
        Attr :
            "seed", 0

Node* Unary(Graph* g, const string& func, Node* input, int index)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: func
        Input: input, index

Node* Binary(Graph* g, const string& func, Node* in0, Node* in1)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: func
        Input: in0, in1

Node* Multi(Graph* g, const string& func, gtl::ArraySlice<Node*> ins)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: func
        Input: ins

Node* Identity(Graph* g, Node* input, int index)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: Identity
        Input: input, index

Node* Add(Graph* g, Node* in0, Node* in1)
    Binary(g, "Add", in0, in1);

Node* Reverse(Graph* g, Node* tensor, Node* axis)
    Binary(g, "ReverseV2", tensor, axis);

Node* Error(Graph* g, Node* input, const string& errmsg)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: Error
        Input: input
        Attr : message : errmsg

Node* InvalidRefType(Graph* g, DataType out_type, DataType invalid_type)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: InvalidRefType
        Attr :
            Tin: out_type
            Tout: invalid_type

Node* Delay(Graph* g, Node* input, Microseconds delay_micros)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Delay"
        Input: input
        Attr : micros: delay_micros

Node* NoOp(Graph* g, const std::vector<Node*>& control_inputs)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "NoOp"
        ControlInput: control_inputs

Node* Switch(Graph* g, Node* in0, Node* in1)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Switch"
        Input: in0, in1

Node* Enter(Graph* g, Node* input, const string& frame_name) {

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Enter"
        Input: input
        Attr : "frame_name" : frame_name

Node* Exit(Graph* g, Node* input)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Exit"
        Input: input

Node* Merge(Graph* g, Node* in0, Node* in1) {

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Merge"
        Input: in0, in1

Node* Merge(Graph* g, Node* in0, gtl::ArraySlice<string> remaining_in) {

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Merge"
        Input: in0,
            remaining_in[0], 0, in0.dt
            remaining_in[1], 0, in0.dt
            ...
            remaining_in[n], 0, in0.dt

Node* Concat(Graph* g, Node* concat_dim, gtl::ArraySlice<Node*> tensors)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Concat"
        Input: concat_dim
            tensors[0]
            tensors[1]
            ...
            tensors[n]

Node* ConcatV2(Graph* g, gtl::ArraySlice<Node*> tensors, Node* concat_dim)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "ConcatV2"
        Input:
            tensors[0]
            tensors[1]
            ...
            tensors[n]
            concat_dim

Node* Next(Graph* g, const string& name, Node* input)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "NextIteration"
        Input:
            input

Node* LoopCond(Graph* g, Node* input)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "LoopCond"
        Input:
            input

Node* Less(Graph* g, Node* in0, Node* in1)
    Binary(g, "Less", in0, in1);

Node* Select(Graph* g, Node* c, Node* inx, Node* iny)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Select"
        Input:
            c
            inx
            iny

Node* Cast(Graph* g, Node* in, DataType dst)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Cast"
        Input: in
        Attr: "DstT" : dst

Node* Gather(Graph* g, Node* in0, Node* in1, Node* axis)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "GatherV2"
        Input: in0, in1, axis

Node* GetSessionTensor(Graph* g, Node* in) {

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "GetSessionTensor"
        Input: in
        Attr: "dtype", DT_FLOAT

Node* Relu(Graph* g, Node* in)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Relu"
        Input: in
        Attr: "T", DT_FLOAT

Node* Relu6(Graph* g, Node* in)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Relu6"
        Input: in
        Attr: "T", DT_FLOAT

Node* BiasAdd(Graph* g, Node* value, Node* bias)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "BiasAdd"
        Input: value, bias
        Attr: "T", DT_FLOAT

Node* Conv2D(Graph* g, Node* in0, Node* in1)

    NodeBuilder
        Name:  g->NewName("n")
        Operation: "Conv2D"
        Input: in0, in1 
        Attr: "T", DT_FLOAT
            "strides", {1, 1, 1, 1}
            "padding", "SAME"

void ToGraphDef(Graph* g, GraphDef* gdef)
    g->ToGraphDef(gdef);


### graph partition

给定一个 Graph:

如果有一个 Node 的名字以 "_cloop" 开头, 那么
该  Graph 必然有 Enter, Merge, Switch, NextInteration 四个
op 的 Node

如果有一个 Node 的 op 是 "_Recv"，那么该 Node 的输入中
必然有一个 Node 以 "^" 开头



struct PartitionOptions {
  // A function that returns a location for the execution of a given
  // Node.
  typedef std::function<string(const Node*)> NodeToLocFunc;
  NodeToLocFunc node_to_loc = nullptr;

  // A function that returns a unique graph node name with the given
  // prefix.
  typedef std::function<string(const string&)> NewNameFunc;
  NewNameFunc new_name = nullptr;

  // A function that returns the incarnation of a device given the
  // device's fullname. If not found, GetIncarnationFunc should return
  // kIllegalIncarnation.
  static const uint64 kIllegalIncarnation = 0;
  typedef std::function<uint64(const string&)> GetIncarnationFunc;
  GetIncarnationFunc get_incarnation = nullptr;

  // If specified, flib_def defines a function library that should be
  // partitioned and replicated into each resulting partition graphs.
  const FunctionLibraryDefinition* flib_def = nullptr;

  // True if all the control flow "code" has already been added. The
  // control flow code needs to be added when we still have the entire
  // graph before any partitioning. So this flag should be false for
  // the first partitioning but true for all subsequent partitioning.
  //
  // TODO(yuanbyu): We could also make the addition of the control
  // flow code incremental based on 'node_to_loc'. This makes the
  // communication a broadcast tree, which could be more efficient when
  // the number of participating devices is large.
  bool control_flow_added = false;

  // A function that returns the data type into which the tensor
  // should be cast before sent over the wire.
  typedef std::function<DataType(const Edge*)> ShouldCastFunc;
  ShouldCastFunc should_cast = nullptr;

  // Schedule the execution of the recvs based on their start times
  // computed by some scheduling algorithm. The recvs are divided into
  // epochs based on their start times. A recv is enabled only when
  // execution reaches its epoch - N for some predefined N.
  bool scheduling_for_recvs = false;
  // The start time for each node in the graph computed by some scheduling
  // algorithm. If 'need_to_record_start_times' is true, we record them
  // in the graph as a node attribute.
  bool need_to_record_start_times = false;
  std::vector<Microseconds> start_times;

struct GraphInfo
  std::vector<DeviceType> device_types;
  MemoryTypeMap input_types;
  MemoryTypeMap output_types;
  std::vector<ControlFlowInfo> cf_info;


struct DupRecvKey
  int src_node_id;           // Edge's src node id
  int src_output_slot;       // Edge's src node output slot
  GraphDef* dst_graph;       // Edge's dst node is in this subgraph
  bool recv_output_on_host;  // The output of recv is on host

struct RecvInfo
  NodeDef* recv;
  NodeDef* real_recv;
  int64 start_time;

struct ControlLoop
  Node* enter = nullptr;
  Node* merge = nullptr;
  Node* switch_node = nullptr;


typedef std::unordered_map<DupRecvKey, RecvInfo, DupRecvKeyHash, DupRecvKeyEq> DupRecvTable;


bool NeedSameDeviceSendRecv(const Edge* edge, const GraphInfo& info)

同时满足以下四个条件才返回 true
0. edge 不是 control edge
1. edge 的 src node 和  dst node  的 device_name 相同
2. src node 的 device type 不在 CPU
3. edge 的两端分别在不同的 MemoryType

bool IsDstInputOnHost(const Edge* edge, const GraphInfo& info)

满足以下任一条件返回 true
1. edge 的 dst 设备类型为不是 CPU, edge 的 dst, dst_input 在 HOST_MEMORY
2. edge 的 dst 设备类型为是 CPU

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot)

dst 增加 input src_name:src_slot

void AddReadControl(const std::vector<NodeDef*>& recvs, const std::vector<string>& inputs)

将 inputs 每一个以 control input 的方式加入每一个 recvs

void SetSendRecvAttrs(const PartitionOptions& opts, const Edge* edge, NodeDefBuilder* builder)

设置 builder 属性
tensor_name : edge_"edge->id()"_"edge->src()->name()"
send_device : edge->src()->assigned_device_name()
send_device_incarnation : opts.get_incarnation(edge->src()->assigned_device_name())
recv_device : edge->dst()->assigned_device_name()
client_terminated : false

NodeDef* AddSend(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge,
                 NodeDefBuilder::NodeOut send_from, int64 start_time,
                 Status* status)

gdef 增加一个 node:
name: src->name()
op : "_HostSend"  或 "_Send"
device : src->assigned_device_name()
input : send_from
attr : `_start_time`: start_time
       tensor_name : edge_"edge->id()"_"edge->src()->name()"
       send_device : edge->src()->assigned_device_name()
       send_device_incarnation : opts.get_incarnation(edge->src()->assigned_device_name())
       recv_device : edge->dst()->assigned_device_name()
       client_terminated : false

如果类型不满足，还可能需要增加 Cast 类型的节点


NodeDef* AddRecv(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge, NodeDef** real_recv,
                 Status* status) {

gdef 增加一个 node:
name: src->name()
op : "_HostRecv"  或 "_Recv"
device : edge->dst()->assigned_device_name()
input : send_from
attr : `_start_time`: start_time
       tensor_type : EdgeType(edge)
       tensor_name : edge_"edge->id()"_"edge->src()->name()"
       send_device : edge->src()->assigned_device_name()
       send_device_incarnation : opts.get_incarnation(edge->src()->assigned_device_name())
       recv_device : edge->dst()->assigned_device_name()
       client_terminated : false

如果 edge 是 control edge, 需要增加 Identity Node
如果 edge 是 类型需要转换, 需要增加 Cast Node

NodeDef* AddDummyConst(const PartitionOptions& opts, GraphDef* gdef, const Edge* edge, Status* status)

增加节点:
name: src->name()
op : "Const"
device : edge->src()->assigned_device_name()
attr : dtype: start_time
       value: DT_FLOAT, TensorShape({0});

NodeDef* AddControlTrigger(const PartitionOptions& opts, GraphDef* gdef,
                           const string& assigned_device_name, int64 epoch,
                           int64 starttime, Status* status) 

增加节点:
name: synch_"epoch"
op : "ControlTrigger"
device : assigned_device_name
attr : `_start_time`: starttime

void OptimizeControlFlowColocation(Graph* graph)

TODO

Node* AddControlEnter(Graph* g, const string& node_name,
                      const string& device_name, const string& frame_name,
                      const int parallel_iterations, Status* status)


增加节点:
name: node_name
op : "Enter"
device : device_name
input : {"dummy", 0, DT_FLOAT}
attr :  frame_name: frame_name
        parallel_iterations: parallel_iterations

Node* AddControlMerge(const string& in_name1, const string& in_name2, Graph* g,
                      const string& node_name, const string& device_name,
                      Status* status)
增加节点:
name: node_name
op : "Merge"
device : device_name
input : {"in_name1", 0, DT_FLOAT}, {"in_name2", 0, DT_FLOAT}

Node* AddControlSwitch(NodeBuilder::NodeOut input1, NodeBuilder::NodeOut input2,
                       const string& device_name, const GraphDefBuilder::Options& bopts)

增加节点:
name: TODO
op : "Switch"
device : device_name
input : input1, input2

Node* AddControlNext(NodeBuilder::NodeOut input, const string& device_name,
                     const GraphDefBuilder::Options& bopts)

增加节点:
name: TODO
op : "NextIteration"
device : device_name
input : input

Node* EmptyConst(const GraphDefBuilder::Options& options)

增加节点:
name: TODO
op : "Const"
attr : dtype : DT_FLOAT
       value : DT_FLOAT, TensorShape({0});

Node* AddControlConst(const string& device_name, const GraphDefBuilder::Options& bopts)

增加节点:
name: TODO
op : "Const"
device : device_name
attr : dtype : DT_FLOAT
       value : DT_FLOAT, TensorShape({0});

void AddControlFlowInfo(const Node* node, const Node* src, std::vector<ControlFlowInfo>* cf_info)

用  cf_info[src->id()] 设置 cf_info[node->id()]

Status AddControlLoop(const PartitionOptions& opts, Graph* g, const Node* src,
                      const Edge* edge, Node* loop_cond,
                      std::vector<ControlFlowInfo>* cf_info, ControlLoop* loop)

enter
next  -> merge
         loop_cond -> switch -> next

1. 创建 enter, next, merge, loop_cond, switch 节点
2. enter, merge, switch, next 的 ControlFlowInfo 都为 src
3. meger 的 input 设置为 enter, next; enter 和 next 的 output edge 设置为 merge
4. 设置 loop 的 enter, merger switch_node

增加节点

enter
name: `_loop`opts.new_name(edge->dst()->name())
op : "Enter"
device : device_name
input : {"dummy", 0, DT_FLOAT}
attr :  frame_name: frame_name
        parallel_iterations: (*cf_info)[src->id()].frame->attrs()

merge
name: `_loop`opts.new_name(edge->dst()->name())
op : "Merge"
device : device_name
input : {"enter", 0, DT_FLOAT}, {"next", 0, DT_FLOAT}

switch
name: `_loop`opts.new_name(edge->dst()->name())
op : "Switch"
device : device_name
input : {"merge", 0, DT_FLOAT}, {"next", 0, DT_FLOAT}

Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info) {

遍历 g 中所有节点，设置 info

const Node* InputFrame(const Node* node, const std::vector<ControlFlowInfo>& cf_info)

获取 node 的输入帧, 对于非 Enter Node, 就是本身，对于 Enter 为 cf_info[node->id()].parent_frame;

const Node* OutputFrame(const Node* node, const std::vector<ControlFlowInfo>& cf_info)

获取 node 的输出帧, 对于非 Exit Node, 就是本身，对于 Exit 为 cf_info[node->id()].parent_frame;

Status AddControlFlow(const PartitionOptions& opts, Graph* g, GraphInfo* g_info)

TODO

Status TopologicalSortNodesWithTimePriority(const GraphDef* gdef, std::vector<std::pair<const NodeDef*, int64>>* nodes,
    std::unordered_map<const NodeDef*, int64>* node_to_start_time_out) {

1. 遍历所有节点
1.1 初始化 node_to_output_nodes， 保存了节点:节点的输出节点的映射关系
1.2 初始化 inputs_needed 保存 节点:节点输入节点个数
1.3 初始化 node_to_start_time 保存 节点：节点开始时间
1.4 将所有没有输入的节点加入 enqueue
2. 如果当前节点是  Merge，当前节点的输入是 inputs_needed 减一
3. 从第一层节点开始，利用广度遍历算法， 遍历所有节点
nodes 保存了  node:node->start_time 的映射关系
node_to_start_time_out : 保存了 node: node 的开始时间的映射关系，确保任一节点的输出节点的 start_time 大于该节点

Status AddControlEdges(const PartitionOptions& opts, std::unordered_map<string, GraphDef>* partitions)

遍历 partitions 中每个 graph
1. 增加 ControlTrigger 节点，节点的输入是
TODO

void SetIncarnation(const PartitionOptions& opts, NodeDef* ndef)

获取  ndef 的属性 send_device_incarnation, 如果没有 获取到，用 ops.get_incarnation[node->send_device]

void SetIncarnation(const PartitionOptions& opts, GraphDef* gdef)

设置  gdef 中所有节点的  send_device_incarnation

Status Partition(const PartitionOptions& opts, Graph* g, std::unordered_map<string, GraphDef>* partitions)

TODO
2. 遍历当前节点的所有输入节点
2.1
2.2 获取当前节点和输入节点的 `_start_time`, 分布记录为 send_start_time, recv_start_time


### GraphRunner

Status GraphRunner::Run(Graph* graph, FunctionLibraryRuntime* function_library,
                        const NamedTensorList& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs)

1. 遍历 inputs  通过 SimpleRendezvous 将每个 Tensor 发送出去（ 实际是将每个 Tensor 加入一个 map 中）
2. 将修改 graph 中的在 input_names 和 output_names 的节点，重新构建子图。参考 subgraph
3. 创建一个  Local Executor，执行
4. 遍历 output_names 通过 SimpleRendezvous 将每个 Tensor   收到的 Tensor 加入 outputs


### GraphMgr

Status GraphMgr::DecorateAndPublishGraphForDebug(const DebugOptions& debug_options, Graph* graph, Device* device)

Status GraphMgr::InitItem(const string& session, const GraphDef& gdef,
    const GraphOptions& graph_options, const DebugOptions& debug_options, Item* item)

1. 将 gdef 转换为 graph
2. 调用 Partition 将 graph 以 device_name 划分为 partitions
3. 将 participants  中的每个图转化为 device_graph，保存在 partition_graph
4. 对 partition_graph 进行 POST_PARTITIONING 优化(MklLayoutRewritePass,MklToTfConversionPass)
5. 遍历 partition_graph 的每个元素，初始化
5.2 调用 optimizer(graph_options.optimizer_opts) 优化
5.3 确保 g 中每个 Edge 的 MemoryType 是一样的，对于不能兼容的 Edge，将 Edge 修改为 "e->src+e->src_ouput -> `_Send` -> `_Recv` -> e->dst+e->dst_input"
5.4 初始化 unit

Status GraphMgr::Register(const string& session, const GraphDef& gdef,
                          const GraphOptions& graph_options,
                          const DebugOptions& debug_options, string* handle)

创建  item 加入 table_

Status GraphMgr::Deregister(const string& handle)

找到  handle 对应的 item，从 table_ 中删除

Status GraphMgr::DeregisterAll()

清空  table_ 的元素

Status GraphMgr::SendInputsToRendezvous(Rendezvous* rendezvous, const NamedTensors& in)

用 rendezvous 将  in 中的 tensor 发送出去

Status GraphMgr::RecvOutputsFromRendezvous(Rendezvous* rendezvous, NamedTensors* out)

用  rendezvous 接收  out 中对应的 tensor

void GraphMgr::RecvOutputsFromRendezvousAsync(Rendezvous* rendezvous, NamedTensors* out, const StatusCallback& done)

用 rendezvous 从 out 中 key  对应的  Tensor, 重置  out 中  key 对应的
value，收完 out 所有元素后，调用  done

Status GraphMgr::SendInputs(const int64 step_id, const NamedTensors& in)

获取  step_id 对应的  rendezvous，将  in 中的 tensor 发送出去

Status GraphMgr::RecvOutputs(const int64 step_id, NamedTensors* out)

获取  step_id 对应的  rendezvous，将 out 中的 tensor 接受保存到 out

void GraphMgr::RecvOutputsAsync(const int64 step_id, NamedTensors* out, StatusCallback done)

获取  step_id 对应的 rendezvous，从 out 中 key  对应的  Tensor, 重置  out 中  key 对应的
value，收完 out 所有元素后，调用  done

void GraphMgr::ExecuteAsync(const string& handle, const int64 step_id,
                            WorkerSession* session, const ExecutorOpts& opts,
                            StepStatsCollector* collector,
                            MutableRunGraphResponseWrapper* response,
                            CancellationManager* cancellation_manager,
                            const NamedTensors& in, StatusCallback done)

1. 找到  step_id 对应的 rendezvous
2. 将 in 中的  tensor  通过  rendezvous 发送出去
3. 找到 handle 对应的 item
4.  遍历 item 中 unit 的并执行


void GraphMgr::BuildCostModel(Item* item, StepStatsCollector* collector, CostGraphDef* cost_graph)

TODO
