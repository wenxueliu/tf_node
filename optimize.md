
## 数据结构

GraphOptimizationPassOptions -> GraphOptimizationPass

OptimizationPassRegistry* global_optimization_registry //保持所有注册的 GraphOptimizationPass

struct GraphOptimizationPassOptions
  string session_handle;
  const SessionOptions* session_options = nullptr;
  const CostModel* cost_model = nullptr;
  FunctionLibraryDefinition* flib_def = nullptr;  // Not owned.
  const DeviceSet* device_set = nullptr;  // Not owned.
  std::unique_ptr<Graph>* graph = nullptr;
  std::unordered_map<string, std::unique_ptr<Graph>>* partition_graphs = nullptr

class GraphOptimizationPass
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;

class OptimizationPassRegistry
  enum Grouping
    PRE_PLACEMENT,          // after cost model assignment, before placement.
    POST_PLACEMENT,         // after placement.
    POST_REWRITE_FOR_EXEC,  // after re-write using feed/fetch endpoints.
    POST_PARTITIONING,      // after partitioning
  typedef std::map<int, std::vector<std::unique_ptr<GraphOptimizationPass>>> GraphOptimizationPasses;
  std::map<Grouping, GraphOptimizationPasses> groups_;

class OptimizationPassRegistration
  OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass)
    OptimizationPassRegistry::Global()->Register(grouping, phase, std::move(pass));

REGISTER_OPTIMIZATION(grouping, phase, optimization) //将 optimization 加入 `groups_[grouping][phase]`
  static optimization_registration::OptimizationPassRegistration
      register_optimization___COUNTER__(grouping, phase, std::unique_ptr<GraphOptimizationPass>(new optimization))

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 20, EncapsulateSubgraphsPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 30, BuildXlaLaunchOpsPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0, ParallelConcatRemovePass)
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0, ResourceVariableReadPass)
const OptimizationPassRegistry::Grouping kMklLayoutRewritePassGroup = OptimizationPassRegistry::POST_PARTITIONING;
REGISTER_OPTIMIZATION(kMklLayoutRewritePassGroup, 1, MklLayoutRewritePass)
const OptimizationPassRegistry::Grouping kMklTfConvPassGroup = OptimizationPassRegistry::POST_PARTITIONING;
REGISTER_OPTIMIZATION(kMklTfConvPassGroup, 2, MklToTfConversionPass)

## 源码分析

### OptimizationPassRegistry

OptimizationPassRegistry* OptimizationPassRegistry::Global()

初始化全局变量 global_optimization_registry

void OptimizationPassRegistry::Register(Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass)

pass 加入 `groups_[grouping][phase]`

Status OptimizationPassRegistry::RunGrouping(Grouping grouping, const GraphOptimizationPassOptions& options)

找到 grouping 中的所有  GraphOptimizationPass，调用对应的 Run(options) 方法


### 常量展开

如果图中某个节点的依赖都是常量， 在 CPU Device  上将计算后的结果替代已有的节点，这样可以节省计算资源的消耗。
与计算机语言中的常量优化的思路是一样的


void AddNodeToConstantGraph(Node* n, std::unordered_map<Node*, std::vector<Node*>>* node_map, Graph* constant_graph)


Graph* GetConstantGraph(const Graph* orig_graph, const std::vector<Node*>& nodes,
    const std::unordered_map<const Node*, std::vector<Tensor>>& shape_replacement_map,
    std::map<NodeAndOutput, Node*>* tensors_to_fetch)

遍历 orig_graph

bool IsConstantFoldable(const Node* n, const std::unordered_map<const Node*, std::vector<PartialTensorShape>>*
        shape_map, const std::function<bool(const Node*)>& consider,
        std::unordered_map<const Node*, std::vector<Tensor>>* shape_replacement_map)

返回 n 是否是一个常量节点

第一种情况 n 是 IsConstant()
第二种情况 n 同时满足如下条件:
1. 无状态的 stateful 为 false
2. 不是 IsSource() IsSink() IsSwitch() IsMerge() IsEnter() IsExit() IsNextIteration() IsGetSessionHandle() IsGetSessionTensor() IsDeleteSessionTensor()
3. n->def 没有注册 CPU 类型的 OpKernel
4. consider(n) 返回 true

struct ConstantFoldingOptions
  // If "consider" is not a nullptr, then only constant fold a node "n" if consider(n) returns true.
  std::function<bool(const Node*)> consider = nullptr;
  // If shape_map is not a nullptr, it is a map from node n to a vector of the (potentially partially-known) shapes of its outputs.
  const std::unordered_map<const Node*, std::vector<PartialTensorShape>>* shape_map;

1. 如果 n 是一个常量节点

void ConsiderConstantFoldableNode(
    Node* n, const ConstantFoldingOptions& opts, std::vector<Node*>* nodes,
    std::unordered_map<const Node*, gtl::FlatSet<Node*>>* constant_control_deps,
    std::unordered_map<const Node*, std::vector<Tensor>>* shape_replacement_map,
    bool* internal_node_inserted)

internal_node_inserted :  当 n 满足 IsConstantFoldable 中的第二种情况时，即 n 不为叶子节点，为 true
constant_control_deps : key : node value: node 所有的输入节点的依赖树
nodes : 所有满足  IsConstantFoldable 条件的 node


void FindConstantFoldableNodes(
    const Graph* graph, const ConstantFoldingOptions& opts, std::vector<Node*>* nodes,
    std::unordered_map<const Node*, gtl::FlatSet<Node*>>* constant_control_deps,
    std::unordered_map<const Node*, std::vector<Tensor>>* shape_replacement_map)

nodes : 所有满足  IsConstantFoldable 条件的 node
constant_control_deps : key : node value: node 所有的输入节点的依赖树

反向遍历 graph，将所有常量节点加入  nodes, 将每个常量的依赖节点加入 constant_control_deps


Status ConstantFold(const ConstantFoldingOptions& opts, FunctionLibraryRuntime* function_library, Env* env,
                    Device* partition_device, Graph* graph, bool* was_mutated)

1. 反向遍历 graph，将所有常量节点加入  nodes, 将每个常量的依赖节点加入 constant_control_deps, 如果没有找到常量节点，返回
2. 遍历找到的所有常量节点

TODO


### Parallel concat removal

class ParallelConcatRemovePass : public GraphOptimizationPass

Status Run(const GraphOptimizationPassOptions& options)

1. 遍历 options.graph 中的所有节点，找到 op 为 ParallelConcat 的 Node
2. 按照如下图进行修改

```
input1         output1
input2    n    output2
input3_c       output3_c
input4_c       output4_c

变为

input1 ->  update1  \                /  ouput1
input2 ->  iupdte2  -> identity_node -> ouput2
            ^                        \  ouput3_c
            |
input3_c  start
input4_c
```

对于 ParallelConcat 类型的节点 n
1. 创建  start 节点
2. n 的所有 control in_edges 与 start 建立 control edge
3. n 的所有非 control in_edges  创建对应的 update  节点，加入 control_nodes, 每个节点都以  start 和 src+src_ouput 作为输入
4. 创建 identity_node 节点， 输入为 control_nodes，identity_node 与 n 的所有输出 control edge 建立 control edge，非 control edge 建立 edge


创建节点 start
name: n->name/Internal/_${NUM}
op : `_ParallelConcatStart`
attr :
    shape: n->attr("shape")
    dtype: n->attr("T")

创建节点 update
name: n->name/Internal/_${NUM}
op : `_ParallelConcatUpdate`
input : start, src+src_output
attr :
    col: i

创建节点 identity_node
name : n->name
op : Identity
input : control_nodes
device : n->requested_device()
attr :
   `_class`: n->attr("_class")


### ResourceVariableReadPass

```
Replaces ReadVariableOp nodes which are only used by Sends, sinks,
and function Retvals with _UnsafeReadVariable nodes, as this
transformation is safe and will improve performance.
```

class ResourceVariableReadPass : public GraphOptimizationPass

Status Run(const GraphOptimizationPassOptions& options)

1. 找到 ReadVariableOp 的节点加入 matches（过滤掉该节点的输出节点必须满足 IsSend 或 "_Retval" 或 "_SINK" 节点)
2. 用  op_`_UnsafeReadVariable` 替换 ReadVariableOp
