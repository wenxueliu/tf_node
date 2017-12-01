

## 数据结构

struct SimpleGraphExecutionStateOptions
  const DeviceSet* device_set = nullptr;
  const SessionOptions* session_options = nullptr;
  // A map from node name to device name, representing the unchangeable
  // placement of stateful nodes.
  std::unordered_map<string, string> stateful_placements;

struct SimpleClientGraph
  std::unique_ptr<FunctionLibraryDefinition> flib_def;
  Graph graph;
  DataTypeVector feed_types;
  DataTypeVector fetch_types;

class SimpleGraphExecutionState
  std::unordered_map<string, string> stateful_placements_;
  GraphDef original_graph_def_;            // Immutable after ctor.
  const DeviceSet* device_set_;            // Not owned
  const SessionOptions* session_options_;  // Not owned
  mutable mutex mu_;
  CostModel costs_ GUARDED_BY(mu_);
  NodeNameToCostIdMap node_name_to_cost_id_map_; //n->name(): n->costs_id() 映射关系
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<subgraph::RewriteGraphMetadata> rewrite_metadata_;
  Graph* graph_;


## 源码分析

Status SimpleGraphExecutionState::MakeForBaseGraph(
    GraphDef* graph_def, const SimpleGraphExecutionStateOptions& options,
    std::unique_ptr<SimpleGraphExecutionState>* out_state)

根据 graph_def,options 创建 SimpleGraphExecutionState，并设置默认值。 创建
的对象保持在 out_state

Status SimpleGraphExecutionState::MakeForPrunedGraph(
    const FunctionDefLibrary& func_def_lib,
    const SimpleGraphExecutionStateOptions& options, const GraphDef& graph_def,
    const BuildGraphOptions& subgraph_options,
    std::unique_ptr<SimpleGraphExecutionState>* out_state,
    std::unique_ptr<SimpleClientGraph>* out_client_graph)

1. 根据 options, graph_def  创建 SimpleGraphExecutionState
2. 设置默认值
TODO

Status SimpleGraphExecutionState::InitBaseGraph(const BuildGraphOptions& options)

1. 遍历所有 graph 所有节点，初始化  node_name_to_cost_id_map_
2. 根据选项重构当前 graph 的子图
3. 用 stateful_placements_  重置 graph 中的 node
4. 初始化  costmodel
5. 调用 PRE_PLACEMENT 阶段的优化
6. SimplePlacer
5. 调用 POST_PLACEMENT 阶段的优化
7. 遍历所有 graph 所有节点，将 n->name():n->assigned_device_name() 保持在 stateful_placements_
TODO


void SimpleGraphExecutionState::RestoreStatefulNodes(Graph* graph)

对于 graph 中的所有 stateful  为  true 的节点， 如果存在于 stateful_placements_  就设置其 assigned_device_name
