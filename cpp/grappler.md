
struct GrapplerItem
  string id;  // A unique id for this item
  GraphDef graph;
  std::vector<std::pair<string, Tensor>> feed;
  std::vector<string> fetch;
  std::vector<string> init_ops;
  int64 expected_init_time = 0;
  std::vector<QueueRunnerDef> queue_runners;



## 源码分析

### GrapplerItem

GrapplerItem::GrapplerItem(const GrapplerItem& other, GraphDef&& graphDef)

用  other, graphDef 初始化

std::vector<const NodeDef*> GrapplerItem::MainOpsFanin()

从 fetch 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组

std::vector<const NodeDef*> GrapplerItem::EnqueueOpsFanin()

从 queue_runners 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组

std::vector<const NodeDef*> GrapplerItem::InitOpsFanin()

从 init_ops 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组

std::vector<const NodeDef*> GrapplerItem::MainVariables()

从 init_ops 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点，如果是变量节点，加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组

std::vector<const NodeDef*> ComputeTransitiveFanin(const GraphDef& graph, const std::vector<string>& terminal_nodes)

从 terminal_nodes 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组

std::vector<const NodeDef*> ComputeTransitiveFanin(const GraphDef& graph, const std::vector<string>& terminal_nodes, bool* ill_formed)

从 terminal_nodes 中的元素( 必须存在于 graph)反向广度遍历，将所有遍历的节点加入数组并返回。
注：如果 terminal_nodes 中有不存在与 graph 的节点将返回空数组
