
class SlackAnalysis
  const Graph* graph_;
  const CostModel* cost_model_;


class GreedyScheduler
  struct Sim
    int degree_parallelism;
    int num_running;
    std::vector<Node*> ready_nodes;

  struct Event
    Node* node;
    Microseconds time;
    bool is_completion;

  const DeviceSet* devices_;
  const CostModel* cost_model_;
  const Graph* graph_;
  std::vector<int64>* priority_;
  std::unordered_map<string, Sim*> device_states_;

class PriorityScheduler
  const DeviceSet* devices_;
  const CostModel* cost_model_;
  const Graph* graph_;


## 源码分析

void InitializePending(const Graph* graph, std::vector<int>* pending)

遍历 graph 的每个 node:
如果 IsMerge(node) 只记录 control edge 数量，以 control edge 的两倍保存在 pending。其中 pending 索引为 node->id()。
否则 node.in_edges().size() 保存在 pending 中。其中  pending 索引为 node->id()。

bool UpdatePending(const Edge* edge, std::vector<int>* pending_count)

如果 node 是  merge 节点，那么当

TODO

### SlackAnalysis

SlackAnalysis::SlackAnalysis(const Graph* g, const CostModel* cost_model): graph_(g), cost_model_(cost_model) {}

Microseconds ComputeAsap(std::vector<Microseconds>* asap_times)

正向广度遍历，计算每个节点开始时间，如果相邻两个节点之间在不同的设备，增加 10 ms
的延迟(该模型需要修正）。 每个节点的时间由 cost_model_->TimeEstimate(curr)  设置

Microseconds ComputeAlap(std::vector<Microseconds>* alap_times)

正向广度遍历，计算每个节点开始时间，如果相邻两个节点之间在不同的设备，增加 10 ms
的延迟(该模型需要修正）。 每个节点的时间由 cost_model_->TimeEstimate(curr)  设置

void ComputeSlack(std::vector<int64>* slacks)

 每个节点的的  slack[node->id()] = alap_times[node->id()] - asap_times[node->id() - makespan;

### GreedyScheduler

GreedyScheduler::GreedyScheduler(const DeviceSet* devices, const CostModel* cost_model, const Graph* g, std::vector<int64>* priority)

devices_(devices),
cost_model_(cost_model),
graph_(g),
priority_(priority)

遍历 devices_->devices() 设置 device_states_

Microseconds GreedyScheduler::ComputeSchedule(std::vector<Microseconds>* start_times)

正向广度遍历 graph_ 的所有节点，计算每个节点的 start_times，每个节点最多有两个节点运行。TODO 大体意思明白了，但细节需要进一步理解。

Node* GetNodeWithHighestPriority(const std::vector<Node*>& nodes)

遍历  nodes，找到优先级最小的节点， 返回该节点。

### PriorityScheduler

PriorityScheduler::PriorityScheduler(const DeviceSet* devices, const CostModel* cost_model, const Graph* g)

devices_(devices), cost_model_(cost_model), graph_(g)

Microseconds ComputeSchedule(std::vector<Microseconds>* start_times);

遍历 graph_, cost_model_ 计算每个节点的开始时间。

Microseconds AssignPriorities(std::vector<int64>* priorities);

将每个节点优先级赋值给 priorities，返回最大完成时间。
