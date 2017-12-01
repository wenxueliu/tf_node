

 执行器几乎是任何一个系统应该具备的组件之一，有的语言标准库就
 实现支持了执行器，从原理上来说大同小异。

## 数据结构

static const Tensor* const kEmptyTensor = new Tensor;

class Executor
  struct Args
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    StepStatsCollector* stats_collector = nullptr;
    FunctionCallFrame* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    bool sync_on_finish = false; // If true, calls Sync() on the device.
    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;
    // A callback that is invoked each time a node has finished executing.
    typedef std::function<Status(const string& node_name, const int output_slot, const Tensor* tensor, const bool is_ref, OpKernelContext* ctx)> NodeOutputsCallback;
    NodeOutputsCallback node_outputs_cb = nullptr;
  typedef std::function<void(const Status&)> DoneCallback;

struct LocalExecutorParams
  Device* device;
  FunctionLibraryRuntime* function_library = nullptr;
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  std::function<void(OpKernel*)> delete_kernel;
  Executor::Args::NodeOutputsCallback node_outputs_cb;

class ExecutorImpl : public Executor
  struct ControlFlowInfo
    gtl::FlatSet<string> unique_frame_names;
    std::vector<string> frame_names;
  struct FrameInfo
    int input_count;
    int total_inputs;
    PendingCounts::Layout pending_counts_layout;
    PendingCounts* pending_counts;  // Owned
    std::vector<const Node*>* nodes;  // Owned
  LocalExecutorParams params_;
  const Graph* graph_;
  GraphView gview_; //将  graph 转换为该类型对象
  bool device_record_tensor_accesses_ = false;
  std::vector<const Node*> root_nodes_;
  gtl::FlatMap<string, FrameInfo*> frame_info_;

class ExecutorBarrier {
  typedef std::function<void(const Status&)> StatusCallback;
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr; //当所有的 Executor 都执行完之后的回调函数
  mutable mutex mu_;
  int pending_  = 0;  //当前还有多少个 Executor 没有执行完
  Status status_; // 已经结束的 Executor  状态，只有所有都是 OK，才是 OK

struct NodeItem // 与 Node 相关
  const Node* node = nullptr;
  OpKernel* kernel = nullptr;
  bool kernel_is_expensive : 1;  // True iff kernel->IsExpensive()
  bool kernel_is_async : 1;      // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;             // True iff IsMerge(node)
  bool is_enter : 1;             // True iff IsEnter(node)
  bool is_exit : 1;              // True iff IsExit(node)
  bool is_control_trigger : 1;   // True iff IsControlTrigger(node)
  bool is_sink : 1;              // True iff IsSink(node)
  bool is_enter_exit_or_next_iter : 1; // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // Number of output edges.
  size_t num_output_edges;

  PendingCounts::Handle pending_id;

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class GraphView
  int32 num_nodes_ = 0; // g->num_node_ids();
  uint32* node_offsets_ = nullptr; 索引为某个 node->id()，值为 node 对应的 NodeItem 在 space_ 中的索引
  char* space_;  // 保存了 NodeItem 列表  如 获取第 i 个 NodeItem: reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));



## 源码分析

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib, const NodeDef& ndef, int graph_def_version, OpKernel** kernel)

void DeleteNonCachedKernel(OpKernel* kernel)

Status InferAllocAttr(const Node* n, const Node* dst, const DeviceNameUtils::ParsedName& local_dev_name, AllocatorAttributes* attr);


### GraphView

NodeItem* GraphView::node(size_t id) //从 space_ 中获取第 i 个 NodeItem
size_t GraphView::NodeItemBytes(const Node* n) //返回 n  对应的  NodeItem 所需要的长度(byte)

char* GraphView::InitializeNode(char* ptr, const Node* n) //在  ptr 分配 NodeItem 保存 n 的相关信息.

void GraphView::Initialize(const Graph* g) // 将 g 转换成 GraphView

void GetMaxPendingCounts(const Node* n, size_t* max_pending, size_t* max_dead_count)

用 n 初始化  max_pending，max_dead_count
max_pending : 如果 IsMerge(n)，返回输入 control edge 乘以 2 + 1；否则 n->in_edges().size();
max_dead_count : n->in_edges().size();


### Executor

virtual void RunAsync(const Args& args, DoneCallback done) = 0;

Status Run(const Args& args)

::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params, const Graph* graph, Executor** executor);

### ExecutorImpl

Status ExecutorImpl::Initialize()

Status ExecutorImpl::BuildControlFlowInfo(const Graph* g, ControlFlowInfo* cf_info)

广度遍历 g 的所有节点，用每个节点的 frame_name，初始化 cf_info。如果遇到 Exit
节点，返回到父节点

### ExecutorBarrier

StatusCallback ExecutorBarrier::Get() //return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1)

void ExecutorBarrier::WhenDone(const Status& s)
