

 执行器几乎是任何一个系统应该具备的组件之一，有的语言标准库就
 实现支持了执行器，从原理上来说大同小异。

## 数据结构

ExecutorImpl -> FrameState  ->  iterations  -> IterationState


TaggedNode -> Node node             ->  gview_.node(node->id()) -> NodeItem
           -> FrameInfo input_frame
           -> int64 input_iter

```
    Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    int64 input_iter = tagged_node.input_iter;
    int id = node->id();
    NodeItem& item = *gview.node(id);
    input_frame->iterations[input_iter]
    Entry* input_tensors = input_frame->iterations[input_iter]->input_tensors;
```



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
    bool sync_on_finish = false;
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
    gtl::FlatSet<string> unique_frame_names; // 对应 graph 所有节点的名称
    std::vector<string> frame_names;         // 索引为节点 id，值为其所属根节点的 "frame_name"
  struct FrameInfo
    int input_count;                      // 为 Enter Node 的个数
    int total_inputs;                     // 所有节点的输入 edge 之和
    PendingCounts::Layout pending_counts_layout;
    PendingCounts* pending_counts;        // 来自于 pending_counts_layout
    std::vector<const Node*>* nodes;      //
  LocalExecutorParams params_;
  const Graph* graph_;
  GraphView gview_;                             //将 graph 转换为该类型对象
  bool device_record_tensor_accesses_ = false;
  std::vector<const Node*> root_nodes_;         //没有输入的节点，形成的初始化的节点队列
  gtl::FlatMap<string, FrameInfo*> frame_info_; //frame 和 FrameInfo 的映射关系

class ExecutorBarrier
  typedef std::function<void(const Status&)> StatusCallback;
  Rendezvous* rendez_;
  StatusCallback done_cb_;  //当所有的 Executor 都执行完之后的回调函数
  mutable mutex mu_;
  int pending_  = 0;        //当前还有多少个 Executor 没有执行完
  Status status_;           // 已经结束的 Executor  状态，只有所有都是 OK，才是 OK


struct NodeItem // 与 Node 相关
  const Node* node;              //所对应的 Node
  OpKernel* kernel ;             // 对应 Node 通过 LocalExecutorParams->create_kernel 创建的 OpKernel
  bool kernel_is_expensive : 1;  // True iff kernel->IsExpensive() 判断依据是什么？
  bool kernel_is_async : 1;      // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;             // True iff IsMerge(node)
  bool is_enter : 1;             // True iff IsEnter(node)
  bool is_exit : 1;              // True iff IsExit(node)
  bool is_control_trigger : 1;   // True iff IsControlTrigger(node)
  bool is_sink : 1;              // True iff IsSink(node)
  bool is_enter_exit_or_next_iter : 1; // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
  int num_inputs;                // 对应node 的输入个数
  int num_outputs;               // 对应 node 的输出个数
  int input_start = 0;           // 某个节点的输入个数
  size_t num_output_edges;       // 对应 node output edge 的数量
  PendingCounts::Handle pending_id;

该类有一个可变长区域

```
  EdgeInfo            out_edges[num_out_edges];  //保存一个  Node 的 out_edge
  AllocatorAttributes output_attr[num_outputs];
  uint8               input_type[num_inputs];
  uint8               output_type[num_outputs];
```

struct EdgeInfo //保存 Node 的基本信息
  int dst_id;           //n->out_edge->dst()->id()
  int output_slot : 31; //n->out_edge->src_output()
  bool is_last : 1;     //n->out_edge->src_output() >=0 就为  true
  int input_slot;       //n->out_edges()->dst_input()

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class GraphView
  int32 num_nodes_ = 0;   // g->num_node_ids();
  uint32* node_offsets_;  // 某个 NodeItem 在  space_ 中的偏移，索引为某个 node->id()，值为 node 对应的 NodeItem 在 space_ 中的索引
  char* space_;           // 保存了 NodeItem 列表； 如获取第 i 个 NodeItem: reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));

class ExecutorState
  DeviceContextMap device_context_map_;
  struct TaggedNode;
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;
  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.
  const bool log_memory_;
  int64 step_id_;
  Rendezvous* rendezvous_;
  SessionState* session_state_;
  TensorStore* tensor_store_;
  ScopedStepContainer* step_container_;
  StepStatsCollector* stats_collector_;
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  FunctionCallFrame* call_frame_;
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;       //运行任务的线程池
  bool sync_on_finish_;                 // 如果 true, 调用 device.Sync()
  bool dumped_on_error_ = false;
  FrameState* root_frame_;              // FrameState(impl_, 1)
  Executor::DoneCallback done_cb_;
  std::atomic_int_fast32_t num_outstanding_ops_;
  mutex mu_;
  Status status_ GUARDED_BY(mu_);
  gtl::FlatMap<string, FrameState*> outstanding_frames_; frame_name:FrameState 映射关系

  struct Entry
    ManualConstructor<Tensor> val;    // 对应的 Tensor
    Tensor* ref;                      //  如果是引用的 Tensor
    mutex* ref_mu;                    //
    bool has_value = false;
    bool val_field_is_set = false;    // val 是否设置
    AllocatorAttributes alloc_attr;
    DeviceContext* device_context;

  struct IterationState
    Entry* input_tensors;          //个数为  FrameState 的 total_input_tensors
    size_t outstanding_ops;        //
    int outstanding_frame_count;   //
    PendingCounts counts_;         // FrameState 的 pending_counts

  struct FrameState // frame_name/parent_frame 唯一标记一个 FrameState
    const ExecutorImpl* executor ; //所属的 ExecutorImpl
    string frame_name;             //
    uint64 frame_id;               //Hash64(frame_name)
    int64 parent_iter = -1;
    FrameState* parent_frame ;
    const int max_parallel_iterations;      // 允许并行运行的 FrameState 的个数，node->attrs("parallel_iterations")
    int num_pending_inputs = 0;             // 处于还没有执行的  input 的数量
    int64 iteration_count  = 0;             // iterations 已经迭代了多少次，每调用一次 IncrementIteration，加 1
    int num_outstanding_iterations = 1;     //  每次调用 IncrementIteration 加 1， 调用 CleanupIterations 减 1
    gtl::InlinedVector<IterationState*, 12> iterations;           // 每个元素 new IterationState(pending_counts, total_input_tensors);
    std::vector<std::pair<const Node*, Entry>> next_iter_roots ;
    std::vector<std::pair<const Node*, Entry>> inv_values;
    std::vector<const Node*> dead_exits;
    PendingCounts* pending_counts;          //
    int total_input_tensors = 0;            // executor->frame_info_ 总共有多少个输入
    std::vector<const Node*>* nodes;
    mutex mu;

  struct TaggedNode
    const Node* node ;              //对应的 Node
    FrameState* input_frame;        //对应的 FrameState
    int64 input_iter = -1;          //在 FrameState 的  iterations 中的索引
    bool is_dead = false;

  class TaggedNodeReadyQueue  //用  vector 替代队列。
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;


## 源码分析

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib, const NodeDef& ndef, int graph_def_version, OpKernel** kernel)

void DeleteNonCachedKernel(OpKernel* kernel)

Status InferAllocAttr(const Node* n, const Node* dst, const DeviceNameUtils::ParsedName& local_dev_name, AllocatorAttributes* attr);


### GraphView

NodeItem* GraphView::node(size_t id) //从 space_ 中获取第 i 个 NodeItem
size_t GraphView::NodeItemBytes(const Node* n) //返回 n 对应的  NodeItem 所需要的内存空间的大小(byte)

char* GraphView::InitializeNode(char* ptr, const Node* n) //将 n 的信息保存在 ptr  所在的位置 NodeItem

void GraphView::Initialize(const Graph* g) // 将 g 转换成 GraphView

void GetMaxPendingCounts(const Node* n, size_t* max_pending, size_t* max_dead_count)

用 n 初始化  max_pending，max_dead_count
max_pending : 如果 IsMerge(n)，返回输入 control edge 乘以 2 + 1；否则 n->in_edges().size();
max_dead_count : n->in_edges().size();

### ExecutorState

string MakeFrameName(FrameState* frame, int64 iter_id, const string& name) //${frame->frame_name}";"${iter_id}";"${name}


void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node, FrameState** child)

1. 构造 node 对应的 frame_name(frame.name + ";" + iter + ";" node.attr("frame_name"))
2. 从 outstanding_frames_ 查找 frame_name 对应的 child(FrameState)，如果找的不断就创建之

void ExecutorState::DeleteFrame(FrameState* frame, TaggedNodeSeq* ready)

TODO 删除 frame, 如果  frame 的父节点的  input 为 0，加入 ready

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter, TaggedNodeSeq* ready)

清除 frame 从 iter 开始的 IterationState. TODO

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec)

从 tagged_node 开始

如果 tagged_node 对应的 Node 是 kernel_is_async 构造 AsyncState 调用 impl_->params_.device->ComputeAsync
否则调用 impl_->params_.device->Compute

Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    DeviceContextVec* input_device_contexts,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead)

从 first_input 开始，初始化 inputs, input_device_contexts, input_alloc_attrs

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             TaggedNodeReadyQueue* inline_ready)

TODO

void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready, TaggedNodeReadyQueue* inline_ready)



如果  inline_ready 不为空, 遍历 ready 中的每个 node
1. 如果 node 不是 kernel_is_expensive 节点，加入 inline_ready 等待执行
2. 如果 node 是 kernel_is_expensive 的节点， 并且 curr_expensive_node  不为空，创建新的线程执行 curr_expensive_node，并这是  curr_expensive_node 为 node

void ExecutorState::Finish()

1. 如果 sync_on_finish_ 为  true, 调用  impl_->params_.device.Sync()
2. 新的线程调用 done_cb

void ExecutorState::RunAsync(Executor::DoneCallback done)

从 impl_->root_nodes_ 中的节点开始遍历，

### Executor

virtual void RunAsync(const Args& args, DoneCallback done) = 0;

Status Run(const Args& args)

::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params, const Graph* graph, Executor** executor);

  ExecutorImpl* impl = new ExecutorImpl(params, graph);
  impl->Initialize();
  executor = impl;

### ExecutorImpl

FrameInfo* EnsureFrameInfo(const string& fname)

从 frame_info_ 获取 fname 对应的  FrameInfo，如果不存在，就创建之

Status ExecutorImpl::Initialize()

1. 用 graph_ 初始化 gview_
2. 调用  BuildControlFlowInfo
3. 遍历 grpah_ 每个 Node，调用 params_->create_kernel，初始化对应的  NodeItem
4.

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done)

  (new ExecutorState(args, this))->RunAsync(std::move(done));

Status ExecutorImpl::BuildControlFlowInfo(const Graph* g, ControlFlowInfo* cf_info)

广度遍历 g 的所有节点，用每个节点的 frame_name，初始化 cf_info。

对于 Enter 节点，cf_info[node_id] 为节点属性 "frame_names", parent = node_id
对于 Exit 节点为与其父节点一样
对于其他节点，为其父节点的节点

问题：貌似，最终所有节点的名称相同与其根节点名称相同

n1 -> n2  -> n4   其中 n2，n3, n4 的 frame_name 都为 n1 的 "frame_names" 属性
   -> n3

void ExecutorImpl::InitializePending(const Graph* graph, const ControlFlowInfo& cf_info)

TODO

Status GraphView::SetAllocAttrs(const Graph* g, const Device* device)

TODO

Status InferAllocAttr(const Node* n, const Node* dst, const DeviceNameUtils::ParsedName& local_dev_name, AllocatorAttributes* attr)

检查  n, dst 是否在同一地址空间

### ExecutorBarrier

StatusCallback ExecutorBarrier::Get() //return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1)

void ExecutorBarrier::WhenDone(const Status& s)

pending_ 减一，当  pending_ 为 0 时，调用 done_cb_

### IterationState

int pending(PendingCounts::Handle h)                  // return counts_.pending(h);
int decrement_pending(PendingCounts::Handle h, int v) // return counts_.decrement_pending(h, v);
void mark_live(PendingCounts::Handle h)               // counts_.mark_live(h);
void mark_started(PendingCounts::Handle h)            // counts_.mark_started(h);
void mark_completed(PendingCounts::Handle h)          // counts_.mark_completed(h);
PendingCounts::NodeState node_state(PendingCounts::Handle h)  //counts_.node_state(h);
int dead_count(PendingCounts::Handle h)               // return counts_.dead_count(h);
void increment_dead_count(PendingCounts::Handle h)    // counts_.increment_dead_count(h);
void adjust_for_activation(PendingCounts::Handle h,
        bool increment_dead, int* pending_result,
        int* dead_result) // counts_.adjust_for_activation(h, increment_dead, pending_result, dead_result);



### FrameState

IterationState* GetIteration(int64 iter) // iterations[iter % iterations.size()]
void SetIteration(int64 iter, IterationState* state)  //iterations[iter % iterations.size()] = state
bool DecrementOutstandingOps(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)
bool DecrementOutstandingOpsLocked(const GraphView* gview, int64 iter, TaggedNodeSeq* ready) //

1. iterations[iter]->outstanding_ops--，
2. 如果 iterations[iter]->outstanding_ops 为 0 返回  true, 否则调用 CleanupIterations(gview, iter, ready);

bool IsFrameDone() //num_pending_inputs == 0 && num_outstanding_iterations == 0

bool IsIterationDone(int64 iter) //

void IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)

void ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)

void ActivateLoopInvs(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)

void AddLoopInv(const NodeItem* item, const Entry& value, TaggedNodeSeq* ready)

void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter, EntryVector* outputs, TaggedNodeSeq* ready)

bool IsIterationDone(int64 iter)

需要满足如下任意条件
1. iterations[iter].outstanding_ops == 0 && iterations[iter].outstanding_frame_count == b && iter == 0
2. iterations[iter].outstanding_ops == 0 && iterations[iter].outstanding_frame_count == b &&  iterations[iter -1 ] == nullptr

void ExecutorState::FrameState::IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)

1. iterations 迭代到下一个元素
2. 遍历 next_iter_roots  每一个元素，调用 ActivateNodes，清空 next_iter_roots
3. 遍历 inv_values 每一个元素，调用 ActivateNodes，清空 next_iter_roots

void ExecutorState::FrameState::ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)

遍历 next_iter_roots，调用 ActivateNodes，清空  next_iter_roots

void ExecutorState::FrameState::ActivateLoopInvs(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)

遍历 inv_values，调用 ActivateNodes，清空 next_iter_roots

void ExecutorState::FrameState::AddLoopInv(const NodeItem* item, const Entry& entry, TaggedNodeSeq* ready)

```
  inv_values.push_back({item->node, entry});
  bool is_dead = !entry.has_value;
  for (int i = 0; i <= iteration_count; ++i) {
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, i, &outputs, ready);
  }
```

void ExecutorState::FrameState::ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                                              EntryVector* outputs, TaggedNodeSeq* ready)

TODO

### NodeItem


char* var()                               // this + sizeof(NodeItem)
EdgeInfo* output_edge_base()              // this + sizeof(NodeItem))
AllocatorAttributes* output_attr_base()   // this + sizeof(NodeItem) + sizeof(EdgeInfo) * num_output_edges
const EdgeInfo* output_edge_list()        // EdgeInfo(this + sizeof(NodeItem))
uint8* input_type_base()                  // this + sizeof(NodeItem) + sizeof(EdgeInfo) * num_output_edges + sizeof(AllocatorAttributes) * num_outputs
uint8* output_type_base()                 // this + sizeof(NodeItem) + sizeof(EdgeInfo) * num_output_edges + sizeof(AllocatorAttributes) * num_outputs + sizeof(uint8) * num_inputs
EdgeInfo& output_edge(int i)              // output_edge_base()[i]
DataType input_type(int i)                // input_type_base()[i]
DataType output_type(int i)               // output_type_base()[i]
AllocatorAttributes* output_attrs()       // output_attr_base();
