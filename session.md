
## Session

依赖 thread_pool, device

所有设备中的第一个设备非常关键， 作为  client device(CPU device)，用于喂和提取 tensor

### 术语

log memory
partial_run


### 环境变量

TF_SYNC_ON_FINISH

### 源文件

tensorflow/core/public/session_options.h
tensorflow/core/common_runtime/session_factory.h

### 数据结构

DirectSessionFactory -> DirectSession
采用工厂模式

所有的  session 都保存在全局变量 factories, 通过 Register 注册一个
SessionFactory，

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
static SessionFactories* factories = new SessionFactories;

typedef std::pair<int32, thread::ThreadPool*> MapValue;
static std::map<string, MapValue>* global_pool_map = new std::map<string, MapValue>; //key : 线程名 value: 线程数：线程池对象

static DirectSessionRegistrar registrar; //向全局变量 factories 增加 DIRECT_SESSION:new DirectSessionFactory() 对象

class SessionFactory
class DirectSessionFactory : public SessionFactory
class GrpcSessionFactory : public SessionFactory

class SessionFactory
  Session* NewSession(const SessionOptions& options) = 0;
  bool AcceptsOptions(const SessionOptions& options) = 0;
  Status Reset(const SessionOptions& options, const std::vector<string>& containers);
  Status GetFactory(const SessionOptions& options, SessionFactory** out_factory);

class DirectSessionFactory //本地  device 上执行
  mutex sessions_lock_;
  std::vector<DirectSession*> sessions_ ; //所有的 DirectSession

class Session
class DirectSession : public Session

struct SessionOptions
  Env* env;
  string target; 默认是 local,  ip:port, host:port 等根据任务不同而不同
  ConfigProto config;

message ThreadPoolOptionProto
  int32 num_threads = 1;   //如果为 0, 默认 CPU 的个数
  string global_name = 2;  //如果为空 Compute:$NUM

message ConfigProto
  map<string, int32> device_count = 1;
  int32 intra_op_parallelism_threads = 2;
  int32 inter_op_parallelism_threads = 5; //默认为 CPU 的数量
  bool use_per_session_threads = 9;
  repeated ThreadPoolOptionProto session_inter_op_thread_pool = 12;
  int32 placement_period = 3;
  repeated string device_filters = 4;
  GPUOptions gpu_options = 6;
  bool allow_soft_placement = 7;
  bool log_device_placement = 8;
  GraphOptions graph_options = 10;
  int64 operation_timeout_in_ms = 11;
  RPCOptions rpc_options = 13;
  ClusterDef cluster_def = 14;

message GraphOptions
  reserved "skip_common_subexpression_elimination";
  reserved 1;
  bool enable_recv_scheduling = 2;
  OptimizerOptions optimizer_options = 3;
  int64 build_cost_model = 4;
  int64 build_cost_model_after = 9;
  bool infer_shapes = 5;
  bool place_pruned_graph = 6;
  bool enable_bfloat16_sendrecv = 7;
  int32 timeline_step = 8;
  RewriterConfig rewrite_options = 10;

  struct RunState
    mutex mu_;
    Status status GUARDED_BY(mu_);
    IntraProcessRendezvous* rendez = nullptr;
    std::unique_ptr<StepStatsCollector> collector;
    Notification executors_done;
    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched
    TensorStore tensor_store;
    ScopedStepContainer step_container;

  struct ExecutorsAndKeys
    std::atomic_int_fast64_t step_count;
    std::unique_ptr<Graph> graph;
    NameNodeMap name_to_node; //保存 graph->nodes() 中 name 与 Node 的映射关系
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::vector<PerPartitionExecutorsAndLib> items;
    /*
     * 保存 inputs 和 outputs 中 元素与索引的键值，其中 !partial_run 使用不带
     * rendezvous 的接口
     */
    std::unordered_map<string, size_t> input_name_to_index;
    std::unordered_map<string, string> input_name_to_rendezvous_key;
    std::unordered_map<string, size_t> output_name_to_index;
    std::unordered_map<string, string> output_name_to_rendezvous_key;
    DataTypeVector input_types;
    DataTypeVector output_types;

  struct RunStateArgs {
    bool is_partial_run = false;
    string handle;
    std::unique_ptr<Graph> graph;
    const DebugOptions& debug_options;

class DirectSession
    const SessionOptions options_;
    const std::unique_ptr<const DeviceMgr> device_mgr_;
    std::vector<Device*> devices_;  //所有设备
    DeviceSet device_set_; //所有设备
    string session_handle_;
    bool graph_created_ GUARDED_BY(graph_def_lock_) = false;
    mutex graph_def_lock_;
    GraphDef graph_def_ //
    std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_; //所有的线程池，优先级 options_.config.session_inter_op_thread_pool_size() > options_.config.use_per_session_threads()
    Status init_error_;  // Set to an error if construction failed.
    bool sync_on_finish_ = true; //如果为 true, 阻塞直到所有设备完成队列中的操作
    mutex executor_lock_;  // protects executors_
    std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
    std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
    SessionState session_state_; 目前所有活跃的 tensor
    DirectSessionFactory* const factory_;  // not owned
    CancellationManager* cancellation_manager_;
    std::unordered_map<string, string> stateful_placements_
    std::unique_ptr<SimpleGraphExecutionState> execution_state_
    std::unique_ptr<FunctionLibraryDefinition> flib_def_;
    mutex closed_lock_;
    bool closed_ = false;
    std::atomic<int64> edge_name_counter_ = {0};
    std::atomic<int64> handle_name_counter_ = {0};
    // For generating step ids that are unique across all sessions.
    static std::atomic_int_fast64_t step_id_counter_;
    const int64 operation_timeout_in_ms_ = 0;
    CostModelManager cost_model_manager_;
    Executor::Args::NodeOutputsCallback node_outputs_callback_ = nullptr;


Status SessionFactory::GetFactory(const SessionOptions& options, SessionFactory** out_factory)

从 factories 中找到  options 对应的 SessionFactory 保存在 out_factory。
注：如果存在多个或不存在，会报错


Session* Session::NewSession(const SessionOptions& options) {

1. 根据 options 从 factories 找到对应的  SessionFactory factory
2. factory.NewSession(options) 创建新的 Session

Status Session::Reset(const SessionOptions& options, const std::vector<string>& containers)

1. 根据 options 从 factories 找到对应的  SessionFactory factory
2. factory.Reset(options, containers) 创建新的 Session

Status NewThreadPoolFromThreadPoolOptions(const SessionOptions& options,
    const ThreadPoolOptionProto& thread_pool_options, int pool_number, thread::ThreadPool** pool, bool* owned)

根据 options, thread_pool_options, pool_number 创建线程池，
1. 如果线程没有配置名字，创建创建线程池对象，初始化  own 为 true
2. 如果线程配置了名字
2.1 如果 global_pool_map 中存在对应的线程池对象，初始化 pool, own(false)
2.2 如果 global_pool_map 不存在对应的线程池对象，初始化 global_pool_map，pool, own(false)

thread::ThreadPool* GlobalThreadPool(const SessionOptions& options)

更加  options 创建线程池对象，并返回


bool DirectSessionFactory::AcceptsOptions(const SessionOptions& options) //return options.target.empty();

Session* DirectSessionFactory::NewSession(const SessionOptions& options)

1. 创建一个 device
2.  根据  options，device 创建一个 session, 保存在 sessions_ 中

Status DirectSessionFactory::Reset(const SessionOptions& options, const std::vector<string>& containers)

用 containers 重置所有 sessions_

void DirectSessionFactory::Deregister(const DirectSession* session)

从  sessions_ 中删除 session

DirectSession::DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr, DirectSessionFactory* const factory)

1. 创建线程池
    thread_pools_.emplace_back(pool, owned);
    thread_pools_.emplace_back(NewThreadPoolFromSessionOptions(options_), true);
    thread_pools_.emplace_back(GlobalThreadPool(options), false);
2. 初始化设备

   for (auto d : device_mgr_->ListDevices())
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;

Session* DirectSession::NewSession(const SessionOptions& options) override {

Status DirectSession::MaybeInitializeExecutionState(const GraphDef& graph, bool* out_already_initialized)

  flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
  GraphDef temp(graph);
  SimpleGraphExecutionState::MakeForBaseGraph(&temp, options, &execution_state_)

void DirectSession::SchedClosure(thread::ThreadPool* pool, std::function<void()> c) //pool->Schedule(std::move(c));

Status DirectSession::Create(const GraphDef& graph)

Status DirectSession::Extend(const GraphDef& graph)

Status DirectSession::ExtendLocked(const GraphDef& graph) {

TODO
  MaybeInitializeExecutionState(graph, &already_initialized)
  if (already_initialized)
    flib_def_->AddLibrary(graph.library())
    std::unique_ptr<SimpleGraphExecutionState> state;
    execution_state_->Extend(graph, &state)
    execution_state_.swap(state);


Run(const NamedTensorList& inputs,
    const std::vector<string>& output_names,
    const std::vector<string>& target_nodes,
    std::vector<Tensor>* outputs) {

  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);



Status DirectSession::CreateDebuggerState(
    const DebugOptions& debug_options, int64 session_run_index,
    int64 executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_names,
    std::unique_ptr<DebuggerStateInterface>* debugger_state)

    DebuggerStateRegistry::CreateState(debug_options, debugger_state)
    debugger_state->get()->PublishDebugMetadata(debug_options.global_step(), session_run_index, executor_step_index, input_names, output_names, target_names)

Status DirectSession::DecorateAndPublishGraphForDebug(const DebugOptions& debug_options, Graph* graph, Device* device)

    DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator)
    decorator->DecorateGraph(graph, device)
    decorator->PublishGraph(*graph, device->name())


Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata)

1. direct_session_runs->GetCell()->IncrementBy(1);
2. GetOrCreateExecutors(pool, input_tensor_names, output_names, target_nodes, &executors_and_keys, &run_state_args));
3. TODO

Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle)

1.  GetOrCreateExecutors(pool, input_names, output_names, target_nodes, &executors_and_keys, &run_state_args);
2.  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state->rendez, [run_state](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(run_state->mu_);
          run_state->status.Update(ret);
        }
        run_state->executors_done.Notify();
      });
3. 初始化 Executor::Args
4.
  for (auto& item : executors_and_keys->items)
    item.executor->RunAsync(args, barrier->Get());
    handle = run_state_args.handle;

PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {

1. key = str_util::Split(handle, ';')
2. executors_and_keys = executors_.find(key)->second.get()
3. run_state = partial_runs_.find(handle)->second.get();
4. 检验 CheckFetch(inputs, output_names, executors_and_keys, run_state));
5. 发送输入 SendPRunInputs(inputs, executors_and_keys, run_state->rendez);
6. 接受输出 RecvPRunOutputs(output_names, executors_and_keys, run_state, outputs);
7. 保存结果 run_state->tensor_store.SaveTensors(output_names, &session_state_);
8. 见下面

    for (const auto& input : inputs)
      auto it = run_state->pending_inputs.find(input.first);
      it->second = true;
    for (const auto& name : output_names)
      auto it = run_state->pending_outputs.find(name);
      it->second = true;

    done = run_state->PendingDone();
    WaitForNotification(run_state, cancellation_manager_, operation_timeout_in_ms_);
    partial_runs_.erase(handle);


Status ResourceHandleToInputTensor(const Tensor& resource_tensor,
        Tensor* retrieved_tensor)
  const ResourceHandle& resource_handle = resource_tensor.scalar<ResourceHandle>()();
  return session_state_.GetTensor(resource_handle.name(), retrieved_tensor);


SendPRunInputs(const NamedTensorList& inputs,
            const ExecutorsAndKeys* executors_and_keys,
            IntraProcessRendezvous* rendez) {

1. 遍历 inputs 找到 executors_and_keys->output_name_to_rendezvous_key
   对应的 input_key
2. 调用 Rendezvous::ParseKey(input_key, parsed) 初始化 parsed
3. if (input.second.dtype() == DT_RESOURCE) rendez->Send(parsed, Rendezvous::Args(), tensor_from_handle, false);
   else s = rendez->Send(parsed, Rendezvous::Args(), input.second, false);
3. rendez->Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead, operation_timeout_in_ms_);
4. (*outputs)[output_offset] = output_tensor;

DirectSession::RecvPRunOutputs(
    const std::vector<string>& output_names,
    const ExecutorsAndKeys* executors_and_keys, RunState* run_state,
    std::vector<Tensor>* outputs)

1. 遍历 output_names 找到 executors_and_keys->output_name_to_rendezvous_key
   对应的 output_key
2. 调用 Rendezvous::ParseKey(output_key, parsed) 初始化 parsed
3. rendez->Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead, operation_timeout_in_ms_);
4. (*outputs)[output_offset] = output_tensor;

CheckFetch(const NamedTensorList& feeds,
        const std::vector<string>& fetches,
        const ExecutorsAndKeys* executors_and_keys,
        const RunState* run_state)

1. 将 run_state->pending_inputs 与 executors_and_keys->name_to_node 的交集减去 feeds 保存在 pending_feeds
2. 确保 fetches 中每个节点的 in_edge 不在 pending_feeds 中

GetOrCreateExecutors(
      thread::ThreadPool* pool, gtl::ArraySlice<string> inputs,
      gtl::ArraySlice<string> outputs, gtl::ArraySlice<string> target_nodes,
      ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args);

1. 构造 key，. 从 executors_ 中查找 key 对应的 ExecutorsAndKeys, 如果找到返回，没有找到继续步骤 2
2. 构造 sorted_key，从 executors_ 中查找 sorted_key 对应的 ExecutorsAndKeys, 如果找到返回，没有找到继续步骤 3
3. 调用 CreateGraphs 创建 graphs
4. 分别调用 NewLocalExecutor, 将其加入 executors_

其中 executors_and_keys 保存新加入的 ExecutorsAndKeys

CreateGraphs( const BuildGraphOptions& options,
      std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      RunStateArgs* run_state_args, DataTypeVector* input_types,
      DataTypeVector* output_types);

    TODO

ListDevices(std::vector<DeviceAttributes>* response)
    将 devices_ 中的所有设备的属性保存在 response
Reset()
  device_mgr_->ClearContainers(containers);

Close()
  cancellation_manager_->StartCancel();
  closed_ = true;
  if (factory_ != nullptr) factory_->Deregister(this);



### 配置选项




### python

InteractiveSession 主要用于 ipython 调试使用

关键变量

graph
config
session
opened
closed
add_shapes


关键方法

graph(self)
graph_def(self)
sess_str(self)

as_default(self)

    ```python
    c = tf.constant(..)
    sess = tf.Session()

    with sess.as_default():
      assert tf.get_default_session() is sess
      print(c.eval())
    ```

    *N.B.* The `as_default` context manager *does not* close the
    session when you exit the context, and you must close the session
    explicitly.

    Alternatively, you can use `with tf.Session():` to create a
    session that is automatically closed on exiting the context,
    including when an uncaught exception is raised.

    *N.B.* The default session is a property of the current thread. If you
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

    *N.B.* Entering a `with sess.as_default():` block does not affect
    the current default graph. If you are using multiple graphs, and
    `sess.graph` is different from the value of @{tf.get_default_graph},
    you must explicitly enter a `with sess.graph.as_default():` block
    to make `sess.graph` the default graph.

    ```python
    c = tf.constant(...)
    sess = tf.Session()
    with sess.as_default():
      print(c.eval())
    # ...
    with sess.as_default():
      print(c.eval())

    sess.close()
    ```
run(self, fetches, feed_dict=None, options=None, run_metadata=None):

由具体的 C 实现


    The `fetches` argument may be a single graph element, or an arbitrarily
    nested list, tuple, namedtuple, dict, or OrderedDict containing graph
    elements at its leaves.  A graph element can be one of the following types:

    * An @{tf.Operation}.
      The corresponding fetched value will be `None`.
    * A @{tf.Tensor}.
      The corresponding fetched value will be a numpy ndarray containing the
      value of that tensor.
    * A @{tf.SparseTensor}.
      The corresponding fetched value will be a
      @{tf.SparseTensorValue}
      containing the value of that sparse tensor.
    * A `get_tensor_handle` op.  The corresponding fetched value will be a
      numpy ndarray containing the handle of that tensor.
    * A `string` which is the name of a tensor or operation in the graph.

    The value returned by `run()` has the same shape as the `fetches` argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    ```python
       a = tf.constant([10, 20])
       b = tf.constant([1.0, 2.0])
       # 'fetches' can be a singleton
       v = session.run(a)
       # v is the numpy array [10, 20]
       # 'fetches' can be a list.
       v = session.run([a, b])
       # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
       # 1-D array [1.0, 2.0]
       # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
       MyData = collections.namedtuple('MyData', ['a', 'b'])
       v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
       # v is a dict with
       # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
       # 'b' (the numpy array [1.0, 2.0])
       # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
       # [10, 20].
    ```

partial_run(self, handle, fetches, feed_dict=None)


    ```python
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    res = sess.partial_run(h, r2, feed_dict={c: res})
    ```

partial_run_setup(self, fetches, feeds=None)

TODO

make_callable(self, fetches, feed_list=None, accept_options=False):

TODO

\_do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata):

对 target_list, fetch_list, feed_list 做转换，不同的 handle 是否为 None,
调用不同的函数

\_register_dead_handle(self, handle):

当 self._dead_handles 的数量超过 10 个, 经过处理后，加入 feed 和 fetches, 之后调用 self.run

所做的处理:

从 graph 中的 handle 对应的设备删除 deleter_key 对应的 handler, 将被删除的
handler 加入 graph._handle_deleters

\_update_with_movers(self, feed_dict, feed_map):

TODO



