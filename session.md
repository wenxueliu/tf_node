
依赖 thread_pool, device

所有设备中的第一个设备非常关键， 作为  client device(CPU device)，用于喂和提取 tensor

遵循 Register-Factorory 模式，先注册，后通过工厂类创建

所有的  session 都保存在全局变量 factories, 通过 Register 注册一个
SessionFactory. 目前已经注册的包含

```
    DIRECT_SESSION : new DirectSessionFactory()
    GRPC_SESSION : new GrpcSessionFactory()
```


### 术语

log memory
partial_run

### 环境变量

TF_SYNC_ON_FINISH

### 源文件

tensorflow/core/public/session_options.h
tensorflow/core/public/session.h
tensorflow/core/framework/session_state.h
tensorflow/core/common_runtime/session_state.cc
tensorflow/core/common_runtime/session_factory.h
tensorflow/core/common_runtime/session_factory.cc
tensorflow/core/common_runtime/session.cc
tensorflow/core/common_runtime/session_options.cc
tensorflow/core/common_runtime/direct_session.h
tensorflow/core/common_runtime/direct_session.cc
tensorflow/core/distributed_runtime/rpc/grpc_session.h
tensorflow/core/distributed_runtime/rpc/grpc_session.cc

### 数据结构

设置 SessionOptions 的 target 控制具体想用哪种 Session， Session 直接通过 NewSession 创建，目前实现为
1. target = "grpc://" 时，创建 GrpcSession
2. target = "" 时，创建 DirectSession

DirectSessionFactory -> DirectSession

GrpcSessionRegistrar -> GrpcSession -> GrpcRemoteMaster -> MasterService::Stub -> ::grpc::internal::BlockingUnaryCall()

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
static SessionFactories* factories = new SessionFactories;

typedef std::pair<int32, thread::ThreadPool*> MapValue;
static std::map<string, MapValue>* global_pool_map = new std::map<string, MapValue>; //key : 线程名 value: 线程数：线程池对象

class SessionFactory //创建或重置 Session
  Session* NewSession(const SessionOptions& options) = 0;
  bool AcceptsOptions(const SessionOptions& options) = 0;
  Status Reset(const SessionOptions& options, const std::vector<string>& containers);
  Status GetFactory(const SessionOptions& options, SessionFactory** out_factory);

class Session

### WorkerSession

struct WorkerSession //封装和和 session 相关的状态
  string worker_name; //"/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}
  std::unique_ptr<WorkerCacheInterface> worker_cache; //new WorkerFreeListCache(worker_cache)
  std::unique_ptr<DeviceMgr> device_mgr; //worker_env->device_mgr
  std::unique_ptr<GraphMgr> graph_mgr;  //new GraphMgr(worker_env, worker_env->device_mgr)

class SessionMgr //对 WorkerSession 的增删查
  typedef std::function<Status(const ServerDef&, WorkerCacheInterface**)> WorkerCacheFactory;
  const WorkerEnv* const worker_env_;  //WorkerCacheFactory(worker_cache_factory_options(server_def_), worker_cache)
  WorkerSession legacy_session_;
  const WorkerCacheFactory worker_cache_factory_; //WorkerCacheFactory(worker_cache_factory_options(server_def_), worker_cache)
  mutex mu_;
  std::map<string, std::unique_ptr<WorkerSession>> sessions_ GUARDED_BY(mu_);

### Direction Session

static DirectSessionRegistrar registrar; //向全局变量 factories 增加 DIRECT_SESSION:new DirectSessionFactory() 对象

class DirectSessionRegistrar

class DirectSessionFactory : public SessionFactory //本地  device 上执行
  mutex sessions_lock_;
  std::vector<DirectSession*> sessions_ ; //所有的 DirectSession

class DirectSession : public Session
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

### GrpcSession

message CreateSessionRequest
  GraphDef graph_def = 1;
  ConfigProto config = 2;
  string target = 3; //目的地址

message CreateSessionResponse
  string session_handle = 1; //创建 session 后服务端返回该字符串，后续请求都需要携带该字段唯一标明该 session
  int64 graph_version = 2;

message ExtendSessionRequest //扩展之前的  graph, 比如增加 Node
  string session_handle = 1; //CreateSession 时返回的 session_handle
  GraphDef graph_def = 2;    //想要增加的 node
  int64 current_graph_version = 3;

message ExtendSessionResponse
  int64 new_graph_version = 4;

message RunStepRequest
  string session_handle = 1;
  repeated NamedTensorProto feed = 2;
  repeated string fetch = 3;
  repeated string target = 4; //
  RunOptions options = 5;
  string partial_run_handle = 6;

message RunStepResponse
  repeated NamedTensorProto tensor = 1;
  RunMetadata metadata = 2;

message PartialRunSetupRequest
  string session_handle = 1;
  repeated string feed = 2;
  repeated string fetch = 3;
  repeated string target = 4;

message PartialRunSetupResponse
  string partial_run_handle = 1;

message CloseSessionRequest
  string session_handle = 1;

message CloseSessionResponse

message ResetRequest
  repeated string container = 1;
  repeated string device_filters = 2; //至于匹配 device_filters 的被重置

message ResetResponse

message ListDevicesRequest
  string session_handle = 1;

message ListDevicesResponse
  repeated DeviceAttributes local_device = 1;
  repeated DeviceAttributes remote_device = 2;

由 MasterService::AsyncService 实现

实现了 master.proto

static GrpcSessionRegistrar registrar; //注册 "GRPC_SESSION", new GrpcSessionFactory(

class GrpcSessionRegistrar

class GrpcSessionFactory : public SessionFactory

class GrpcSession : public Session
  SessionOptions options_;                  //server_def.default_session_config()
  std::unique_ptr<MasterInterface> master_; //GrpcRemoteMaster(master_channel)
  mutex mu_;
  string handle_;                           // 从 CreateSessionResponse 中获取
  int64 current_graph_version_;             //默认 -1，用 CreateSessionRequest 的应答来设置

### 配置

struct SessionOptions
  Env* env;
  string target; 与 LocalMaster 对应，对于  grpc 为 grpc://${HOST}:${PORT}； 该字段必须为空
  ConfigProto config; //server_def.default_session_config()

message ThreadPoolOptionProto
  int32 num_threads = 1;   //如果为 0, 默认 CPU 的个数
  string global_name = 2;  //如果为空 Compute:$NUM

message GPUOptions {
  double per_process_gpu_memory_fraction = 1;
  string allocator_type = 2;
  int64 deferred_deletion_bytes = 3;
  bool allow_growth = 4;
  string visible_device_list = 5; //GPU 列表当与  ConfigProto.device_count 中 GPU 对应的数字不一致时，以 visible_device_list 中个数为准
  int32 polling_active_delay_usecs = 6;
  int32 polling_inactive_delay_msecs = 7;
  bool force_gpu_compatible = 8;

message ConfigProto
  map<string, int32> device_count = 1; //每种类型的设备的数量，比如 {"CPU",2}, {"GPU", 3}，在创建设备的时候就会创建对应个数的设备, 比如  /CPU:0, /CPU:1, /GPU:0, /GPU:1, /GPU2, 具体参考 device.md 的  full name 和 legacy name
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

message OptimizerOptions
  bool do_common_subexpression_elimination = 1; //  如果为  true，用子表达式消除(common subexpression elimination)优化
  bool do_constant_folding = 2; //如果为 True，用常量展开(constant folding)优化
  bool do_function_inlining = 4; // 如果为 True，用函数内联(function inlining)优化
  enum Level
    L1 = 0; //默认, 包括子表达式消除与常量展开
    L0 = -1; //不优化
  Level opt_level = 3;
  // Control the use of the compiler/jit.  Experimental.
  enum GlobalJitLevel
    DEFAULT = 0;  //默认不开启
    OFF = -1;
    ON_1 = 1; //值更大优化更激进
    ON_2 = 2;
  GlobalJitLevel global_jit_level = 5;

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

## 源码分析

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

### DirectSessionFactory

bool DirectSessionFactory::AcceptsOptions(const SessionOptions& options) //return options.target.empty();

Session* DirectSessionFactory::NewSession(const SessionOptions& options)

1. 创建一个 device
2.  根据  options，device 创建一个 session, 保存在 sessions_ 中

Status DirectSessionFactory::Reset(const SessionOptions& options, const std::vector<string>& containers)

用 containers 重置所有 sessions_

void DirectSessionFactory::Deregister(const DirectSession* session)

从  sessions_ 中删除 session

### DirectSession

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

### SessionMgr

SessionMgr::SessionMgr(
    WorkerEnv* worker_env, const string& default_worker_name,
    std::unique_ptr<WorkerCacheInterface> default_worker_cache,
    WorkerCacheFactory worker_cache_factory)

初始化数据成员

Status SessionMgr::CreateSession(const string& session, const ServerDef& server_def)

string SessionMgr::WorkerNameFromServerDef(const ServerDef& server_def) // "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}

Status SessionMgr::CreateSession(const string& session, const ServerDef& server_def) //构造  WorkerSession 对象，加入 sessions_ 中

Status SessionMgr::DeleteSession(const string& session) //将 session 对应的 WorkerSession 删除

WorkerSession* SessionMgr::WorkerSessionForSessionUnlocked(const string& session) // 从 sessions_ 中查找 session 对应的 WorkerSession，如果找到，返回，如果找不到返回 legacy_session_

WorkerSession* SessionMgr::WorkerSessionForSession(const string& session) // 从 sessions_ 中查找 session 对应的 WorkerSession，如果找到，返回 ，如果找不到返回 legacy_session_
WorkerSession* SessionMgr::LegacySession() // return &legacy_session_


### GrpcSession

Status GrpcSession::Create(const SessionOptions& options, std::unique_ptr<GrpcSession>* out_session)

创建一个 GrpcSession 并保存到  out_session

1. 创建 GrpcSession 对象 session
2. local_master_registry_  中查找  options.target 对应的 master(LocalMaster)
3. 设置  GrpcSession 的 master_ 为 master
4. 将 out_session 指向 session


Status GrpcSession::CreateImpl(CallOptions* call_options, const GraphDef& graph)

构建一个 CreateSessionRequest  请求
```
    CreateSessionRequest req;
    *req.mutable_config() = options_.config;
    *req.mutable_graph_def() = graph;
    req.set_target(options_.target);
    ReEncodeConsts(req.mutable_graph_def());
    CreateSessionResponse resp;
```

```
  master_->CreateSession(call_options, &req, &resp);
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    SetDeadline(&ctx, call_options->GetTimeout());
    master_.stub->CreateSession(ctx, &req, &resp)
        ::grpc::internal::BlockingUnaryCall(channel_, rpcmethod_ExtendSession_, ctx, req, resp);
    handle_ = resp.mutable_session_handle(
    current_graph_version_ = resp.graph_version();
```

Status GrpcSession::Create(const GraphDef& graph)

  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return CreateImpl(&call_options, graph);

Status GrpcSession::Create(const RunOptions& run_options, const GraphDef& graph)

  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return CreateImpl(&call_options, graph);

Status GrpcSession::ExtendImpl(CallOptions* call_options, const GraphDef& graph)

1. 如果  handle_ 为空，调用  Create 初始化

2. 构造一个 CreateSessionRequest 请求

```
  ExtendSessionRequest req;
  req.set_session_handle(handle_);
  *req.mutable_graph_def() = graph;
  req.set_current_graph_version(current_graph_version_);
  ExtendSessionResponse resp;
```

```
  master_->ExtendSession(call_options, &req, &resp);
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    SetDeadline(&ctx, call_options->GetTimeout());
    master_.stub->CreateSession(ctx, &req, &resp)
        ::grpc::internal::BlockingUnaryCall(channel_, rpcmethod_ExtendSession_, ctx, req, resp);
    current_graph_version_ = resp.graph_version();
```

Status GrpcSession::Extend(const GraphDef& graph)

  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return ExtendImpl(&call_options, graph);

Status GrpcSession::Extend(const RunOptions& run_options, const GraphDef& graph)

  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return ExtendImpl(&call_options, graph);

Status GrpcSession::RunHelper(const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata, const string& prun_handle)

构造 MutableRunStepRequestWrapper 消息，
```
  req->set_session_handle(handle_);
  master_->RunStep(call_options, req, resp);
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    SetDeadline(&ctx, call_options->GetTimeout());
    master_.stub->RunStep(ctx, &req, &resp)
        ::grpc::internal::BlockingUnaryCall(channel_, rpcmethod_CreateSession_, ctx, req, resp);
```
获取应答 MutableRunStepResponseWrapper 初始化 outputs, run_metadata

Status GrpcSession::Run(const RunOptions& run_options,
                        const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs,
                        RunMetadata* run_metadata) {

  return RunHelper(run_options, inputs, output_tensor_names, target_node_names, outputs, run_metadata, "");

Status GrpcSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs) {

  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return Run(run_options, inputs, output_tensor_names, target_node_names, outputs, nullptr);

Status GrpcSession::RunProto(CallOptions* call_options, MutableRunStepRequestWrapper* req, MutableRunStepResponseWrapper* resp)

```
  req->set_session_handle(handle_);
  master_->RunStep(call_options, req, resp);
    ::grpc::ClientContext ctx;
    ctx.set_fail_fast(false);
    SetDeadline(&ctx, call_options->GetTimeout());
    master_.stub->RunStep(ctx, &req, &resp)
        ::grpc::internal::BlockingUnaryCall(channel_, rpcmethod_RunStep_, ctx, req, resp);
```

Status GrpcSession::PRunSetup(const std::vector<string>& input_names, const std::vector<string>& output_names,
                              const std::vector<string>& target_nodes, string* handle) {


input_names, output_names, target_nodes, handle_ 构造 PartialRunSetupRequest 消息，

```
  master_->partialrunsetup(call_options, req, resp);
    ::grpc::clientcontext ctx;
    ctx.set_fail_fast(false);
    setdeadline(&ctx, call_options->gettimeout());
    master_.stub->partialrunsetup(ctx, &req, &resp)
        ::grpc::internal::blockingunarycall(channel_, rpcmethod_partialrunsetup_, ctx, req, resp);
```
并获取应答 PartialRunSetupResponse 初始化 handle

Status GrpcSession::PRun(const string& handle, const std::vector<std::pair<string, Tensor>>& inputs,
                         const std::vector<string>& output_names, std::vector<Tensor>* outputs)

  return RunHelper(run_options, inputs, output_names, {}, outputs, nullptr, handle);

Status GrpcSession::Close()

1. 创建 CloseSessionRequest
```
  master_->CloseSession(call_options, req, resp);
    ::grpc::clientcontext ctx;
    ctx.set_fail_fast(false);
    setdeadline(&ctx, call_options->gettimeout());
    master_.stub->CloseSession(ctx, &req, &resp)
        ::grpc::internal::blockingunarycall(channel_, rpcmethod_CloseSession_, ctx, req, resp);
```

Status GrpcSession::ListDevices(std::vector<DeviceAttributes>* response)

1. 创建 ListDevicesRequest

```
  master_->ListDevices(call_options, req, resp);
    ::grpc::clientcontext ctx;
    ctx.set_fail_fast(false);
    setdeadline(&ctx, call_options->gettimeout());
    master_.stub->ListDevices(ctx, &req, &resp)
        ::grpc::internal::blockingunarycall(channel_, rpcmethod_ListDevices_, ctx, req, resp);
```

3. 接受 ListDevicesResponse  设置 response

void GrpcSession::SetRemoteMaster(std::unique_ptr<MasterInterface> master) //master_ = std::move(master);

Status GrpcSession::Reset(const SessionOptions& options, const std::vector<string>& containers)

1. 创建 grpc channel ::grpc::CreateCustomChannel("dns:///" + options.target, ::grpc::InsecureChannelCredentials(), args);
2. 创建 grpc GrpcRemoteMaster 对象
3. 构造 ResetRequest

```
  master_->Reset(call_options, req, resp);
    ::grpc::clientcontext ctx;
    ctx.set_fail_fast(false);
    setdeadline(&ctx, call_options->gettimeout());
    master_.stub->Reset(ctx, &req, &resp)
        ::grpc::internal::blockingunarycall(channel_, rpcmethod_Reset_, ctx, req, resp);
```

### GrpcSessionFactory

bool AcceptsOptions(const SessionOptions& options)

options.target 以 “grpc://" 开头，返回  true

Session* NewSession(const SessionOptions& options)

新建一个 GrpcSession

Status Reset(const SessionOptions& options, const std::vector<string>& containers)

发送重置 ResetRequest，GrpcSession::Reset(options, containers);


### GrpcSessionRegistrar

GrpcSessionRegistrar()

factories 加入 {"GRPC_SESSION", new GrpcSessionFactory()}

