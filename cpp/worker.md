
每个 node 在一个 Device 上运行
graph 被划分成多个  subgraph
一个 subgraph 的所有 node 在一个 worker 上运行，也即一个 worker 可以拥有多个 device



GraphMgr 注意管理和 graph 相关的操作，每个  graph 对应一个  handler, 每个
hander 有一个  step_id，同一个  graph 可以包含多个 executor

源文件

./core/protobuf/worker_service.proto
./core/protobuf/worker.proto
./core/distributed_runtime/worker.h
./core/distributed_runtime/worker.cc
./core/distributed_runtime/worker_cache.h
./core/distributed_runtime/worker_cache_partial.cc
./core/distributed_runtime/worker_interface.h
./core/distributed_runtime/rpc/grpc_remote_worker.cc
./core/distributed_runtime/rpc/grpc_worker_cache.cc
./core/distributed_runtime/rpc/grpc_worker_service.cc
./core/distributed_runtime/rpc/grpc_worker_service_impl.cc


## 数据结构

NewGrpcWorkerCache(cc)
  new GrpcWorkerCache(cc, nullptr, "");

WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(cc, local_worker, local_target)
  new GrpcWorkerCache(cc, local_worker, local_target);

NewGrpcWorker(env)
    GrpcWorker(env)

NewGrpcWorkerService(woker, builder)
    GrpcWorkerService(worker, builder)

typedef std::function<void(const Status&)> StatusCallback;

WorkerEnv -> Worker

WorkerInterface -> WorkerFreeListCache -> WorkerSession

WorkerInterface -> Worker -> GrpcWorker

WorkerCacheInterface -> WorkerCachePartial -> GrpcWorkerCache

struct WorkerEnv                          //一个 woker 需要的相关依赖
  Env* env;                               // Env::Default()
  SessionMgr* session_mgr ;               // SessionMgr(worker_env_, name_prefix, worker_cache, WorkerCacheFactory(options, worker_cache))
  std::vector<Device*> local_devices;     // 所有已经注册的设备列表，默认只包含 CPU， 与 MasterEnv.local_devices  相同
  DeviceMgr* device_mgr;                  // DeviceMgr(worker_env_.local_devices)
  RendezvousMgrInterface* rendezvous_mgr; // RpcRendezvousMgr(&worker_env_) 接受队列中的请求
  thread::ThreadPool* compute_pool;       // ComputePool(sess_opts) ; Worker 执行请求线程池，master 为什么不需要?

class WorkerInterface //与 TensorFlow WorkerService 通信

class Worker : public WorkerInterface
  typedef WorkerInterface ME;
  WorkerEnv* const env_;  // Not owned.
  PartialRunMgr partial_run_mgr_; // id 与 CancellationManager 映射关系
  mutex mu_;
  CancellationManager* cancellation_manager_;

class GrpcWorker : public Worker

### worker cache

class WorkerCacheInterface
  virtual void ListWorkers(std::vector<string>* workers) const = 0;
  virtual WorkerInterface* CreateWorker(const string& target) = 0;
  virtual void ReleaseWorker(const string& target, WorkerInterface* worker) {
  virtual bool GetDeviceLocalityNonBlocking(const string& device, DeviceLocality* locality) = 0;
  virtual void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality, StatusCallback done) = 0;
  virtual void SetLogging(bool active) {}
  virtual void ClearLogs() {}
  virtual bool RetrieveLogs(int64 step_id, StepStats* ss) { return false; }

class WorkerCachePartial : public WorkerCacheInterface
  mutex mu_;
  typedef std::unordered_map<string, DeviceAttributes> StatusMap;
  StatusMap device_status_cache_;

用于为 channel_cache_ 中的每个 target 创建一个对应的 GrpcRemoteWorker

当初始化 GrpcWorkerCache 对象之后，

一方面，创建一个 "grpc_worker_cache" 线程从 completion_queue_ 中取数据调用  OnCompleted 方法。

另一方面，当没有远程节点建立连接的话，该队列始终是空的，当通过 CreateWorker 方法
创建于远程节点的连接之后，就可以给远程发送各种消息(可执行的方法参考GrpcWorkerMethod)，
要求其执行，执行的方式就是对 CreateWorker 返回的  worker 调用对应的方法。
如果 GetStatusAsync，CreateWorkerSessionAsync 等等。 要执行的方法会以消息
的形式发送到 CompletionQueue。而此时 grpc_worker_cache 线程就可以收到该
消息，就会调用对应的方法。

class GrpcWorkerCache : public WorkerCachePartial
  const string local_target_;             // 本地 target  格式 "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}
  WorkerInterface* const local_worker_;   // 本地 GrpcWorker(所有 WorkerCacheInterface 共享)
  GrpcCounter live_rpc_counter_;
  GrpcChannelCache* channel_cache_;       // 单机版 SparseGrpcChannelCache; 集群版 MultiGrpcChannelCache(caches)
  ::grpc::CompletionQueue completion_queue_;
  Thread* polling_thread_;                // "grpc_worker_cache" 从 completion_queue_ 中取数据调用  OnCompleted 方法。
  WorkerCacheLogger logger_;

class WorkerFreeListCache : public WorkerCacheInterface // 对 WorkerCacheInterface 的简单封装，在创建 WorkerCacheInterface 的时候， 如果已经存在，重用以前的
  std::unique_ptr<WorkerCacheInterface> wrapped_;    // 由该成员变量代理各种操作
  mutex mu_;                                         // 保护 workers_
  std::unordered_map<string, WorkerState> workers_;  // 保存 target : WorkerState 的映射关系
  struct WorkerState
    WorkerInterface* worker;
    // TODO(jeff,sanjay): Add reference count if we support eviction.

class GrpcWorkerService : public AsyncServiceInterface
  GrpcWorker* worker_ ;                               //负责处理各类消息，如 GetStatus 等
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_; //来自 builder
  grpc::WorkerService::AsyncService worker_service_;  //将不同种类的消息加入队列
  mutex shutdown_mu_;
  bool is_shutdown_;
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

### 消息

message GetStatusRequest

message GetStatusResponse
  repeated DeviceAttributes device_attributes = 1;

message CreateWorkerSessionRequest
  string session_handle = 1;
  ServerDef server_def = 2;

message CreateWorkerSessionResponse

message RegisterGraphRequest
  string session_handle = 1;
  GraphDef graph_def = 2;
  GraphOptions graph_options = 4;
  DebugOptions debug_options = 5;

message RegisterGraphResponse
  string graph_handle = 1;

message DeregisterGraphRequest
  string session_handle = 2; //获取 WorkerSession
  string graph_handle = 1;

message DeregisterGraphResponse

message CleanupAllRequest
  repeated string container = 1; //与  device_mgr 有关

message CleanupAllResponse

message RunGraphRequest
  string session_handle = 8;
  string graph_handle = 1;
  int64 step_id = 2;
  ExecutorOpts exec_opts = 5;
  repeated NamedTensorProto send = 3; //保存发送的 {name, Tensor} 映射关系
  repeated string recv_key = 4; //保存接收到的 Tensor 的 key
  bool is_partial = 6;
  bool is_last_partial_run = 7;

message RunGraphResponse
  repeated NamedTensorProto recv = 1;
  StepStats step_stats = 2;
  CostGraphDef cost_graph = 3;
  repeated GraphDef partition_graph = 4;

message CleanupGraphRequest
  int64 step_id = 1;

message CleanupGraphResponse

message RecvTensorRequest {
  int64 step_id = 1;
  string rendezvous_key = 2;
  bool dma_ok = 3;
  DeviceLocality client_locality = 4;
  DeviceLocality server_locality = 5;
  google.protobuf.Any transport_options = 6;

message RecvTensorResponse {
  TensorProto tensor = 1;
  bool is_dead = 2;
  int64 send_start_micros = 3;
  google.protobuf.Any transport_options = 4;

message LoggingRequest
  bool rpc_logging = 1;
  bool clear = 2;
  repeated int64 fetch_step_id = 3;

message LabeledStepStats
  int64 step_id = 1;
  StepStats step_stats = 2;

message LoggingResponse
  repeated LabeledStepStats step = 1;

message TraceOpts
  double duration = 1;
  bool use_step_profiler = 2;
  bool use_kernel_profiler = 3;
  bool use_extended_profiler = 4;
  bool use_gpu_profiler = 5;
  bool use_sample_profiler = 6;

message TracingRequest
  TraceOpts options = 1;

message TracingResponse

enum class GrpcWorkerMethod //与远程节点进行交互的消息类型
  kGetStatus,
  kCreateWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kLogging,
  kTracing,

static const int kGrpcNumWorkerMethods = static_cast<int>(GrpcWorkerMethod::kTracing) + 1;

service WorkerService
  rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);
  rpc CreateWorkerSession(CreateWorkerSessionRequest) returns (CreateWorkerSessionResponse);
  rpc RegisterGraph(RegisterGraphRequest) returns (RegisterGraphResponse);
  rpc DeregisterGraph(DeregisterGraphRequest) returns (DeregisterGraphResponse);
  rpc RunGraph(RunGraphRequest) returns (RunGraphResponse);
  rpc CleanupGraph(CleanupGraphRequest) returns (CleanupGraphResponse);
  rpc CleanupAll(CleanupAllRequest) returns (CleanupAllResponse);
  rpc RecvTensor(RecvTensorRequest) returns (RecvTensorResponse) {}
  rpc Logging(LoggingRequest) returns (LoggingResponse);
  rpc Tracing(TracingRequest) returns (TracingResponse);

class WorkerService
  class AsyncService : public ::grpc::Service

## 源码分析

### WorkerInterface

支持的操作

* CreateWorkerSession
* RegisterGraph
* DeregisterGraph
* CleanupGraph
* CleanupAll
* Logging
* Tracing

virtual void GetStatusAsync(const GetStatusRequest* request, GetStatusResponse* response, StatusCallback done) = 0;
virtual void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request, CreateWorkerSessionResponse* response, StatusCallback done) = 0;
virtual void RegisterGraphAsync(const RegisterGraphRequest* request, RegisterGraphResponse* response, StatusCallback done) = 0;
virtual void DeregisterGraphAsync(const DeregisterGraphRequest* request, DeregisterGraphResponse* response, StatusCallback done) = 0;
virtual void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request, MutableRunGraphResponseWrapper* repsonse, StatusCallback done) = 0;
virtual void RunGraphAsync(CallOptions* opts, const RunGraphRequest* request, RunGraphResponse* response, StatusCallback done)
```cpp
    RunGraphRequestWrapper* wrapped_request = new ProtoRunGraphRequest(request);
    MutableRunGraphResponseWrapper* wrapped_response =
        new NonOwnedProtoRunGraphResponse(response);
    RunGraphAsync(opts, wrapped_request, wrapped_response,
                  [wrapped_request, wrapped_response, done](const Status& s) {
                    done(s);
                    delete wrapped_request;
                    delete wrapped_response;
                  });
```

virtual MutableRunGraphRequestWrapper* CreateRunGraphRequest()  //return new MutableProtoRunGraphRequest;
virtual MutableRunGraphResponseWrapper* CreateRunGraphResponse() //new OwnedProtoRunGraphResponse;
Status GetStatus(const GetStatusRequest* request, GetStatusResponse* response) // return CallAndWait(&ME::GetStatusAsync, request, response);
Status CreateWorkerSession(const CreateWorkerSessionRequest* request, CreateWorkerSessionResponse* response) //CallAndWait(&ME::CreateWorkerSessionAsync, request, response);
Status RegisterGraph(const RegisterGraphRequest* request, RegisterGraphResponse* response) //CallAndWait(&ME::RegisterGraphAsync, request, response);
Status DeregisterGraph(const DeregisterGraphRequest* request, DeregisterGraphResponse* response) //CallAndWait(&ME::DeregisterGraphAsync, request, response);
Status CleanupGraph(const CleanupGraphRequest* request, CleanupGraphResponse* response) //CallAndWait(&ME::CleanupGraphAsync, request, response);
Status CleanupAll(const CleanupAllRequest* request, CleanupAllResponse* response) //CallAndWait(&ME::CleanupAllAsync, request, response);
Status Logging(const LoggingRequest* request, LoggingResponse* response) //CallAndWait(&ME::LoggingAsync, request, response);
Status Tracing(const TracingRequest* request, TracingResponse* response) //CallAndWait(&ME::TracingAsync, request, response);
Status CallAndWait(Method func, const Req* req, Resp* resp) //同步调用 func(req, resp, done)
RunGraphResponse* get_proto_from_wrapper(MutableRunGraphResponseWrapper* wrapper) //wrapper->get_proto();


### Worker

支持的操作

* CreateWorkerSessionAsync
* RegisterGraphAsync
* DeregisterGraphAsync
* RunGraphAsync
* CleanupGraphAsync
* CleanupAllAsync
* RecvTensorAsync
* LoggingAsync
* TracingAsync

void Worker::GetStatusAsync(const GetStatusRequest* request, GetStatusResponse* response, StatusCallback done)

直接从 env->device_mgr 中获取 DeviceAttributes 初始化 response.device_attributes，最后调用 done

void Worker::CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request, CreateWorkerSessionResponse* response, StatusCallback done)

1. env_->session_mgr->CreateSession(request->session_handle(), request->server_def());
2. done(s)

void Worker::RegisterGraphAsync(const RegisterGraphRequest* request, RegisterGraphResponse* response, StatusCallback done)

```cpp
  WorkerSession* session = env_->session_mgr->WorkerSessionForSession(request->session_handle());
  Status s = session->graph_mgr->Register(
      request->session_handle(), request->graph_def(), request->graph_options(),
      request->debug_options(), response->mutable_graph_handle());
  done(s);
```
void Worker::DeregisterGraphAsync(const DeregisterGraphRequest* request, DeregisterGraphResponse* response, StatusCallback done)

```
  WorkerSession* session = env_->session_mgr->WorkerSessionForSession(request->session_handle());
  Status s = session->graph_mgr->Deregister(request->graph_handle());
  done(s);
```

void Worker::AbortStep(int64 step_id)

```
  Rendezvous* rendez = env_->rendezvous_mgr->Find(step_id);
  SchedNonBlockingClosureAfter(1000000, [rendez, step_id]() {
    // Delay a bit before aborting the step. This way, the root
    // cause may return first back to the client instead of this
    // cancellation generated abort error.
    rendez->StartAbort(errors::Aborted("Step ", step_id));
    rendez->Unref();
  });
```

Status Worker::PrepareRunGraph(RunGraphRequestWrapper* req, GraphMgr::NamedTensors* in, GraphMgr::NamedTensors* out)

```
  static Tensor empty_tensor(DT_FLOAT);
  if (req->num_sends() > 0) {
    Tensor val;
    for (size_t i = 0; i < req->num_sends(); ++i) {
      TF_RETURN_IF_ERROR(req->SendValue(i, &val));
      in->insert({req->send_key(i), val});
    }
  }
  for (size_t i = 0; i < req->num_recvs(); ++i) {
    out->insert({req->recv_key(i), empty_tensor});
  }
  return Status::OK();
```

void Worker::RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request, MutableRunGraphResponseWrapper* response, StatusCallback done)

```
  if (request->is_partial()) {
    DoPartialRunGraph(opts, request, response, std::move(done));
  } else {
    DoRunGraph(opts, request, response, std::move(done));
  }
```

MutableRunGraphRequestWrapper* Worker::CreateRunGraphRequest() //return new InMemoryRunGraphRequest;
MutableRunGraphResponseWrapper* Worker::CreateRunGraphResponse() // return new InMemoryRunGraphResponse;

void Worker::DoRunGraph(CallOptions* opts, RunGraphRequestWrapper* request, MutableRunGraphResponseWrapper* response, StatusCallback done)

1. 根据 parsed.src_incarnation 获取  WorkerSession session.
2. 创建并初始化 StepStatsCollector，CancellationManager
3. 调用 session->graph_mgr->ExecuteAsync()
4. session->graph_mgr->RecvOutputs 根据 step_id 获取 Tensor
5. 用 Tensor 初始化  response

void Worker::DoPartialRunGraph(CallOptions* opts, RunGraphRequestWrapper* request, MutableRunGraphResponseWrapper* response, StatusCallback done)

1. 根据 parsed.src_incarnation 获取  WorkerSession session.
2. 创建并初始化 StepStatsCollector，CancellationManager
3. session->graph_mgr->RecvOutputsAsync(
4. 用 Tensor 初始化  response

void Worker::CleanupGraphAsync(const CleanupGraphRequest* request, CleanupGraphResponse* response, StatusCallback done)

```
  const int64 step_id = request->step_id();
  env_->rendezvous_mgr->Cleanup(step_id);
  done(Status::OK());
```

void Worker::CleanupAllAsync(const CleanupAllRequest* request, CleanupAllResponse* response, StatusCallback done)

```
  std::vector<string> containers;
  for (const auto& c : request->container()) containers.push_back(c);
  env_->device_mgr->ClearContainers(containers);
  done(Status::OK());
```

void Worker::LoggingAsync(const LoggingRequest* request, LoggingResponse* response, StatusCallback done)
  done(errors::Unimplemented("Logging"));

void Worker::TracingAsync(const TracingRequest* request, TracingResponse* response, StatusCallback done)
  done(errors::Unimplemented("Tracing"));

Status Worker::PrepareRecvTensor(const Rendezvous::ParsedKey& parsed, Device** src_dev)

1. 在 env_->device_mgr 中，查找 parsed.src_device  对应的 src_dev(Device)
2. src_dev->attributes().incarnation() 与 parsed.src_incarnation 必须相同

void Worker::RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request, TensorResponse* response, StatusCallback done)

  done(errors::Unimplemented("Worker::RecvTensorAsync()"));

### WorkerCachePartial

bool WorkerCachePartial::GetDeviceLocalityNonBlocking(const string& device_name, DeviceLocality* locality)

从  device_status_cache_ 中查找 name 对应的 DeviceAttributes，并获取其中的 locality 保存在  locality

void WorkerCachePartial::GetDeviceLocalityAsync(const string& device_name, DeviceLocality* locality, StatusCallback done)

如果找不到 device_name 对应的 DeviceLocality, 刷新后重试。最后调用  done

Status WorkerCachePartial::RefreshDeviceStatus(const string& device_name)

创建  worker，发送 GetStatusRequest 获取设备， 更新 device_status_cache_

void WorkerCachePartial::FlushStatusCache()

清空 device_status_cache_

### WorkerFreeListCache

void ListWorkers(std::vector<string>* workers) //wrapped_->ListWorkers(workers);
WorkerInterface* CreateWorker(const string& target) // 如果 workers_ 中已经存在，直接返回，如果不存在，创建并加入  workers_，返回新建的 worker
void ReleaseWorker(const string& target, WorkerInterface* worker) //nothint to do
bool GetDeviceLocalityNonBlocking(const string& device, DeviceLocality* locality) //return wrapped_->GetDeviceLocalityNonBlocking(device, locality);
void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality, StatusCallback done) //wrapped_->GetDeviceLocalityAsync(device, locality, done);
void SetLogging(bool active) //wrapped_->SetLogging(active);
bool RetrieveLogs(int64 step_id, StepStats* ss) //return wrapped_->RetrieveLogs(step_id, ss);

### WorkerSession

WorkerSession::WorkerSession(const string& worker_name,
                             std::unique_ptr<WorkerCacheInterface> worker_cache,
                             std::unique_ptr<DeviceMgr> device_mgr,
                             std::unique_ptr<GraphMgr> graph_mgr)

初始化  WorkerSession 各个成员变量


### grpc_remote_worker

WorkerInterface* NewGrpcRemoteWorker(GrpcCounter* live_rpc_counter, SharedGrpcChannelPtr channel, ::grpc::CompletionQueue* completion_queue, WorkerCacheLogger* logger);


### GrpcWorkerCache

void ListWorkers(std::vector<string>* workers) //channel_cache_->ListWorkers(workers);
WorkerInterface* CreateWorker(const string& target) // 如果 target 对应的不是本地 local_worker_ 就创建一个 GrpcRemoteWorker
void ReleaseWorker(const string& target, WorkerInterface* worker) //删除 worker
void SetLogging(bool v) //logger_.SetLogging(v);
void ClearLogs() // logger_.ClearLogs();
bool RetrieveLogs(int64 step_id, StepStats* ss) //logger_.RetrieveLogs(step_id, ss);

### GrpcWorker

GrpcWorker::GrpcWorker(WorkerEnv* worker_env) // Worker(worker_env)
WorkerEnv* GrpcWorker::env() //return env_;

void GrpcWorker::RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                                 ::grpc::ByteBuffer* response, StatusCallback done)

1. 解析 request->rendezvous_key()

2. 如果是 GPU，TODO
2.1. GPUUtil::SetProtoFromGPU(val, src_dev, send_dev_context, tmp->mutable_tensor(), is_dead, response_ready);
2.2. grpc::EncodeRecvTensorResponseToByteBuffer(*tmp, response);

3. 如果不是 GPU，将收到的 Tensor 编码成 response

### GrpcWorkerService


GrpcWorkerService(GrpcWorker* worker, ::grpc::ServerBuilder* builder)

```
worker_(worker), is_shutdown_(false) {
builder->RegisterService(&worker_service_);
cq_ = builder->AddCompletionQueue();
```

void Shutdown() // is_shutdown_ = true; did_shutdown = true;

void HandleRPCsLoop() //依次从队列中取出一个元素，调用它的 OnCompleted 方法
void Schedule(std::function<void()> f) //worker_->env()->compute_pool->Schedule(std::move(f));
void GetStatusHandler(WorkerCall<GetStatusRequest, GetStatusResponse>* call)

```
worker_->GetStatus(&call->request, &call->response)
call->SendResponse(ToGrpcStatus(s));
```

void CreateWorkerSessionHandler(WorkerCall<CreateWorkerSessionRequest, CreateWorkerSessionResponse>* call)

```
worker_->CreateWorkerSession(&call->request, &call->response);
call->SendResponse(ToGrpcStatus(s));
```

void CleanupAllHandler(WorkerCall<CleanupAllRequest, CleanupAllResponse>* call)

```
worker_->CleanupAll(&call->request, &call->response);
call->SendResponse(ToGrpcStatus(s));
```

void RegisterGraphHandler(WorkerCall<RegisterGraphRequest, RegisterGraphResponse>* call)

```
worker_->RegisterGraph(&call->request, &call->response);
call->SendResponse(ToGrpcStatus(s));
```

void DeregisterGraphHandler(WorkerCall<DeregisterGraphRequest, DeregisterGraphResponse>* call)

```
worker_->DeregisterGraph(&call->request, &call->response);
call->SendResponse(ToGrpcStatus(s));
```

void RunGraphHandler(WorkerCall<RunGraphRequest, RunGraphResponse>* call)

```
      CallOptions* call_opts = new CallOptions;
      ProtoRunGraphRequest* wrapped_request =
          new ProtoRunGraphRequest(&call->request);
      NonOwnedProtoRunGraphResponse* wrapped_response =
          new NonOwnedProtoRunGraphResponse(&call->response);
      worker_->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                             [call, call_opts, wrapped_request,
                              wrapped_response](const Status& s) {
                               call->ClearCancelCallback();
                               delete call_opts;
                               delete wrapped_request;
                               delete wrapped_response;
                               call->SendResponse(ToGrpcStatus(s));
```

void RecvTensorHandlerRaw(WorkerCall<RecvTensorRequest, ::grpc::ByteBuffer>* call)

```
      CallOptions* call_opts = new CallOptions;
      call->SetCancelCallback([call_opts]() { call_opts->StartCancel(); });
      worker_->RecvTensorAsync(call_opts, &call->request, &call->response,
                               [call, call_opts](const Status& s) {
                                 call->ClearCancelCallback();
                                 delete call_opts;
                                 call->SendResponse(ToGrpcStatus(s));
                               });
```

void CleanupGraphHandler(WorkerCall<CleanupGraphRequest, CleanupGraphResponse>* call)

```
    Status s = worker_->CleanupGraph(&call->request, &call->response);
    call->SendResponse(ToGrpcStatus(s));
```

void LoggingHandler(WorkerCall<LoggingRequest, LoggingResponse>* call)

```
    Status s = worker_->Logging(&call->request, &call->response);
    call->SendResponse(ToGrpcStatus(s));
```

void TracingHandler(WorkerCall<TracingRequest, TracingResponse>* call)

```
    Status s = worker_->Tracing(&call->request, &call->response);
    call->SendResponse(ToGrpcStatus(s));
```

void EnqueueRecvTensorRequestRaw()

```
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
           RecvTensorRequest, ::grpc::ByteBuffer>::
          EnqueueRequestForMethod(
              &worker_service_, cq_.get(),
              static_cast<int>(GrpcWorkerMethod::kRecvTensor),
              &GrpcWorkerService::RecvTensorHandlerRaw,
              true /* supports cancel*/);
```

### Grpc Worker Service

const char* GrpcWorkerMethodName(GrpcWorkerMethod id)

根据  GrpcWorkerMethod 获取对应的字符串

WorkerService::AsyncService::AsyncService()

```cpp
  for (int i = 0; i < kGrpcNumWorkerMethods; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        GrpcWorkerMethodName(static_cast<GrpcWorkerMethod>(i)),
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
```
