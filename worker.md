
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

WorkerInterface -> GraphMgr

typedef std::function<void(const Status&)> StatusCallback;

WorkerEnv -> Worker

WorkerInterface -> WorkerFreeListCache -> WorkerSession

struct WorkerEnv
  Env* env = nullptr;
  SessionMgr* session_mgr = nullptr;
  std::vector<Device*> local_devices;
  DeviceMgr* device_mgr = nullptr;
  RendezvousMgrInterface* rendezvous_mgr = nullptr;
  thread::ThreadPool* compute_pool = nullptr;

class WorkerInterface

class Worker : public WorkerInterface
  typedef WorkerInterface ME;
  WorkerEnv* const env_;  // Not owned.
  PartialRunMgr partial_run_mgr_; // id 与 CancellationManager 映射关系
  mutex mu_;
  CancellationManager* cancellation_manager_ GUARDED_BY(mu_);

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
  StatusMap device_status_cache_ GUARDED_BY(mu_);

class WorkerFreeListCache : public WorkerCacheInterface
  std::unique_ptr<WorkerCacheInterface> wrapped_;
  mutex mu_;
  std::unordered_map<string, WorkerState> workers_ GUARDED_BY(mu_);
  struct WorkerState
    WorkerInterface* worker;
    // TODO(jeff,sanjay): Add reference count if we support eviction.

struct WorkerSession
  const string worker_name;
  const std::unique_ptr<WorkerCacheInterface> worker_cache; //new WorkerFreeListCache(worker_cache)
  const std::unique_ptr<DeviceMgr> device_mgr;
  const std::unique_ptr<GraphMgr> graph_mgr;


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

 确保 parsed.src_device 在 env_->device_mgr  中的 incarnation 与 parsed.src_incarnation 是一样的

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

