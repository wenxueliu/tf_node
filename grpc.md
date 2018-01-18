
WorkerFreeListCache.CreateWorker
CreateWorkerSessions
NewRemoteDevices
RecvFromRemoteAsync


## 代码流

NewGrpcMasterService(master, default_timeout_in_ms, builder);
    GrpcMasterService(master, default_timeout_in_ms, builder);

GrpcRemoteMaster.NewGrpcMaster(channel)
    GrpcRemoteMaster(channel)
        MasterService::NewStub(channel, options)
            MasterService::Stub(channel)

MasterService::AsyncService::AsyncService()

WorkerInterface* NewGrpcRemoteWorker(live_rpc_counter, channel, completion_queue, logger)
    GrpcRemoteWorker(live_rpc_counter, std::move(channel), completion_queue, logger);


NewGrpcChannelCache(channel_spec, GetChannelCreationFunction())
  std::vector<GrpcChannelCache*> caches;
  for (auto& job : spec.host_ports_jobs())
    caches.push_back(new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func));
  return caches.size() == 1 ? caches[0] : new MultiGrpcChannelCache(caches);



MasterInterface -> GrpcRemoteMaster -> grpc::MasterService::Stub -> ::grpc::internal::BlockingUnaryCall

AsyncServiceInterface -> GrpcMasterService -> Master -> WorkerSession


## 数据结构

### Worker

class GrpcRemoteMaster : public MasterInterface
  std::unique_ptr<grpc::MasterService::Stub> stub_;

class GrpcRemoteWorker : public WorkerInterface // 每一  grpc:Channel 对应一个 GrpcRemoteWorker
  GrpcCounter* const counter_;   //对应的计数器，所有 GrpcRemoteWorker 共享
  SharedGrpcChannelPtr channel_; //对应的 grpc:Channel
  ::grpc::GenericStub stub_;     // 与 channel_ 相同
  ::grpc::CompletionQueue* cq_;  //对应的队列，所有 GrpcRemoteWorker 共享
  const ::grpc::string getstatus_;
  const ::grpc::string createworkersession_;
  const ::grpc::string registergraph_;
  const ::grpc::string deregistergraph_;
  const ::grpc::string rungraph_;
  const ::grpc::string cleanupgraph_;
  const ::grpc::string cleanupall_;
  const ::grpc::string recvtensor_;
  const ::grpc::string logging_;
  const ::grpc::string tracing_;
  WorkerCacheLogger* logger_;

  class RPCState : public GrpcClientCQTag
    GrpcCounter* const counter_; //
    CallOptions* call_opts_;
    ::grpc::ClientContext context_;
    std::unique_ptr<::grpc::GenericClientAsyncReaderWriter> call_; //grpc::Channel->Call(&context_, method, cq, this)
    ResponseMessage* response_;       //应答消息
    ::grpc::ByteBuffer request_buf_;  //将 request 序列化之后保存在此 buffer
    ::grpc::ByteBuffer response_buf_;
    ::grpc::Status status_;
    StatusCallback done_;
    std::atomic<bool> failure_;             //初始化为 false
    std::atomic<int> remaining_callbacks_;
    Notification call_initialized_;         //发送端与接收端的消息同步

### Master Service

class AsyncServiceInterface
  virtual void HandleRPCsLoop() = 0;
  virtual void Shutdown() = 0;

class GrpcMasterService : public AsyncServiceInterface
  Master* master_impl_ ;  //
  const int64 default_timeout_in_ms_; //server_def_.default_session_config().operation_timeout_in_ms()
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;
  mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

class MasterService
  class StubInterface
  class Stub final : public StubInterface //::grpc::internal::BlockingUnaryCall 实现
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::internal::RpcMethod rpcmethod_CreateSession_;
    const ::grpc::internal::RpcMethod rpcmethod_ExtendSession_;
    const ::grpc::internal::RpcMethod rpcmethod_PartialRunSetup_;
    const ::grpc::internal::RpcMethod rpcmethod_RunStep_;
    const ::grpc::internal::RpcMethod rpcmethod_CloseSession_;
    const ::grpc::internal::RpcMethod rpcmethod_ListDevices_;
    const ::grpc::internal::RpcMethod rpcmethod_Reset_;

  class AsyncService : public ::grpc::Service


class GrpcCounter
  mutex mu_;
  condition_variable empty_;
  int counter_ = 0;

class GrpcClientCQTag
  virtual void OnCompleted(bool ok) = 0;

### 服务调用机制

1. 发送端将收到的请求封装为 Tag 放入队列
2. 接受端从队列取出 Tag， 调用  Tag 的 OnCompleted 方法(根据 Tag callback_ 不同调用 call_ 对应不同的方法)

这样做的好处
1. 抽象：所有的消息都抽象成  RequestReceived, RequestCancelled
2. 解耦：对具体类型的消息处理，初始化 Tag, 并调用 OnCompleted 方法即可

对于  grpc 的进一步要求

1. 存放各个请求消息的队列
2. 每个 Service 实现 EnqueueFunction, 将消息放到请求队列中

class UntypedCall //与类型无关的调用
  virtual void RequestReceived(Service* service, bool ok) = 0;
  virtual void RequestCancelled(Service* service, bool ok) = 0;
  class Tag
    enum Callback
        kRequestReceived
        kResponseSent
        kCancelled
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;

class Call : public UntypedCall<Service> //转为 grpc 实现的一个处理器
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  RequestMessage request;
  ResponseMessage response;
  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;
  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};
  Tag cancelled_tag_{this, Tag::kCancelled};
  mutex mu_;
  std::function<void()> cancel_callback_; //请求被取消时的回调函数

### 读写缓存

class GrpcByteBufferSource : public ::grpc::protobuf::io::ZeroCopyInputStream
  std::vector<::grpc::Slice> slices_;
  int cur_;          // Current slice index.
  int left_;         // Number of bytes in slices_[cur_] left to yield.
  const char* ptr_;  // Address of next byte in slices_[cur_] to yield.
  ::grpc::protobuf::int64 byte_count_;

class GrpcBufferWriter : public ::grpc::protobuf::io::ZeroCopyOutputStream
  const int block_size_;
  int64_t byte_count_;
  grpc_slice_buffer* slice_buffer_; //保存写的数据的缓存
  bool have_backup_;
  grpc_slice backup_slice_;
  grpc_slice slice_;

class GrpcBufferReader : public ::grpc::protobuf::io::ZeroCopyInputStream

  int64_t byte_count_;
  int64_t backup_count_; //跳过的跳过的应该的的数据的长度
  grpc_byte_buffer_reader reader_;
  grpc_slice slice_;

具体每次读多少数据不是由客户端控制，客户端智能获取每次读的数据的首指针和读到的数据的长度

class UnlimitedSizeProtoSerializationTraits

### Grpc Server

typedef std::function<RendezvousMgrInterface*(const WorkerEnv*)> RendezvousMgrCreationFunction;
typedef std::function<void(const WorkerEnv*, ::grpc::ServerBuilder*)> ServiceInitFunction;

class GrpcServer : public ServerInterface
  ServerDef server_def_;
  Env* env_; Env::Default()
  int bound_port_ = 0;
  mutex mu_;
  enum State { NEW, STARTED, STOPPED };
  State state_ ;
  // Implementation of a TensorFlow master, and RPC polling thread.
  MasterEnv master_env_; //参见 MasterEnv
  std::unique_ptr<Master> master_impl_; // CreateMaster(&master_env_)
  AsyncServiceInterface* master_service_; // NewGrpcMasterService(master_impl_.get(), config.operation_timeout_in_ms(), &builder)
  std::unique_ptr<Thread> master_thread_;
  WorkerEnv worker_env_; //参见 WorkerEnv
  std::unique_ptr<GrpcWorker> worker_impl_; // NewGrpcWorker(&worker_env_);
  AsyncServiceInterface* worker_service_; // NewGrpcWorkerService(worker_impl_.get(), &builder)
  std::unique_ptr<Thread> worker_thread_;
  std::unique_ptr<::grpc::Server> server_; ServerBuilder 创建

struct WorkerCacheFactoryOptions
  const ClusterDef* cluster_def; //server_def.cluster()
  const string* job_name; //server_def.job_name()
  int task_index; // server_def.task_index()
  const string* protocol; //server_def.protocol()

struct WorkerEnv
  Env* env; //Env::Default()
  SessionMgr* session_mgr ; new SessionMgr(worker_env_, "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}, worker_cache, WorkerCacheFactory(options, worker_cache))
  std::vector<Device*> local_devices; //与  MasterEnv.local_devices  相同
  DeviceMgr* device_mgr; //new DeviceMgr(worker_env_.local_devices)
  RendezvousMgrInterface* rendezvous_mgr; //RpcRendezvousMgr(&worker_env_)
  thread::ThreadPool* compute_pool; //ComputePool(sess_opts)

struct MasterEnv {
  Env* env ; Env::Default()
  WorkerCacheInterface* worker_cache ; NewGrpcWorkerCacheWithLocalWorker(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()), worker_impl_.get(), "/job:"${options.job_name}"/replica:0/task:"${options.task_index})
  const OpRegistryInterface* ops ; //OpRegistry::Global();
  std::vector<Device*> local_devices; //所有已经注册的设备列表，默认只包含 CPU
  `std::function<MasterSession*(SessionOptions, MasterEnv*,
      std::unique_ptr<std::vector<std::unique_ptr<Device>>>,
      std::unique_ptr<WorkerCacheInterface>,
      std::unique_ptr<DeviceSet> device_set)>
      master_session_factory;` // return new MasterSession(options, env, std::move(remote_devs), std::move(worker_cache), std::move(device_set), CreateNoOpStatsPublisher);

  `std::function<Status(const WorkerCacheFactoryOptions&, WorkerCacheInterface**)> worker_cache_factory;` // NewGrpcWorkerCacheWithLocalWorker(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()), worker_impl_.get(), "/job:"${options.job_name}"/replica:0/task:"${options.task_index})


### Grpc Cache

class GrpcChannelSpec
  struct HostPortsJob
    const string job_id; //job name
    const std::map<int, string> host_ports; // task_id : host:port
  std::vector<HostPortsJob> host_ports_jobs_; //server_def.cluster_def
  std::set<string> job_ids_; //server_def.cluster_def

class GrpcChannelCache
  virtual void ListWorkers(std::vector<string>* workers) const = 0;
  virtual SharedGrpcChannelPtr FindWorkerChannel(const string& target) = 0;
  virtual string TranslateTask(const string& task) = 0;

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;
typedef std::function<SharedGrpcChannelPtr(string)> ChannelCreationFunction;


class CachingGrpcChannelCache : public GrpcChannelCache
  mutex mu_;
  std::unordered_map<string, SharedGrpcChannelPtr> channels_;

class MultiGrpcChannelCache : public CachingGrpcChannelCache
  const std::vector<GrpcChannelCache*> caches_;
  mutex mu_;
  std::unordered_map<string, GrpcChannelCache*> target_caches_; //target : GrpcChannelCache, 避免解析 target 中 task

class SparseGrpcChannelCache : public CachingGrpcChannelCache
  const string job_id_; //job name
  const std::map<int, string> host_ports_; //{task_id : host:port}
  const ChannelCreationFunction channel_func_;

## 源码分析

###Call

Call(HandleRequestFunction handle_request_function) : handle_request_function_(handle_request_function), responder_(&ctx_) {}
void RequestReceived(Service* service, bool ok) //(service->*handle_request_function_)(this);
void SendResponse(::grpc::Status status) //responder_.Finish(response, status, &response_sent_tag_);
void RequestCancelled(Service* service, bool ok) //cancel_callback_();
void SetCancelCallback(std::function<void()> callback) //cancel_callback_ = std::move(callback);
void ClearCancelCallback() //cancel_callback_ = nullptr;
std::multimap<::grpc::string_ref, ::grpc::string_ref>& client_metadata() //return ctx_.client_metadata();
void RegisterCancellationHandler()  //ctx_.AsyncNotifyWhenDone(&cancelled_tag_);

static void EnqueueRequest(GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
    EnqueueFunction enqueue_function, HandleRequestFunction handle_request_function, bool supports_cancel)

调用 grpc_service->*enqueue_function 将请求放到 cq 队列中
```cpp
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }
    (grpc_service->*enqueue_function)(&call->ctx_, &call->request,
                                      &call->responder_, cq, cq,
                                      &call->request_received_tag_);
```

static void EnqueueRequestForMethod(GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function, bool supports_cancel)

调用 grpc_service->RequestAsyncUnary 将请求放到 cq 队列中
```cpp
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    if (supports_cancel) {
      call->RegisterCancellationHandler();
    }

    // Initial ref for call handed to grpc; released in Tag callback.
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
```

### grpc_util

Status FromGrpcStatus(const ::grpc::Status& s) //grpc::Status 到 tensorflow:Status 的转换
::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) //grpc::Status 到 tensorflow:Status 的转换
string GrpcIdKey() //return "tf-rpc";
void GrpcUnparseProto(const protobuf::Message& src, ::grpc::ByteBuffer* dst) //src 序列化之后存储到 dst
bool GrpcParseProto(const ::grpc::ByteBuffer& src, protobuf::Message* dst) // src 解序列化之后存储到  dst

### GrpcMasterService

GrpcMasterService(Master* master, int64 default_timeout_in_ms, ::grpc::ServerBuilder* builder)

``` cpp
    master_impl_(master), default_timeout_in_ms_(default_timeout_in_ms), is_shutdown_(false)
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
```
void Shutdown() // is_shutdown_ = true; did_shutdown = true;

void HandleRPCsLoop()

1. 将 CreateSession,ExtendSession,PartialRunSetup,RunStep,CloseSession,ListDevices,Reset 加入队列
2. 从队列 cq_ 中取元素:
2.1 如果 tag 不为空，调用 callback_tag->OnCompleted(this, ok);
2.2 如果 tag 为空，调用 cq_->Shutdown();

void CreateSessionHandler(MasterCall<CreateSessionRequest, CreateSessionResponse>* call)

1. 创建一个 MasterSession，加入  master_impl_.sessions_ 中
2. 将  CreateSession 加入 cq_ 队列

void ExtendSessionHandler(MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call)

1. master_impl_->ExtendSession
2. 将 ExtendSession 加入 cq_

void PartialRunSetupHandler(MasterCall<PartialRunSetupRequest, PartialRunSetupResponse>* call)

1. master_impl_->PartialRunSetup
2. 将 PartialRunSetup 加入 cq_

void RunStepHandler(MasterCall<RunStepRequest, RunStepResponse>* call)

1. master_impl_->RunStep
2. 将 RunSetup 加入 cq_

void CloseSessionHandler(MasterCall<CloseSessionRequest, CloseSessionResponse>* call)

1. master_impl_->CloseSession
2. 将 CloseSession 加入 cq_

void ListDevicesHandler(MasterCall<ListDevicesRequest, ListDevicesResponse>* call)

1. master_impl_->ListDevices
2. 将 ListDevices 加入 cq_

void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call)

1. master_impl_->Reset
2. 将 Reset 加入 cq_

port::Tracing::TraceMe* TraceRpc(StringPiece name, const std::multimap<::grpc::string_ref, ::grpc::string_ref>& metadata) //new port::Tracing::TraceMe(name, id);


### GrpcBufferWriter

bool GrpcBufferWriter::Next(void** data, int* size)

1. byte_count_ 增加 size
2. data 指向写的数据的首指针，size 表示写的数据的大小
3. 将要写的数据写入 slice_buffer_

void BackUp(int count)

TODO

### GrpcBufferReader

void GrpcBufferReader::ReaderInit(NewReaderInitAPI ptr, grpc_byte_buffer_reader* reader, grpc_byte_buffer* buffer)

(g_core_codegen_interface->*ptr)(reader, buffer);

bool GrpcBufferReader::Next(const void** data, int* size)

1. byte_count_ 增加 size
2. 从 reader_ 中读 slices_ 的数据，data 指向读到数据的首指针，size 为读的数据的大小

void BackUp(int count) //backup_count_ = count;

bool Skip(int count) // 跳过 count byte 数据

grpc::protobuf::int64 ByteCount() //byte_count_ - backup_count_;

### UnlimitedSizeProtoSerializationTraits

static Status Serialize(const T& msg, grpc_byte_buffer** bp, bool* own_buffer)

TODO

static Status Deserialize(grpc_byte_buffer* buffer, T* msg, int max_message_size = INT_MAX)

TODO

### GrpcServer

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)

```
    server_def_(server_def)
    env_(env)
    state_(NEW)
```

Status GrpcServer::Init(ServiceInitFunction service_func, const RendezvousMgrCreationFunction& rendezvous_mgr_func)

1. 初始化  worker_env_ 和  master_env_
1.1. 创建 Device "/job/${server_def_.job_name()}/replica:0/task:${server_def_.task_index()}" 保存在  master_env_.local_devices 中
1.2. 从 cluster 的 server_def_.cluster().job() 中查找与 server_def_.job.name() 名称相同的  job, 并解析端口
1.3. 创建一个 Master
1.4. 创建一个 GrpcMasterService
1.5. 创建一个 Worker
1.6. 创建一个 GrpcWorkerService
1.7. service_func(&worker_env_, &builder);
1.8. NewGrpcChannelCache(ParseChannelSpec(options, &channel_spec), GetChannelCreationFunction()))
1.8. worker_cache : NewGrpcWorkerCacheWithLocalWorker(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()), worker_impl_.get(), "/job:"${options.job_name}"/replica:0/task:"${options.task_index})
1.8. worker_env_.session_mgr = new SessionMgr(worker_env_, "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}, worker_cache, WorkerCacheFactory(options, worker_cache))
1.9. ComputePool(sess_opts);
1.10. new MasterSession
2. 注册 grpc://localhost:${PORT{} : MasterInfo(master_impl_, config.operation_timeout_in_ms() 到全局变量  local_master_registry_

Status GrpcServer::Start()

开启两个线程
"TF_master_service": master_service_->HandleRPCsLoop()
"TF_worker_service": worker_service_->HandleRPCsLoop()
state_ = STARTED;

Status GrpcServer::Stop() //略
Status GrpcServer::Join() //略

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(const ServerDef& server_def)  //::grpc::InsecureServerCredentials();
ChannelCreationFunction GrpcServer::GetChannelCreationFunction() //ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
std::unique_ptr<Master> GrpcServer::CreateMaster(MasterEnv* master_env) //std::unique_ptr<Master>(new Master(master_env, 0.0));

Status GrpcServer::Create(const ServerDef& server_def, Env* env, std::unique_ptr<ServerInterface>* out_server)

1. GrpcServer(server_def, env == nullptr ? Env::Default() : env)->init(service_func, NewRpcRendezvousMgr)
2. out_server = std::move(ret);

bool AcceptsOptions(const ServerDef& server_def) //return server_def.protocol() == "grpc";
Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) //GrpcServer::Create(server_def, Env::Default(), out_server);


### GrpcChannel

Status NewHostPortGrpcChannel(const string& target, SharedGrpcChannelPtr* channel_pointer)

  channel_pointer = ::grpc::CreateCustomChannel("dns:///" + target, ::grpc::InsecureChannelCredentials(), args);


ChannelCreationFunction ConvertToChannelCreationFunction(const std::function<Status(string, SharedGrpcChannelPtr*)>& new_channel_func_ptr)

``` cpp
  return [new_channel_func_ptr](const string& target) -> SharedGrpcChannelPtr {
    SharedGrpcChannelPtr channel_ptr;
    if (new_channel_func_ptr(target, &channel_ptr).ok()) {
      return channel_ptr;
    } else {
      return nullptr;
    }
  };
```

Status GrpcChannelSpec::AddHostPortsJob(const string& job_id, const std::vector<string>& host_ports) //初始化 job_ids_ 和 host_ports_jobs_
Status GrpcChannelSpec::AddHostPortsJob(const string& job_id, const std::map<int, string>& host_ports) //初始化 job_ids_ 和 host_ports_jobs_

SharedGrpcChannelPtr CachingGrpcChannelCache::FindWorkerChannel(const string& target) //从  channel_ 中查找 target 对应的 SharedGrpcChannelPtr

void MultiGrpcChannelCache::ListWorkers(std::vector<string>* workers) //列出 cache_ 中每个 host_ports_ 的所有 task 加入 workers， 格式 “/job:${job}/replica:0/task:${task}”

string MultiGrpcChannelCache::TranslateTask(const string& target) //先从 target_caches_ 中查找，找不到再遍历各个 cache 找到 target 中 task id  对应的  cache

SharedGrpcChannelPtr MultiGrpcChannelCache::FindChannelOnce(const string& target) // 遍历 cache_ 中所有的 cache 解析 target 中的 task，找到 task 对应的 "host:port"，调用 channel_func_("host_port")，目前就是调用 NewHostPortGrpcChannel 函数

void SparseGrpcChannelCache::ListWorkers(std::vector<string>* workers) //列出 channel_cache_ 对应的  GrpcWorker  “/job:${job}/replica:0/task:${task}”
string SparseGrpcChannelCache::TranslateTask(const string& target) //从 host_ports_ 中找到  target 中 task 对应的 "host:port"
SharedGrpcChannelPtr SparseGrpcChannelCache::FindChannelOnce(const string& target) //解析 target 中的 task，找到 task 对应的 "host:port"，调用 channel_func_("host_port")

### RPCState

RPCState::RPCState(GrpcCounter* counter, ::grpc::GenericStub* stub,
             ::grpc::CompletionQueue* cq, const ::grpc::string& method,
             const protobuf::Message& request, ResponseMessage* response,
             StatusCallback done, CallOptions* call_opts)

1. 初始化成员变量
2. 解析  request 到 request_buf_
3. 初始化  call_ 为  stub->Call(&context_, method, cq, this)

```
        if (remaining_callbacks_.fetch_sub(1) == 4) {
            call_->Write(request_buf_, this);
            call_->Read(&response_buf_, this);
            remaining_callbacks_.fetch_sub(2);
          call_->Finish(&status_, this);
        else
          用  response_buf_ 初始化 response_
          done_(s);
```


### GrpcRemoteWorker

void IssueRequest(const protobuf::Message* request, protobuf::Message* response,
        const ::grpc::string& method, StatusCallback done,
        CallOptions* call_opts = nullptr) //创建一个 RPCState 对象

void IssueRequest(const protobuf::Message* request, TensorResponse* response,
                    const ::grpc::string& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) //创建一个 RPCState

const char* Method(GrpcWorkerMethod id) //从 GrpcWorkerMethodName 找到 id 对应的方法名

如下方法创建一个 RPCState 对象，将消息发送到队列 cq 中

void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request, CreateWorkerSessionResponse* response, StatusCallback done)
void RegisterGraphAsync(const RegisterGraphRequest* request, RegisterGraphResponse* response, StatusCallback done)
void DeregisterGraphAsync(const DeregisterGraphRequest* request, DeregisterGraphResponse* response, StatusCallback done)
void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request, RunGraphResponse* response, StatusCallback done)
void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request, MutableRunGraphResponseWrapper* response, StatusCallback done)
void CleanupGraphAsync(const CleanupGraphRequest* request, CleanupGraphResponse* response, StatusCallback done)
void CleanupAllAsync(const CleanupAllRequest* request, CleanupAllResponse* response, StatusCallback done)
void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request, TensorResponse* response, StatusCallback done)
void LoggingAsync(const LoggingRequest* request, LoggingResponse* response, StatusCallback done)
void TracingAsync(const TracingRequest* request, TracingResponse* response, StatusCallback done)

