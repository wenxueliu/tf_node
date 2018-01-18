
目前实现包括 Grpc, MPI, Verbs 三种方式，后两种是对第一种的简单封装。

session_mgr ->  worker_session1  -> worker_cache1
            ->  worker_session2  -> worker_cache2
            ->  worker_session2  -> worker_cache3

worker_cache1  ->   grpc_channel - grpc_worker
                    grpc_channel - grpc_remote_worker
                    grpc_channel - grpc_remote_worker

worker_cache2  ->   grpc_channel - grpc_worker
                    grpc_channel - grpc_remote_worker
                    grpc_channel - grpc_remote_worker

worker_cache3  ->   grpc_channel - grpc_worker
                    grpc_channel - grpc_remote_worker     <-> CompletionQueue
                    grpc_channel - grpc_remote_worker     <->


每个 SessionMgr 下有多个 WorkerSession，每个 WorkerSession 与一个 GrpcWorkerCache 对应，
每个节点都默认会创建 GrpcWorker 用于和本节点进行消息传递，根据配置会创建
GrpcRemoteWorker 用于与远程节点进行消息传递。 与远程节点进行消息传递的时候
会共享同一个队列。

当初始化 GrpcWorkerCache 对象之后，

一方面，创建一个 "grpc_worker_cache" 线程从 completion_queue_ 中取数据调用  OnCompleted 方法。

另一方面，当没有远程节点建立连接的话，该队列始终是空的，当通过 CreateWorker 方法
创建于远程节点的连接之后，就可以给远程发送各种消息(可执行的方法参考GrpcWorkerMethod)，
要求其执行，执行的方式就是对 CreateWorker 返回的  worker 调用对应的方法。
如果 GetStatusAsync，CreateWorkerSessionAsync 等等。 要执行的方法会以消息
的形式发送到 CompletionQueue。而此时 grpc_worker_cache 线程就可以收到该
消息，就会调用对应的方法。

## 源代码

contrib/mpi/mpi_server_lib.cc
contrib/verbs/verbs_server_lib.cc
core/distributed_runtime/server_lib.h
core/distributed_runtime/server_lib.cc
core/distributed_runtime/rpc/grpc_server_lib.cc
core/distributed_runtime/rpc/grpc_tensorflow_server.cc

## 代码流

一种 "registration/factory-based"  机制

注册非常简单，就是定义一个 xxxRegistor 对象即可(通过默认构造函数注册)

ServerFactory -> GrpcServerFactory

ServerFactory -> MPIServerFactory

ServerFactory -> VerbsServerFactory

typedef std::unordered_map<string, ServerFactory*> ServerFactories;
static ServerFactories* factories = new ServerFactories;

目前已经注册包括

```
"GRPC_SERVER", new GrpcServerFactory()
"VERBS_SERVER", new VerbsServerFactory()
"MPI_SERVER", new MPIServerFactory()
```

## 系统启动流程分析

1. 注册

GrpcServerRegistrar()
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory())
        factories.insert({"GRPC_SERVER", new GrpcServerFactory()})

2. 获取并初始化

tensorflow::NewServer(server_def, out_server)
    ServerFactory::GetFactory(server_def, factory)
    factory->NewServer(server_def, out_server)
        //如果这里 server_def.protocol() == "grpc"
        GrpcServerFactory.NewServer(server_def, out_server)
            GrpcServer::Create(server_def, Env::Default, out_server)
                res = new GrpcServer(server_def, env == nullptr ? Env::Default() : env)
                ServiceInitFunction service_func = nullptr;
                res->Init(service_func, NewRpcRendezvousMgr)
                out_server = res
        //如果这里 server_def.protocol() == "grpc+mpi"
        MPIServerFactory.NewServer(server_def, out_server)
            MPIServer::Create(server_def, Env::Default(), out_server)
                res = new MPIServer(server_def, Env::Default()); //背后完全是 GrpcServer
                ServiceInitFunction service_func = nullptr;
                ret->Init(service_func, NewMPIRendezvousMgr);
                out_server = res;
        //如果这里 server_def.protocol() == "grpc+verbs"
        VerbsServerFactory.NewServer(server_def, out_server)
            VerbsServer::Create(server_def, Env::Default(), out_server)
                res = new VerbsServer(server_def, Env::Default()); //背后完全是 GrpcServer 加锁
                ServiceInitFunction service_func = SetNewVerbsService(&ret->verbs_service_, worker_env, builder)
                ret->Init(service_func, NewRdmaRendezvousMgr)
                out_server = res;

GrpcServer.Init(null, NewRpcRendezvousMgr) //初始化  WorkerService, MasterService
    name_prefix = "/job:${server_def_.job_name()}/replica:0/task:${server_def_.task_index()}"
    worker_env_.local_devices = DeviceFactory::AddDevices(options, name_prefix, devices) //详细参考 device.md
    worker_env_.device_mgr = DeviceMgr(worker_env_.local_devices)
    worker_env_.rendezvous_mgr = rendezvous_mgr_func == nullptr ? new RpcRendezvousMgr(&worker_env_) : rendezvous_mgr_func(&worker_env_);
    ::grpc::ServerBuilder builder;  TODO 具体成员 //有两个  CompletionQueue，一个给 worker, 一个给  master
    builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port), GetServerCredentials(server_def_), &bound_port_);
    builder.SetMaxMessageSize(std::numeric_limits<int32>::max());
    builder.SetOption(std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
    master_impl_ = CreateMaster(&master_env_)
        master_impl_ = Master(master_env_, 0.0) //注：没有开启 GC 线程
    master_service_ = NewGrpcMasterService(master_impl_.get(), config.operation_timeout_in_ms(), &builder)
        master_service_ = GrpcMasterService(master_impl_.get(), config.operation_timeout_in_ms(), &builder);
    worker_impl_ = NewGrpcWorker(&worker_env_);
        worker_impl_ = GrpcWorker(worker_env_)
    worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder).release();
        worker_service_ = new GrpcWorkerService(worker, builder)
    server_ = builder.BuildAndStart();
    WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
        channel_cache = NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()) //实际与每个 host:port 建立一个 ::grpc::CreateCustomChannel
            单机版 SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func))
            集群版 MultiGrpcChannelCache(caches)
        worker_cache = NewGrpcWorkerCacheWithLocalWorker(channel_cache, worker_impl_, name_prefix)
            worker_cache = GrpcWorkerCache(channel_cache, worker_impl_, name_prefix)
    worker_env_.session_mgr = SessionMgr(worker_env_, name_prefix, worker_cache, worker_cache);
        WorkerSession(name_prefix, worker_cache, worker_env_->device_mgr, GraphMgr(worker_env_, worker_env_->device_mgr))
    options = server_def_.default_session_config()
    worker_env_.compute_pool = ComputePool(options);
         worker_env_.compute_pool = thread::ThreadPool(Env::Default(), "Compute", options.inter_op_parallelism_threads);
    master_env_.ops = OpRegistry::Global();
    master_env_.worker_cache = worker_cache;
    master_env_.master_session_factory = { MasterSession(options, env, remote_devs, worker_cache, device_set, CreateNoOpStatsPublisher); }
    master_env_.worker_cache_factory = { WorkerCacheFactory(options, worker_cache); }
    LocalMaster::Register("grpc://localhost:"${port}, master_impl_.get(), options.operation_timeout_in_ms());

name_prefix = "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}

class GrpcServer : public ServerInterface
  ServerDef server_def_;
  Env* env_;                                // Env::Default()
  MasterEnv master_env_;                    //参见 MasterEnv
  std::unique_ptr<Master> master_impl_;     // CreateMaster(&master_env_)
  AsyncServiceInterface* master_service_;   // NewGrpcMasterService(master_impl_.get(), config.operation_timeout_in_ms(), &builder)
  std::unique_ptr<Thread> master_thread_;
  std::unique_ptr<GrpcWorker> worker_impl_; // NewGrpcWorker(&worker_env_);
  AsyncServiceInterface* worker_service_;   // NewGrpcWorkerService(worker_impl_.get(), &builder)
  std::unique_ptr<Thread> worker_thread_;
  std::unique_ptr<::grpc::Server> server_;  // ServerBuilder 创建， 给 worker_service_ 和  master_service_  提供 CompletionQueue

GrpcMasterService : AsyncServiceInterface
  Master* master_impl_ ;                             //负责处理各类消息，如 CreateSession 等
  grpc::ServerCompletionQueue cq_;                   //来自 builder
  grpc::MasterService::AsyncService master_service_; //将不同种类的消息加入队列

GrpcWorkerService : AsyncServiceInterface
  GrpcWorker* worker_ ;                              //负责处理各类消息，如 GetStatus 等
  grpc::ServerCompletionQueue cq_;                   //来自 builder
  grpc::WorkerService::AsyncService worker_service_; //将不同种类的消息加入队列

WorkerInterface //对  Worker 的抽象，对 Graph 的注册，注销，查看

Worker : WorkerInterface
  WorkerEnv* env_;                                   //
  PartialRunMgr partial_run_mgr_;                    // id 与 CancellationManager 映射关系
  CancellationManager* cancellation_manager_;

GrpcWorker : Worker

Master
  MasterEnv* env_ ;                                    // 参考 MasterEnv
  std::unordered_map<string, MasterSession*> sessions_ // 保存所有已经创建的 session

MasterSession
  SessionOptions session_opts_;
  const MasterEnv* env_;
  const string handle_;
  std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs_;
  const std::unique_ptr<WorkerCacheInterface> worker_cache_;
  std::unique_ptr<DeviceSet> devices_;
  StatsPublisherFactory stats_publisher_factory_;
  std::atomic<int64> partial_run_handle_counter_ = {0};
  std::unique_ptr<SimpleGraphExecutionState> execution_state_;
  typedef std::unordered_map<uint64, ReffedClientGraph*> RCGMap;
  RCGMap run_graphs_;
  RCGMap partial_run_graphs_;
  struct PerStepState
    bool collect_costs = false;
    bool collect_timeline = false;
    bool collect_rpcs = false;
    bool collect_partition_graphs = false;
  struct RunState
    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched
    ReffedClientGraph* rcg = nullptr;
    uint64 step_id;
    int64 count = 0;
    PerStepState pss;
    std::unique_ptr<ProfileHandler> ph;
    bool step_started = false;
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_;
  condition_variable num_running_is_zero_;
  std::unordered_map<uint64, int64> subgraph_execution_counts_;
  int64 next_node_id_ = 0;
  CancellationManager cancellation_manager_; //

WorkerEnv                                 //一个 woker 需要的相关依赖
  Env* env;                               // Env::Default()
  SessionMgr* session_mgr ;               // SessionMgr(worker_env_, name_prefix, worker_cache, WorkerCacheFactory(options, worker_cache))
  std::vector<Device*> local_devices;     // 所有已经注册的设备列表，默认只包含 CPU， 与 MasterEnv.local_devices  相同
  DeviceMgr* device_mgr;                  // DeviceMgr(worker_env_.local_devices)
  RendezvousMgrInterface* rendezvous_mgr; // RpcRendezvousMgr(&worker_env_) 接受队列中的请求
  thread::ThreadPool* compute_pool;       // ComputePool(sess_opts) ; Worker 执行请求线程池，master 为什么不需要?

SessionMgr //对 WorkerSession 的增删查
  WorkerEnv* const worker_env_;             // WorkerCacheFactory(worker_cache_factory_options(server_def_), worker_cache)
  WorkerSession legacy_session_;            // WorkerSession(name_prefix, worker_cache, worker_env_->device_mgr, GraphMgr(worker_env_, worker_env_->device_mgr))
  WorkerCacheFactory worker_cache_factory_; // WorkerCacheFactory(worker_cache_factory_options(server_def_), worker_cache)
  std::map<string, std::unique_ptr<WorkerSession>> sessions_; // key 为 TODO

WorkerSession
  string worker_name;                                 //"/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}
  std::unique_ptr<WorkerCacheInterface> worker_cache; //new WorkerFreeListCache(worker_cache)
  std::unique_ptr<DeviceMgr> device_mgr;              //当为 legacy_session_ 时， 为 worker_env->device_mgr；否则为将 worker_env_->local_devices  每个重命名为 RenamedDevice
  std::unique_ptr<GraphMgr> graph_mgr;                //new GraphMgr(worker_env, device_mgr)

WorkerCacheInterface //对  WorkerInterface 的增删查

GrpcWorkerCache : WorkerCacheInterface //管理 grpc::Channel 与 worker 的关系，默认有一个 local_worker_
  string local_target_;                       // 本地 target  格式 "/job:"${server_def.job_name()}"/replica:0/task:"${server_def.task_index()}
  WorkerInterface* const local_worker_;       // 本地 GrpcWorker(所有 WorkerCacheInterface 共享)
  GrpcChannelCache* channel_cache_;           // 单机版 SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func)); 集群版 MultiGrpcChannelCache
  ::grpc::CompletionQueue completion_queue_;  // 保存所有关与远程通信是的消息队列。对于本地 local_worker_ 的消息不在此队列中
  Thread* polling_thread_;                    // "grpc_worker_cache" 从 completion_queue_ 中取数据调用 OnCompleted 方法。

GrpcRemoteWorker
  SharedGrpcChannelPtr channel_; //对应的 grpc:Channel
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;  //对应的队列，所有 GrpcRemoteWorker 共享

struct MasterEnv
  Env* env ;                          // Env::Default()
  WorkerCacheInterface* worker_cache  // NewGrpcWorkerCacheWithLocalWorker(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()), worker_impl_.get(), name_prefix)
  const OpRegistryInterface* ops ;    // OpRegistry::Global();
  std::vector<Device*> local_devices; // 所有已经注册的设备列表，默认只包含 CPU
  function master_session_factory;    // MasterSession(options, env, remote_devs, worker_cache, device_set, CreateNoOpStatsPublisher);
  std::function worker_cache_factory; // NewGrpcWorkerCacheWithLocalWorker(NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()), worker_impl_.get(), name_prefix)


启动的线程

1. grpc_worker_cache :  从 completion_queue 中依次取出元素，调用对应的 OnCompleted 方法

2. TF_master_GC : 回收超时的  session，目前没有启用。

运行

out_server.Start()

3. "TF_master_service" : master_service_->HandleRPCsLoop()

支持的消息类型
* CreateSession
* ExtendSession
* PartialRunSetup
* RunStep
* CloseSession
* ListDevices
* Reset

4. "TF_worker_service" : worker_service_->HandleRPCsLoop()

支持的消息类型
* GetStatus
* CreateWorkerSession
* CleanupAll
* RegisterGraph
* DeregisterGraph
* RunGraph
* CleanupGraph
* Logging
* Tracing
* RecvTensorHandlerRaw

out_server.Join()

至此，框架已经启动完毕，下面就是 CreateSession, Run, Close 三部曲

## 处理流

### 一个 Worker 请求的处理流

发送端
    Call::EnqueueRequestForMethod(grpc_service, cq, method_id, ${handle_request_function})
        call(handle_request_function)
        grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request, &call->responder_, cq, cq, &call->request_received_tag_);
        其中 request_received_tag_ = {call, Tag::kRequestReceived};
        这样接收端的收到的都是 tag 就为 {call, RequestReceived} 类型的消息，就会调用 call.RequestReceived() 方法

接收端
    从 CompletionQueue 队列中取出 tag
    tag = {call, RequestReceived}
    tag->OnCompleted(${service}, ok);
        call.RequestReceived(service, ok) //由于发送端发送的是  kRequestReceived 类型的消息
            ${service}.${handle_request_function}(call) //这里的 handle_request_function 为  call 构造函数中的 handle_request_function
            其中 service 为 GrpcWorkerService

当发送端 handle_request_function 为 GrpcWorkerService::GetStatusHandler 时，接受端就调用 GetStatusHandler(call)，

具体的实现由 GrpcWorker 来处理，处理完之后调用 call.SendResponse() 发送应答，并将消息类型设置为 kResponseSent，
这样发送端收到的消息根据消息类型调用对应的处理函数。（目前收到应答不做任何处理）


### 一个 Master 请求的处理流

发送端
    Call::EnqueueRequest(grpc_service, cq, enqueue_function, ${handle_request_function}, supports_cancel)
    call(handle_request_function)
        (grpc_service->*enqueue_function)(&call->ctx_, &call->request, &call->responder_, cq, cq, &call->request_received_tag_);
            grpc::MasterService::AsyncService::RequestXXXXX(&call->ctx_, &call->request, &call->responder_, cq, cq, &call->request_received_tag_)
                ::grpc::Service::RequestAsyncUnary(0, &call->ctx_, &call->request, &call->responder_, cq, cq, &call->request_received_tag_);

其中 request_received_tag_ = {call, Tag::kRequestReceived}; XXXX 为 MasterService 支持的消息类型
这样接收端的收到的都是 tag 就为 {call, RequestReceived} 类型的消息，就会调用 call.RequestReceived() 方法

接收端
    从 CompletionQueue 队列中取出 tag
    tag = {call, RequestReceived}
    tag->OnCompleted(${service}, ok);
        call.RequestReceived(service, ok) //由于发送端发送的是  kRequestReceived 类型的消息
            ${service}.${handle_request_function}(call) //这里的 handle_request_function 为  call 构造函数中的 handle_request_function
            其中 service 为 GrpcMasterService

当发送端 handle_request_function 为 GrpcMasterService::CreateSessionHandler 时，就调用 CreateSessionHandler(call)

此外，GrpcMasterService 和 GrpcWorkerService 的 xxxxHandler 方法实际实现者为 Master 和 GrpcWorker


## 调用过程

1. 创建 Graph graph
2. 设置 SessionOptions options
3. 创建 Session session(通过 NewSession)
4. session = NewSession(options)
5. session->Create(graph)
6. session->Run()
7. session->Close()

遵循 registration/factory-based 模式，所有的  session 都保存在全局变量 factories, 
通过 Register 注册一个 SessionFactory. 目前已经注册的包含

```
    DIRECT_SESSION : new DirectSessionFactory()
    GRPC_SESSION : new GrpcSessionFactory()
```

NewSession(options)
    //options.target 以 “grpc://”
    GrpcSessionFactory.NewSession(options) //返回  GrpcSession
        std::unique_ptr<GrpcSession> ret;
        GrpcSession::Create(options, &ret)  //grpc_session.h
            std::unique_ptr<GrpcSession> session(new GrpcSession(options));
            master_channel = ::grpc::CreateCustomChannel("dns:///" + options.target, ::grpc::InsecureChannelCredentials(), args)
            master = NewGrpcMaster(master_channel)  //grpc_remote_master.h
                master = GrpcRemoteMaster(master_channel)
                    master.stub_ = grpc::MasterService::NewStub(master_channel) //grpc_master_service_impl.h
                        master.stub_ = grpc::MasterService::Stub(master_channel)
            session->SetRemoteMaster(std::move(master));
                session._master = master;
            session.options_ = options
            ret = session
    //options.target 以 “” 开始
    DirectSession.NewSession(options)

对于 GrpcSession，后续所有的操作都委托给 GrpcRemoteMaster，而 GrpcRemoteMaster
由将所有的操作委托给 grpc::MasterService::Stub, 最后交由 grpcMasterService_method_names 定义的方法实现


session.Create(graph)
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  CreateImpl(&call_options, graph);
    master_->CreateSession(call_options, &req, &resp); //master_ = GrpcRemoteMaster
        ::grpc::ClientContext ctx;
        ctx.set_fail_fast(false);
        SetDeadline(&ctx, call_options->GetTimeout());
        master_.stub->CreateSession(ctx, &req, &resp) //master_.stub = MasterService::Stub(master_channel)
            ::grpc::internal::BlockingUnaryCall(master_channel, rpcmethod_CreateSession_, ctx, req, resp);
    handle_ = resp.mutable_session_handle(
    current_graph_version_ = resp.graph_version();

session.Create(run_options, graph)
    master_->CreateSession(run_options, &req, &resp); //master_ = GrpcRemoteMaster
        ::grpc::ClientContext ctx;
        ctx.set_fail_fast(false);
        SetDeadline(&ctx, call_options->GetTimeout());
        master_.stub->CreateSession(ctx, &req, &resp)
            ::grpc::internal::BlockingUnaryCall(master_channel, rpcmethod_CreateSession_, ctx, req, resp);
    handle_ = resp.mutable_session_handle(
    current_graph_version_ = resp.graph_version();




## 数据结构

message JobDef
  string name = 1; //job name
  map<int32, string> tasks = 2; //task id : host:port

message ClusterDef
  repeated JobDef job = 1;

message ServerDef
  ClusterDef cluster = 1;
  string job_name = 2;
  int32 task_index = 3;
  ConfigProto default_session_config = 4;
  string protocol = 5; //硬代码 grpc

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

message RPCOptions {
  bool use_rpc_for_inprocess_master = 1; //true 用 GrpcMaster，false 用 LocalMaster

message GPUOptions
  double per_process_gpu_memory_fraction = 1;
  string allocator_type = 2;
  int64 deferred_deletion_bytes = 3;
  bool allow_growth = 4;
  string visible_device_list = 5; //GPU 列表当与  ConfigProto.device_count 中 GPU 对应的数字不一致时，以 visible_device_list 中个数为准
  int32 polling_active_delay_usecs = 6;
  int32 polling_inactive_delay_msecs = 7;
  bool force_gpu_compatible = 8;

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

## Server

static ServerFactories* factories = new ServerFactories; //保存所有注册的服务

class ServerInterface
  virtual Status Start() = 0;
  virtual Status Stop() = 0;
  virtual Status Join() = 0;
  virtual const string target() const = 0;

class ServerFactory
  virtual Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) = 0;
  virtual bool AcceptsOptions(const ServerDef& server_def) = 0;
  static void Register(const string& server_type, ServerFactory* factory);
  static Status GetFactory(const ServerDef& server_def, ServerFactory** out_factory);

Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server);

### Grpc Server

static GrpcServerRegistrar registrar; //这里会调用默认构造函数注册

class GrpcServerRegistrar
  GrpcServerRegistrar()
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory()); //注册 GrpcServerFactory 到 factories

class GrpcServerFactory : public ServerFactory

  bool AcceptsOptions(const ServerDef& server_def)
    return server_def.protocol() == "grpc";

  Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server)
    return GrpcServer::Create(server_def, Env::Default(), out_server);

class GrpcServer : public ServerInterface
  const ServerDef server_def_;
  Env* env_;
  int bound_port_ = 0;
  mutex mu_;
  enum State { NEW, STARTED, STOPPED };
  State state_ GUARDED_BY(mu_);
  // Implementation of a TensorFlow master, and RPC polling thread.
  MasterEnv master_env_;
  std::unique_ptr<Master> master_impl_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::unique_ptr<Thread> master_thread_;
  // Implementation of a TensorFlow worker, and RPC polling thread.
  WorkerEnv worker_env_;
  std::unique_ptr<GrpcWorker> worker_impl_;
  AsyncServiceInterface* worker_service_ = nullptr;
  std::unique_ptr<Thread> worker_thread_ GUARDED_BY(mu_);
  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);

### MPI Server

static MPIServerRegistrar registrar; //这里会调用默认构造函数注册

class MPIServerRegistrar

class MPIServerFactory : public ServerFactory

class MPIServer : public GrpcServer

### VerbsServer

static VerbsServerRegistrar registrar; //这里会调用默认构造函数注册

class VerbsServerRegistrar

class VerbsServerFactory : public ServerFactory

class VerbsServer : public GrpcServer
  RdmaMgr* rdma_mgr_;
  mutex mu_;
  enum State { DISCONNECTED, CONNECTED };
  State verbs_state_ GUARDED_BY(mu_);
  GrpcVerbsService* verbs_service_ = nullptr;
  std::unique_ptr<Thread> verbs_thread_ GUARDED_BY(mu_);
  GrpcChannelCache* channel_cache_ = nullptr;


## 例子

1. A single-process cluster, containing "/job:local/task:0".

   Cluster:
     job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }

   Server:
     cluster { $CLUSTER } job_name: 'local' task_index: 0

2. A two-process cluster, containing "/job:local/task:{0,1}".

   Cluster:
     job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
                         tasks { key: 1 value: 'localhost:2223' } }

   Servers:
     cluster { $CLUSTER } job_name: 'local' task_index: 0
     cluster { $CLUSTER } job_name: 'local' task_index: 1

3. A two-job cluster, containing "/job:worker/task:{0,1,2}" and
   "/job:ps/task:{0,1}".

   Cluster:
     job { name: 'worker' tasks { key: 0 value: 'worker1:2222' }
                          tasks { key: 1 value: 'worker2:2222' }
                          tasks { key: 2 value: 'worker3:2222' } }
     job { name: 'ps'     tasks { key: 0 value: 'ps0:2222' }
                          tasks { key: 1 value: 'ps1:2222' } }

   Servers:
     cluster { $CLUSTER } job_name: 'worker' task_index: 0
     cluster { $CLUSTER } job_name: 'worker' task_index: 1
     cluster { $CLUSTER } job_name: 'worker' task_index: 2
     cluster { $CLUSTER } job_name: 'ps'     task_index: 0
     cluster { $CLUSTER } job_name: 'ps'     task_index: 1

/job:worker/task:0 分配给  worker1:2222
/job:worker/task:1 分配给  worker1:2222
/job:worker/task:2 分配给  worker1:2222

/job:ps/task:0 分配给  ps0:2222
/job:ps/task:1 分配给  ps1:2222

tensorflow --cluster_spec=SPEC --job_name=NAME --task_id=ID

```
SPEC: <JOB>(,<JOB>)*
JOB : <NAME>|<HOST:PORT>(;<HOST:PORT>)*
HOST : IP 或 hostname
PORT : 端口号
```
如: worker1|10.10.10.11:2222;10.10.11.11:2222,worker2|10.20.10.11:2222;10.20.11.11:2222

tensorflow --cluster_spec=worker1|10.10.10.11:2222;10.10.11.11:2222,worker2|10.20.10.11:2222;10.20.11.11:2222 --job_name=worker1 --task_id=0

tensorflow --cluster_spec=worker1|10.10.10.11:2222;10.10.11.11:2222,worker2|10.20.10.11:2222;10.20.11.11:2222 --job_name=worker1 --task_id=1

--tf_job=localhost --tf_task=1 --num_cpus=1 --num_gpus=1

## 源码分析

Status FillServerDef(const string& cluster_spec, const string& job_name, int task_index, ServerDef* options)

解析 cluster_spec, job_name, task_index 初始化 options，参考下面例子

tensorflow --cluster_spec=worker1|10.10.10.11:2222;10.10.11.11:2222,worker2|10.20.10.11:2222;10.20.11.11:2222 --job_name=worker1 --task_id=0

ServerDef
   job_name : worker1
   task_index : 0

ClusterDef
   JobDef
       name : worker1
       tasks : {0, 10.10.10.11:2222} {1, 10.10.11.11:2222}
   JobDef
       name : worker1
       tasks : {0, 10.20.10.11:2222} {1, 10.20.11.11:2222}


int main(int argc, char* argv[])

```
  tensorflow::NewServer(server_def, &server);
  server->Start();
  server->Join();
```

### ServerFactory

ServerFactories* server_factories() //返回  factories

void ServerFactory::Register(const string& server_type, ServerFactory* factory) //{server_type, factory} 加入 factories

Status ServerFactory::GetFactory(const ServerDef& server_def, ServerFactory** out_factory) // 从  factories 中查找  server_def 对应的 ServerFactory 保存咋  out_factory

Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server)

```
  ServerFactory::GetFactory(server_def, &factory);
  return factory->NewServer(server_def, out_server);
```

### MPIServer

MPIServer::MPIServer(const ServerDef& server_def, Env* env) //GrpcServer(server_def, env)

Status MPIServer::Init(ServiceInitFunction service_func, RendezvousMgrCreationFunction rendezvous_mgr_func) //GrpcServer::Init(service_func, rendezvous_mgr_func);

Status MPIServer::Start() //GrpcServer::Init(service_func, rendezvous_mgr_func);
Status MPIServer::Join() //GrpcServer::Join();
Status MPIServer::Create(const ServerDef& server_def, Env* env, std::unique_ptr<ServerInterface>* out_server)

```cpp
  std::unique_ptr<MPIServer> ret(new MPIServer(server_def, Env::Default()));
  ServiceInitFunction service_func = nullptr;
  TF_RETURN_IF_ERROR(ret->Init(service_func, NewMPIRendezvousMgr));
  *out_server = std::move(ret);
```

### MPIServerFactory

bool AcceptsOptions(const ServerDef& server_def) //server_def.protocol() == "grpc+mpi"

Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) //MPIServer::Create(server_def, Env::Default(), out_server);

### VerbsServer

VerbsServer::VerbsServer(const ServerDef& server_def, Env* env) //GrpcServer(server_def, env), verbs_state_(DISCONNECTED) {}

Status VerbsServer::ChannelCacheFactory(const ServerDef& server_def, GrpcChannelCache** channel_cache)

1. 将 server_def.cluster_def.job() 的每一个 job 解析后加入 channel_spec(GrpcChannelSpec);
2. channel_cache = NewGrpcChannelCache(channel_spec, GetChannelCreationFunction());
2.1. channel_spec.host_ports_jobs()  每一个元素创建一个 cache.push_back(new SparseGrpcChannelCache(job.job_id, job.host_ports, channel_func)));
2.2  new MultiGrpcChannelCache(caches)

Status VerbsServer::Init(ServiceInitFunction service_func, RendezvousMgrCreationFunction rendezvous_mgr_func)

```
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  ChannelCacheFactory(server_def(), &channel_cache_);
  rdma_mgr_ = new RdmaMgr(worker_env(), channel_cache_);
  verbs_service_->SetRdmaMgr(rdma_mgr_);
  dynamic_cast<RdmaRendezvousMgr*>(worker_env()->rendezvous_mgr)->SetRdmaMgr(rdma_mgr_);
```

Status VerbsServer::Start()

```
  Status s = GrpcServer::Start();
    mutex_lock l(mu_);
    if (verbs_state_ == DISCONNECTED)
      // verbs_thread needs to be initiated
      // before rdma_mgr sets up the rdma channels.
      verbs_thread_.reset(worker_env()->env->StartThread(
          ThreadOptions(), "TF_verbs_service",
          [this] { verbs_service_->HandleRPCsLoop(); }));
      rdma_mgr_->SetupChannels();
      verbs_state_ = CONNECTED;
```

Status VerbsServer::Join() {

```
  Status s = GrpcServer::Join();
    mutex_lock l(mu_);
    if (verbs_state_ == CONNECTED)
      verbs_state_ = DISCONNECTED;
      verbs_thread_.reset();
```

Status VerbsServer::Create(const ServerDef& server_def, Env* env, std::unique_ptr<ServerInterface>* out_server)

```
  std::unique_ptr<VerbsServer> ret(new VerbsServer(server_def, Env::Default()));
  ServiceInitFunction service_func = [&ret](const WorkerEnv* worker_env,
                                            ::grpc::ServerBuilder* builder) {
    return SetNewVerbsService(&ret->verbs_service_, worker_env, builder);
  };
  TF_RETURN_IF_ERROR(ret->Init(service_func, NewRdmaRendezvousMgr));
  *out_server = std::move(ret);
```


### VerbsServerFactory

bool AcceptsOptions(const ServerDef& server_def) //return server_def.protocol() == "grpc+verbs";

Status NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) // return VerbsServer::Create(server_def, Env::Default(), out_server);
