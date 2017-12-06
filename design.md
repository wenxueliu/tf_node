
理清如下关系
Master
Executor
Worker
Session
Device
Graph
Tensor
Operation
Rendezvous
ExecutionUnit


struct WorkerEnv //对 Env, DeviceMgr, RendezvousMgrInterface, ThreadPool 的简单封装
  Env* env = nullptr;
  SessionMgr* session_mgr = nullptr;
  std::vector<Device*> local_devices;
  DeviceMgr* device_mgr = nullptr;
  RendezvousMgrInterface* rendezvous_mgr = nullptr;
  thread::ThreadPool* compute_pool = nullptr;

SessionMgr //管理 WorkerSession 的增删查
  const WorkerEnv* const worker_env_;
  WorkerSession legacy_session_;
  std::map<string, std::unique_ptr<WorkerSession>> sessions_; //session: new WorkerSession

struct WorkerSession // 对与 session 相关的 GraphMgr, DeviceMgr, WorkerEnv 的封装
  const string worker_name;
  const std::unique_ptr<WorkerCacheInterface> worker_cache;
  const std::unique_ptr<DeviceMgr> device_mgr; //new DeviceMgr(RenamedDevice::NewRenamedDevice(worker_env_.local_devices))
  const std::unique_ptr<GraphMgr> graph_mgr; //new GraphMgr(worker_env_, device_mgr);

WorkerCacheInterface //对  WorkerInterface 的增删查
WorkerInterface //对  Worker 的抽象，对 Graph 的注册，注销，查看

class GraphMgr //对 Graph 和 Executor, Rendezvous 关联起来
  const WorkerEnv* worker_env_;             // Not owned.
  DeviceMgr* device_mgr_;
  std::unordered_map<string, Item*> table_

struct Item
    string handle; //在 GraphMgr.table_ 中对应的 key
    FunctionLibraryDefinition* lib_def //new FunctionLibraryDefinition(OpRegistry::Global(), gdef.library());
    std::vector<ExecutionUnit> units;
    GraphMgr* graph_mgr; //指向所属的 GraphMgr

struct ExecutionUnit //对应到  Graph 进行  Partition 之后的一个 Partition
    Graph* graph = nullptr; //graph 的一个 subgraph
    Device* device = nullptr; //graph 的一个 subgraph 所在的 device
    Executor* root = nullptr; //ExecutorImpl
    FunctionLibraryRuntime* lib = nullptr;


GraphMgr

1. 管理注册到一个 TensorFlow worker 的所有 graph， 每个 Item 与一个 string 关联保存在 GraphMgr 的 table_ 中。
2. GraphDef 执行
2.1 通过 handler 找到 Item
2.2 通过 step_id 从 WorkerEnv->RendezvousMgrInterface  找到  Rendezvous
2.3 通过 Rendezvous  将 Tensor 发送出去，并接受应答的 Tensor (同步或异步)
3. 一个  session 中的 GraphDef 可以被多个 Graph 共享



```
               WorkerEnv
GrpcServer     MasterEnv


GrpcMasterService -> Master master_impl_
                     grpc::MasterService::AsyncService master_service_

               Env* env
MasterEnv  ->  WorkerCacheInterface interface
               OpRegistryInterface* ops

Master   ->    MasterSession

MasterSession  ->  WorkerSession

                    DeviceMgr
WorkerInterface ->  GraphMgr
                    WorkerCacheInterface


                handler:Item
GraphMgr  ->    DeviceMgr device_mgr_
                WorkerEnv worker_env_

        graph_mgr
        handle
Item -> session                         Graph graph
        repeat ExecutionUnit units ->   Device device
                                        Executor root    -> new ExecutorImpl(params, graph)
                                        FunctionLibraryRuntime lib

                LocalExecutorParams params;
ExecutorImpl    Graph* graph_;

                Env
                RendezvousMgrInterface   -> step_id:Rendezvous
WorkerEnv  ->   SessionMgr
                DeviceMgr
                ThreadPool

               Send|Recv -> Tensor
Rendezvous ->  WorkerSession
               DeferredCall
```

GraphMgr.ExecuteAsync ->  ThreadPool.Schedule -> Executor.RunAsync

SessionMgr 管理 WorkerSession
WorkerSession 包含  DeviceMgr, GraphMgr, WorkerCacheInterface


## 创建一个图之后的事情

对图进行 partition

优化

常量展开，子表达式消除(common-subexpression elimination)，函数内联

MemoryType

确保 g 中每个 Edge 的 MemoryType 是一样的， 对于不能兼容的 Edge，将 Edge 修改为 "e->src+e->src_ouput -> `_Send` -> `_Recv` -> e->dst+e->dst_input"

## 调用过程

1. 创建 Graph graph
2. 设置 SessionOptions options
3. 创建 Session session(通过 NewSession)
4. session->Create(graph)
5. session->Run()
6. session->Close()
