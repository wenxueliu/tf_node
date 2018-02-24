
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
