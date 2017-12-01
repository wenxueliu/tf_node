


```
                    DeviceMgr
WorkerInterface     GraphMgr
                    WorkerCacheInterface


                handler:Item
GraphMgr  ->
                Rendezvous

        graph_mgr
        handle
Item -> session                    Graph graph
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

## 创建一个图之后的事情

对图进行 partition

优化

常量展开，子表达式消除(common-subexpression elimination)

MemoryType

确保 g 中每个 Edge 的 MemoryType 是一样的， 对于不能兼容的 Edge，将 Edge 修改为 "e->src+e->src_ouput -> `_Send` -> `_Recv` -> e->dst+e->dst_input"
