

replicas : 使用多机训练时， 一台机器对应一个 replica ——复本
worker : 功能类比于单机多卡中的GPU。
job : 一个 job 中包含多个 task
ps  : 参数服务器，多机训练时计算梯度平均值并执行backward操作的参数服务器，功能类比于单机多GPU（也叫单机多卡）时的CPU。（未考证， TODO）
chief : 指 master
tower：使用多GPU训练时， 一个GPU上对应一个tower。
clone: 由于tensorflow里多GPU训练一般是每个GPU上都有完整的模型，各自forward，得到的梯度交给CPU平均然后统一backward，每个GPU上的模型也叫做一个clone。所以clone与tower指的是同一个东西。




graph, Session, Server, job, task, cluster, clone, tower, ps, worker 之间的关系

一个 cluster 下有多个 job，一个 job 只能属于一个 cluster
一个 job 下有多个 task，一个 task 只能属于一个 job ?
一个 task 对应一个 Server, 一个 server 只能属于一个 task
一个 Server 可以运行多个 Session， 一个 Session 对一个 Server


``` python
    cluster_def = tf.train.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    }).as_cluster_def()
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def,
        job_name="worker",
        task_index=2,
        protocol="grpc")
    server = tf.train.Server(server_def, protocol='grpc',
        job_name='', task_index = 0,
        config=config, start=False)
```


## Server

将 tensorflow_server_pb2.ServerDef 初始化 self._server_def 或
从 dict, ClusterSpec, cluster_pb2.ClusterDef 中找到对应的 self._server_def

def \_make_server_def(server_or_cluster_def, job_name, task_index, protocol, config)

将 tensorflow_server_pb2.ServerDef 初始化 self._server_def 或
从 dict, ClusterSpec, cluster_pb2.ClusterDef 中找到 job_name, task_index, protocol 对应的 self._server_def

def start(self) : 开始运行服务
def join(self) : 等待服务运行完成
def server_def(self) : 返回  self._server_def
def target(self) : 返回目标地址
def create_local_server(config=None, start=True) : 创建一个本地的的 Server

## ClusterSpec

基本概念：

job  : 一个任务
task : 一个 job 下可以包含多个 task，每个 task 是一个 ip:port 的地址，每个 task 对应到一个 Server

因此，一个集群可以有个多个 job, 每个  job 下可以有多个 task, 每个 task 分配给一个 Server

将  dict, ClusterSpec, cluster_pb2.ClusterDef  转为为字典，保存在
self._cluster_def 中，其中 key 为 job_name, value 为 job(类型为字典，
key 为 task index, value 为 task)

self._cluster_def: dict
    key : str, 任务名称
    value : dict  任务
        key : int, 任务索引
        value : str, 具体任务，ip:port


def jobs(self) : self._cluster_def 的 key 数量
def num_tasks(self, job_name): job_name 对应的 task 的数量
def task_indices(self, job_name) : job_name 对应 task 的索引列表
def task_address(self, job_name, task_index): job_name 对应 job 的第 task_index 个 task
def job_tasks(self, job_name) : job_name 对应的 job 中的所有 task 列表
def \_make_cluster_def(self) : 将 self._cluster_def 转为  self._cluster_def

例子

```python
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                             "worker1.example.com:2222",
                                             "worker2.example.com:2222"],
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
```
