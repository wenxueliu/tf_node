
// Master implements the service MasterSerivce.
//
// A Master maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A Master knows ahead of time local devices available as
// client devices.
//
// A Master discovers remote devices on-demand and keeps track of
// statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on the workers.

1. TF_master_GC  每隔 10 s 回收超时的 session，如果某个 session 在 session_gc_seconds_ 没有活跃，就被删掉

## 数据结构

static LocalMasterRegistry* local_master_registry_ = new LocalMasterRegistry;


struct WorkerCacheFactoryOptions
  const ClusterDef* cluster_def = nullptr; //server_def.cluster(); req->config().cluster_def();
  const string* job_name = nullptr; //server_def.job_name()
  int task_index; //server_def.task_index(
  const string* protocol = nullptr; //server_def.protocol()  //grpc

class WorkerCacheInterface
  virtual void ListWorkers(std::vector<string>* workers) const = 0;
  virtual WorkerInterface* CreateWorker(const string& target) = 0;
  virtual void ReleaseWorker(const string& target, WorkerInterface* worker) {
  virtual bool GetDeviceLocalityNonBlocking(const string& device, DeviceLocality* locality) = 0;
  virtual void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality, StatusCallback done) = 0;
  virtual void SetLogging(bool active) {}
  virtual void ClearLogs() {}
  virtual bool RetrieveLogs(int64 step_id, StepStats* ss) { return false; }


struct MasterEnv
  Env* env = nullptr;
  WorkerCacheInterface* worker_cache = nullptr;
  const OpRegistryInterface* ops = nullptr;
  std::vector<Device*> local_devices;
  std::function<MasterSession*(SessionOptions, MasterEnv*, std::unique_ptr<std::vector<std::unique_ptr<Device>>>,
      std::unique_ptr<WorkerCacheInterface>, std::unique_ptr<DeviceSet> device_set)>
      master_session_factory;
  std::function<Status(const WorkerCacheFactoryOptions&, WorkerCacheInterface**)> worker_cache_factory;

class Master
  typedef std::function<void(const Status&)> MyClosure;
  typedef Master ME;
  MasterEnv* env_ = nullptr;
  mutex mu_;
  condition_variable shutdown_cv_;
  bool shutdown_ GUARDED_BY(mu_) = false;
  Thread* gc_thread_;
  std::unordered_map<string, MasterSession*> sessions_ GUARDED_BY(mu_); // Maps session handles to sessions.
  MovingAverage last_1000_steps_ GUARDED_BY(mu_); // Moving average of step times.
  int64 step_count_ GUARDED_BY(mu_); // Cumulative number of steps executed.
  const double session_gc_seconds_; // If a session is not active for this many seconds, it will be closed automatically.

class DeviceFinder
  typedef DeviceFinder ME;
  const MasterEnv* env_;
  WorkerCacheInterface* worker_cache_;
  std::vector<DeviceNameUtils::ParsedName> filters_; //保存检验合法的  device 名称
  mutex mu_;
  int num_pending_ GUARDED_BY(mu_); //targets_  还没有找到的设备数
  condition_variable pending_zero_; //当 num_pending_ 为 0 时， 表明  targets_ 所有设备都已经找到了，唤醒阻塞在 Wait() 的线程
  std::vector<Device*> found_ GUARDED_BY(mu_); //已经找到的所有远程设备的集合
  std::vector<string> targets_; //如果 filters_ 不为空，必须与 filters_ 中的名称相同
  std::vector<bool> seen_targets_ GUARDED_BY(mu_); //在 targets_ 中同样索引的设备找到，设置为 true
  Status status_;


typedef std::unordered_map<string, MasterInfo> LocalMasterRegistry;

class LocalMaster : public MasterInterface
  Master* master_impl_;  // Not owned.
  const int64 default_timeout_in_ms_;

struct MasterInfo
  Master* master;
  const int64 default_timeout_in_ms;



## 源码分析


### DeviceFinder

static Status GetRemoteDevices( const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache,
      std::vector<std::unique_ptr<Device>>* out_remote)

static void GetRemoteWorkers(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache, std::vector<string>* workers)

```
    DeviceFinder finder(device_filters, env, worker_cache);
    *workers = finder.targets_;
```

void Start() //遍历 targets_， 对每一元素创建 remote_device 对象
Status Wait() //等待 targets_.size() 变为 0

void GetRemoteDevices(const std::vector<Device*>& local, std::vector<std::unique_ptr<Device>>* remote)

将 found_ 中的设备不在 local 中，且在 filters_ 的设备加入 remote

void WhenFound(int target_index, const Status& s, std::vector<Device*>* devices)

如果 target_index 设置 seen_targets_[target_index] 为 true, 表示对应的设备找到找到, 将其加入 found_，num_pending_ 减一

如果 targets_ 中对应的设备都找到了，唤醒阻塞的进程

bool Intersects(const DeviceNameUtils::ParsedName& x, const DeviceNameUtils::ParsedName& y)

x 与 y 的 job, replica, task, type, id 都相同

bool MatchFilters(const string& name) //name 是否在 filters_ 中

### Master

void Master::GC()

从 sessions_ 中找到没有活跃时间超过  session_gc_seconds_ 的  session, 调用该 session 的 GarbageCollect，之后从  sessions_ 中删除

void Master::CreateSession(const CreateSessionRequest* req, CreateSessionResponse* resp, MyClosure done)
TODO

void Master::ExtendSession(const ExtendSessionRequest* req, ExtendSessionResponse* resp, MyClosure done)
TODO

void Master::PartialRunSetup(const PartialRunSetupRequest* req, PartialRunSetupResponse* resp, MyClosure done)
TODO

void Master::RunStep(CallOptions* opts, const RunStepRequestWrapper* req, MutableRunStepResponseWrapper* resp, MyClosure done)

void Master::CloseSession(const CloseSessionRequest* req, CloseSessionResponse* resp, MyClosure done)

void Master::ListDevices(const ListDevicesRequest* req, ListDevicesResponse* resp, MyClosure done)

void Master::CleanupWorkers(const ResetRequest& reset)

void Master::Reset(const ResetRequest* req, ResetResponse* resp, MyClosure done)
### LocalMaster

只是对 Master 的简单封装，增加了超时机制

LocalMasterRegistry* local_master_registry() // 返回 local_master_registry_


void LocalMaster::Register(const string& target, Master* master, int64 default_timeout_in_ms) //local_master_registry_ 插入 {target, MasterInfo(master, default_timeout_in_ms)});

std::unique_ptr<LocalMaster> LocalMaster::Lookup(const string& target) //将 local_master_registry_ 中 target 对应的  MasterInfo 转化为 LocalMaster

Status LocalMaster::CreateSession(CallOptions* call_options, const CreateSessionRequest* request, CreateSessionResponse* response) //master_impl_->CreateSession(request, response,)

Status LocalMaster::ExtendSession(CallOptions* call_options, const ExtendSessionRequest* request, ExtendSessionResponse* response) //master_impl_->ExtendSession(request, response,)

Status LocalMaster::PartialRunSetup(CallOptions* call_options, const PartialRunSetupRequest* request, PartialRunSetupResponse* response) //master_impl_->PartialRunSetup(request, response,)

Status LocalMaster::RunStep(CallOptions* call_options, RunStepRequestWrapper* request, MutableRunStepResponseWrapper* response) //master_impl_->RunStep(request, response,)

MutableRunStepRequestWrapper* LocalMaster::CreateRunStepRequest() //return new InMemoryRunStepRequest;

MutableRunStepResponseWrapper* LocalMaster::CreateRunStepResponse() //return new InMemoryRunStepResponse;

Status LocalMaster::CloseSession(CallOptions* call_options, const CloseSessionRequest* request, CloseSessionResponse* response) //master_impl_->CloseSession(request, response,)

Status LocalMaster::ListDevices(CallOptions* call_options, const ListDevicesRequest* request, ListDevicesResponse* response) // master_impl_->ListDevices(request, response,)

Status LocalMaster::Reset(CallOptions* call_options, const ResetRequest* request, ResetResponse* response) // master_impl_->Reset(request, response,)
