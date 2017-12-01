
总共有三种设备类型

* CPU
* GPU
* SYCL

各种设备的优先级

* GPU: 210
* SYCL: 200
* GPUCompatibleCPU: 70
* ThreadPoolDevice: 60
* Default: 50


所有的 Device 放在  DeviceSet 中，

DeviceSpec

  * Job: The job name.
  * Replica: The replica index.
  * Task: The task index.
  * Device type: The device type string (e.g. "CPU" or "GPU").
  * Device index: The device index.

full name

`/job:<name>/replica:<id>/task:<id>/device:CPU:<id>`
`/job:<name>/replica:<id>/task:<id>/device:GPU:<id>`

legacy name
`/job:<name>/replica:<id>/task:<id>/CPU:<id>`
`/job:<name>/replica:<id>/task:<id>/GPU:<id>`

源文件

core/common_runtime/device.h
core/common_runtime/device.cc
core/common_runtime/local_device.h
core/common_runtime/local_device.cc
core/common_runtime/renamed_device.h
core/common_runtime/renamed_device.cc
compiler/jit/xla_device.h
compiler/tf2xla/xla_compilation_device.h
core/common_runtime/sycl/sycl_device.h
core/common_runtime/threadpool_device.h
core/common_runtime/device_set.h
core/common_runtime/device_set.cc
core/common_runtime/device_factory.h
core/common_runtime/device_factory.cc
core/common_runtime/device_mgr.h
core/common_runtime/device_mgr.cc
compiler/jit/xla_cpu_device.cc
compiler/jit/xla_gpu_device.cc
compiler/plugin/executor/device.cc
core/common_runtime/gpu/gpu_device.h
core/common_runtime/gpu/gpu_device_factory.cc
core/common_runtime/sycl/sycl_device_factory.cc
core/common_runtime/threadpool_device_factory.cc

## 数据结构

DeviceAttributes -> Device

DeviceBase -> Device

struct FactoryItem
  std::unique_ptr<DeviceFactory> factory;
  int priority;

static std::unordered_map<string, FactoryItem>* factories = new std::unordered_map<string, FactoryItem>; //key 为  devict_type

message DeviceLocality
  // Optional bus locality of device.  Default value of 0 means
  // no specific locality.  Specific localities are indexed from 1.
  int32 bus_id = 1;

message DeviceAttributes
  string name = 1;  //`job:<name>/replica:<id>/task:<id>/device:CPU:<id>`
  string device_type = 2; //cpu, gpu
  int64 memory_limit = 4;
  DeviceLocality locality = 5; //为了加速访问平台具体的属性
  fixed64 incarnation = 6; // 全局唯一的数字, 不能为 0
  string physical_device_desc = 7; //

class DeviceBase
  Env* const env_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr; //线程
  GpuDeviceInfo* gpu_device_info_ ; //GPU
  Eigen::ThreadPoolDevice* eigen_cpu_device_ ; //CPU
  Eigen::SyclDevice* eigen_sycl_device_ ; //OpenCL

  struct GpuDeviceInfo
    // Make sure all the defaults are NULL, so we can spot missing assignments.
    perftools::gputools::Stream* stream = nullptr;
    DeviceContext* default_context = nullptr;
    EventMgr* event_mgr = nullptr;
    int gpu_id = -1;

  struct CpuWorkerThreads
    int num_threads = 0; //线程数
    thread::ThreadPool* workers = nullptr; //线程池


class Device : public DeviceBase //描述一个设备, 抽象类
  const DeviceAttributes device_attributes_;
  DeviceNameUtils::ParsedName parsed_name_;
  OpSegment op_seg_;
  ResourceMgr* rmgr_ = nullptr; //new ResourceMgr(parsed_name_.job);

class LocalDevice : public Device
  static bool use_global_threadpool_; //默认 true
  struct EigenThreadPoolInfo;
      DeviceBase::CpuWorkerThreads eigen_worker_threads_; //options.config.intra_op_parallelism_threads() or 系统 cpu 核心数
      std::unique_ptr<Eigen::ThreadPoolInterface> eigen_threadpool_wrapper_; //new EigenThreadPoolWrapper(eigen_worker_threads_.workers)
      std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;//new Eigen::ThreadPoolDevice(eigen_threadpool_wrapper_.get(), eigen_worker_threads_.num_threads)
  std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_; //new LocalDevice::EigenThreadPoolInfo(options)

class XlaDevice : public LocalDevice
  const int device_ordinal_;
  // The name of the device that is used to compile Ops for this XlaDevice.
  const DeviceType& jit_device_name_;
  Allocator* xla_allocator_;
  ::perftools::gputools::Platform* platform_;

class XlaCompilationDevice : public LocalDevice
  std::unique_ptr<XlaCompilationAllocator> allocator_;


class BaseGPUDevice : public LocalDevice
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned
  gpu::StreamExecutor* executor_;  // not owned
  struct StreamGroup
    gpu::Stream* compute = nullptr;
    gpu::Stream* host_to_device = nullptr;
    gpu::Stream* device_to_host = nullptr;
    gpu::Stream* device_to_device = nullptr;
  class StreamGroupFactory;
  gtl::InlinedVector<StreamGroup*, 4> streams_;
  gtl::InlinedVector<char*, 4> scratch_;
  std::vector<GPUDeviceContext*> device_contexts_;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  mutex trace_mu_;
  int gpu_id_ = -1;
  const bool sync_every_op_ = false;
  const int32 max_streams_;
  std::unique_ptr<EventMgr> em_;

class SYCLDevice : public LocalDevice
  Allocator         cpu_allocator_;           // not owned
  SYCLAllocator     sycl_allocator_;          // not owned
  SYCLDeviceContext device_context_;

class ThreadPoolDevice : public LocalDevic
  Allocator* allocator_;  // Not owned

class RenamedDevice : public Device //目的仅仅是改变名称
  Device underlying_;
  bool owns_underlying_;

class RemoteDevice : public Device
  const string local_dev_name_;

class DeviceSimple : public DeviceBase

class DeviceMgr
  typedef gtl::InlinedVector<Device*, 8> DeviceVec;
  DeviceVec devices_; //所有 Device
  std::unordered_map<StringPiece, Device*, StringPiece::Hasher> device_map_; // 每个 Device 保存该 Device 的三组映射关系(full name, local name, canonical name )
  core::Arena name_backing_store_;  //默认 128
  std::unordered_map<string, int> device_type_counts_; //不同类型的 Device 的数量

  const char* data_;
  size_t size_;

class DeviceSet //可以遍历所有的 Device，也可以 根据名字查找一个 Device
  std::vector<Device*> devices_; //保存所有的设备
  //full name : job:${job}/replica:${replica}/task:${task}/${device_type}:${id}
  //legacy name: job:${job}/replica:${replica}/task:${task}/device:${device_type}:${id}
  std::unordered_map<string, Device*> device_by_name_; // 这里的每个设备都有两个 key(full, legacy)
  Device* client_device_ = nullptr; //

class DeviceFactory //创建一个设备，同一类型的设备可能存在一个，新注册的设备如果要生效必须优先级比已经存在的高。
  static void Register(const string& device_type, DeviceFactory* factory, int priority);
  static DeviceFactory* GetFactory(const string& device_type);
  static Status AddDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices);
  static Device* NewDevice(const string& type, const SessionOptions& options, const string& name_prefix);
  virtual Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices) = 0;
  static int32 DevicePriority(const string& device_type);

class XlaCpuDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

class XlaGpuDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

class XlaExaDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

class BaseGPUDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)
  Status CreateGPUDevice(const SessionOptions& options, const string& name, int gpu_id, BaseGPUDevice** out_device);
  virtual BaseGPUDevice* CreateGPUDevice(const SessionOptions& options,
                                         const string& name, Bytes memory_limit,
                                         const DeviceLocality& locality,
                                         int gpu_id,
                                         const string& physical_device_desc,
                                         Allocator* gpu_allocator,
                                         Allocator* cpu_allocator) = 0;
  Status GetValidDeviceIds(const string& visible_device_list, std::vector<int>* ids);
  std::unordered_map<int, bool> visible_gpu_initialized_;

class GPUCompatibleCPUDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

class SYCLDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

class ThreadPoolDeviceFactory : public DeviceFactory
  Status CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)
```cpp
注: 系统运行必须注册一个 CPU DeviceFactory

#define REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, ...)
  static ::tensorflow::dfactory::Registrar<device_factory> _____COUNTER__object_(device_type, ##__VA_ARGS__)

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_GPU, XlaGpuDeviceFactory);
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_CPU, XlaCpuDeviceFactory);
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_EXEC, XlaExaDeviceFactory, 40);
REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210)
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70)
REGISTER_LOCAL_DEVICE_FACTORY("SYCL", SYCLDeviceFactory, 200);
REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory, 60);
```
## 源码分析

### Device

string& Device::name() // return device_attributes_.name();
DeviceNameUtils::ParsedName& parsed_name() //return parsed_name_
string& Device::device_type() // return device_attributes_.device_type();
const Device::DeviceAttributes& attributes() //device_attributes_;
void Device::Compute(OpKernel* op_kernel, OpKernelContext* context) //op_kernel->Compute(context);
void Device::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context, AsyncOpKernel::DoneCallback done) //op_kernel->ComputeAsync(context, std::move(done));
void Device::ConsumeListOfAccessedTensors(DeviceContext* context, const TensorReferenceVector& tensors)
virtual Status Device::Sync() = 0;
virtual Status Device::MaybeRewriteGraph(const FunctionDefLibrary& , std::unique_ptr<Graph>*)
virtual Status Device::FillContextMap(const Graph* graph, DeviceContextMap* device_context_map)
OpSegment* Device::op_segment() //return &op_seg_;
ResourceMgr* Device::resource_manager() //preturn rmgr_;
string Device::DebugString() //return ProtoDebugString(device_attributes_);
static DeviceAttributes Device::BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality, const string& physical_device_desc); //初始化  DeviceAttributes

static DeviceAttributes BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality) // return BuildDeviceAttributes(name, device, memory_limit, locality, "");

### DeviceSet

const std::vector<Device*>& DeviceSet::devices() // return devices_;
Device* DeviceSet::client_device() //return client_device_;
void DeviceSet::set_client_device(Device* device) // client_device_ = device;
void DeviceSet::AddDevice(Device* device) //将  {full, device} {legacy, device} 加入 device_by_name_
void DeviceSet::FindMatchingDevices(DeviceNameUtils::ParsedName& spec, std::vector<Device*>* devices) //将 devices_ 中与 spec 模式匹配的 device 加入 devices
Device* DeviceSet::FindDeviceByName(const string& name)  // device_by_name_[name]
int DeviceSet::DeviceTypeOrder(const DeviceType& d) //找到 d 类型的优先级
static bool DeviceSet::DeviceTypeComparator(const DeviceType& a, const DeviceType& b) //设备类型比较器;优先比较两个类型的优先级，如果优先级相同，比较 name
std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() // 对 devices_  找到不同的类型， 对类型根据 DeviceTypeComparator 进行排序后返回

### DeviceBase

设备的抽象基类

DeviceBase(Env* env)
virtual ~DeviceBase();
Env env()
virtual bool RequiresRecordingAccessedTensors()
void set_tensorflow_cpu_worker_threads(CpuWorkerThreads* t)
virtual const CpuWorkerThreads* tensorflow_cpu_worker_threads()
void set_tensorflow_gpu_device_info(GpuDeviceInfo* g)
virtual const GpuDeviceInfo* tensorflow_gpu_device_info()
void set_eigen_cpu_device(Eigen::ThreadPoolDevice* d)
void set_eigen_sycl_device(Eigen::SyclDevice* d)

virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
virtual Allocator* GetStepAllocator(AllocatorAttributes attr, ResourceMgr*)
virtual const Eigen::ThreadPoolDevice* eigen_cpu_device()
virtual const Eigen::SyclDevice* eigen_sycl_device()
virtual PerOpGpuDevice* MakeGpuDevice()
virtual void ReinitializeGpuDevice(OpKernelContext*, PerOpGpuDevice*, DeviceContext*, Allocator* )
virtual const DeviceAttributes& attributes()
virtual Status MakeTensorFromProto(const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs, Tensor* tensor)

### LocalDevice

LocalDevice::LocalDevice(const SessionOptions& options, const DeviceAttributes& attributes)

1. 如果 use_global_threadpool_ 为  true, 用 new LocalDevice::EigenThreadPoolInfo(options) 初始化 global_tp_info
   否则 new LocalDevice::EigenThreadPoolInfo(options) 初始化 owned_tp_info_
2. 设置
  set_tensorflow_cpu_worker_threads(&tp_info->eigen_worker_threads_);
  set_eigen_cpu_device(tp_info->eigen_device_.get());

static void set_use_global_threadpool(bool use_global_threadpool) //use_global_threadpool_ = use_global_threadpool;

### RenamedDevice

Device* RenamedDevice::NewRenamedDevice(const string& new_base, Device* underlying, bool owns_underlying)

保持  underlying 的  type, id 已经  device_attributes_ 不变，根据  new_base 重新创建  RenamedDevice，并返回

RenamedDevice::RenamedDevice(Device* underlying, const DeviceAttributes& attributes, bool owns_underlying)

其余方法与构造函数传递的 Device 名称一样，这里用了代理模式

### RemoteDevice

typedef std::function<void(const Status&, std::vector<Device*>*)> NewRemoteDevicesDone;
void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache, const string& remote_worker, NewRemoteDevicesDone done);

remote_worker : 远程 device 名

向远程发送 GetStatusRequest 请求获取远程设备状态，根据返回的 DeviceAttributes 创建 RemoteDevice 对象，将创建好的 RemoteDevice 对象传递给 done 函数对象

### XlaDevice

Status XlaDevice::Create(
    const string& platform_name, const string& device_name, int device_ordinal,
    const string& jit_device_name, const SessionOptions& options,
    const string& name_prefix, std::unique_ptr<XlaDevice>* device)


Status XlaDevice::GetMetadata(OpKernelContext* ctx, Metadata** metadata)

  ResourceMgr* rm = ctx->resource_manager();
  rm->Lookup<Metadata>(rm->default_container(), "xla_metadata", metadata));

XlaDevice::XlaDevice(const SessionOptions& options,
    const DeviceAttributes& attrs, int device_ordinal, const DeviceType& jit_device_name,
    perftools::gputools::Platform* platform, Allocator* xla_allocator)

   resource_manager()->Create<Metadata>(resource_manager()->default_container(), "xla_metadata",
      new Metadata(device_ordinal_, platform_, jit_device_name_)));

xla::LocalClient* XlaDevice::client() // return xla::ClientLibrary::GetOrCreateLocalClient(platform_).ValueOrDie();

Allocator* XlaDevice::GetAllocator(AllocatorAttributes attr)

  if (attr.on_host()) return cpu_allocator();
  else return xla_allocator_;

Status XlaDevice::FillContextMap(const Graph* graph, DeviceContextMap* device_context_map)

1. XlaDeviceContext* ctx = new XlaDeviceContext(client());
2. 将 ctx  遍历 g 中 所有节点，device_context_map

void XlaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context)

  op_kernel->Compute(context);

void XlaDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context, AsyncOpKernel::DoneCallback done)

  op_kernel->ComputeAsync(context, done);

Status XlaDevice::MakeTensorFromProto(const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs, Tensor* tensor)

1. Tensor parsed(tensor_proto.dtype());
2. parsed.FromProto(cpu_allocator(), tensor_proto)
3. if alloc_attrs.on_host() : tensor = parsed
   else
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());
    XlaTransferManager manager(client());
    manager.CopyCPUTensorToDevice(&parsed, this, &copy,
                                  [&n, &status](const Status& s) {
                                    status = s;
                                    n.Notify();
                                  });
    *tensor = copy;

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device, const char* jit_device)

TODO
1. XlaOpRegistry::RegisterCompilationKernels();
2. XlaDeviceOpRegistrations* registrations = new XlaDeviceOpRegistrations;
3  for (const KernelDef* jit_def : XlaOpRegistry::DeviceKernels(jit_device)) {
    KernelDef* def = new KernelDef(*jit_def);
    def->set_device_type(device);
    registrations->op_kernel_registrars.emplace_back(
        new kernel_factory::OpKernelRegistrar(def, "XlaDeviceDummyOp",
                                              dummy_factory));
  }

### XlaCompilationDevice


void XlaCompilationDevice::Compute(OpKernel* op_kernel, OpKernelContext* context)

  auto* b = XlaContext::Get(context).builder();
  xla::OpMetadata metadata;
  metadata.set_op_type(op_kernel->type_string());
  metadata.set_op_name(op_kernel->name());
  b->SetOpMetadata(metadata);
  op_kernel->Compute(context);
  b->ClearOpMetadata();

### BaseGPUDevice

TODO

### ThreadPoolDevice

Status ThreadPoolDevice::Sync() //return Status::OK();

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) // op_kernel->Compute(context);

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) //return allocator_;

Status ThreadPoolDevice::MakeTensorFromProto(const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs, Tensor* tensor)

将 tensor_proto(TensorProto) 转变为 parse(Tensor)

### SyclDevice

void SYCLDevice::Compute(OpKernel *op_kernel, OpKernelContext *context)

  op_kernel->Compute(context);

Allocator *SYCLDevice::GetAllocator(AllocatorAttributes attr)

  if (attr.on_host()) return cpu_allocator_;
  else return sycl_allocator_;

Status SYCLDevice::MakeTensorFromProto(const TensorProto &tensor_proto, const AllocatorAttributes alloc_attrs, Tensor tensor)

1. 将 tensor_proto(TensorProto) 转变为 parse(Tensor)
2. 如果内存分配器在 host 上，tensor 为 parse, 如果内存分配器不在  host 上，拷贝到  Device, tensor 指向设备上的 tensor

Status SYCLDevice::FillContextMap(const Graph graph, DeviceContextMap device_context_map)

  for (Node *n : graph->nodes())
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;

Status SYCLDevice::Sync() // sycl_allocator_->Synchronize();

### DeviceFactory

int32 DeviceFactory::DevicePriority(const string& device_type)

从 factories 中查找  device_type 对应的优先级

void DeviceFactory::Register(const string& device_type, DeviceFactory* factory, int priority)

如果 device_type 不存在于 factories,  那么 factories[device_type] = {factory, priority}
如果 device_type 存在于 factories, 并且  priority 比已经存在的高，就用 {factory, priority} 替代
如果 device_type 存在于 factories, 并且  priority 与已经存在的一样高， 报错
如果 device_type 存在于 factories, 并且  priority 没有已经存在的高， 什么也不做

DeviceFactory* DeviceFactory::GetFactory(const string& device_type) //factories[device_type]

Status DeviceFactory::AddDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. 查找到已经注册的 CPU 类型的 DeviceFactory，并创建一个 CPU 类型的 Device 加入 devices
2. 遍历 factories 创建其他已经注册的对应类型的 DeviceFactory， 并创建对应的 Device 加入 devices

Device* DeviceFactory::NewDevice(const string& type, const SessionOptions& options, const string& name_prefix)

1. 查找 type 类型的 DeviceFactory，根据 DeviceFactory 创建对应类型的 Device，返回创建的 Device

### XlaCpuDeviceFactory

Status XlaCpuDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(DEVICE_XLA_CPU, DEVICE_CPU_XLA_JIT);
2. XlaDevice::Create("Host", DEVICE_XLA_CPU, 0, DEVICE_CPU_XLA_JIT, options, name_prefix, &device)
3. 将 device 加入 devices 中

### XlaGpuDeviceFactory

Status XlaGpuDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(DEVICE_XLA_GPU, DEVICE_GPU_XLA_JIT);
2. XlaDevice::Create("CUDA", DEVICE_XLA_GPU, 0, DEVICE_GPU_XLA_JIT, options, name_prefix, &device);
3. device 加入 devices 中

### XlaExaDeviceFactory

Status XlaExaDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(DEVICE_XLA_EXEC, DEVICE_EXEC_XLA_JIT);
2. XlaDevice::Create("Executor", DEVICE_XLA_EXEC, 0, DEVICE_EXEC_XLA_JIT, options, name_prefix, &device);
3. device 加入 devices 中

### BaseGPUDeviceFactory

Status BaseGPUDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. 找到 valid_gpu_ids
2.遍历  valid_gpu_ids，对每个元素调用 CreateGPUDevice(options, strings::StrCat(name_prefix, "/gpu:", i), valid_gpu_ids[i], &gpu_device)
3. gpu_device->Init(options)
4. gpu_device 加入 devices 中

### GPUCompatibleCPUDeviceFactory

Status GPUCompatibleCPUDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. 从 options.config.device_count() 中找到 CPU 设备的数量 n
2. 创建 n 个 GPUCompatibleCPUDevice(options, name_prefix/cpu/i, Bytes(256 `<<` 20), DeviceLocality(), cpu_allocator()));
3. 加入 devices

### SYCLDeviceFactory

Status SYCLDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. syclInterface = GSYCLInterface::instance();
2. 从 options.config.device_count() 中找到 CPU 设备的数量 n
3. 创建 SYCLDevice(options, name, Bytes(256 `<<` 20), DeviceLocality() , syclInterface->GetShortDeviceDescription(i) , syclInterface->GetSYCLAllocator(i) , syclInterface->GetCPUAllocator(i) , syclInterface->GetSYCLContext(i)));
4. 加入 devices

### ThreadPoolDeviceFactory

Status ThreadPoolDeviceFactory::CreateDevices(const SessionOptions& options, const string& name_prefix, std::vector<Device*>* devices)

1. options.config.device_count().find("CPU");
2. 创建 ThreadPoolDevice(options, name, Bytes(256 `<<` 20), DeviceLocality(), cpu_allocator()));
3. 加入 devices

### DeviceMgr

DeviceMgr::DeviceMgr(const std::vector<Device*>& devices)

初始化 devices_，device_map_，device_type_counts_

StringPiece DeviceMgr::CopyToBackingStore(StringPiece s)

将  s 的内存拷贝到  name_backing_store_

void DeviceMgr::ListDeviceAttributes(std::vector<DeviceAttributes>* devices)

用  devices_  中每个  Device 的 attr 初始化 devices

std::vector<Device*> DeviceMgr::ListDevices()

获取所有的 device

string DeviceMgr::DeviceMappingString()

将 devices_ 每个 Device 的 physical_device_desc 属性转换成字符串

Status DeviceMgr::LookupDevice(StringPiece name, Device** device)

从  device_map_ 中找到  name 对应的  device

void DeviceMgr::ClearContainers(gtl::ArraySlice<string> containers)

遍历 devices_ 中的每个 dev(Device), 调用 dev->resource_manager()->Cleanup(c)

int DeviceMgr::NumDeviceType(const string& type)

type 对应的 device 的个数
