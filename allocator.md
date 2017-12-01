
 这里采用了装饰器模式

## AllocatorRegistry

struct AllocatorRegistryEntry;
  string name;
  int priority;
  Allocator* allocator;

class AllocatorRegistry
  std::vector<AllocatorRegistryEntry> allocators_; //
  Allocator* m_curr_allocator_;  //allocator_ 中优先级最高的 Allocator

class AllocatorRegistration

struct AllocatorAttributes
  uint32 value = 0; //第一位表示 on_host,  第二位表示 nic 兼容, 第三位表示 gpu 兼容

class AllocatorWrapper : public Allocator //代理模式

```
#define REGISTER_MEM_ALLOCATOR(name, priority, allocator)
  static allocator_registration::AllocatorRegistration
      register_allocator___COUNTER__(name, priority, new allocator)
```

## Allocator

struct AllocatorStats
  int64 num_allocs;       //AllocateRaw 调用次数
  int64 bytes_in_use;     //已经使用的 byte, AllocateRaw 时增加，DeallocateRaw 时减少
  int64 max_bytes_in_use; //历史使用总的 byte, 调用 AllocateRaw  分配的所有空间的大小
  int64 max_alloc_size;   //历史上分配过的最大的空间
  // The upper limit what the allocator can allocate, if such a limit
  // is known. Certain allocator may return 0 to indicate the limit is
  // unknown.
  int64 bytes_limit;

struct AllocationAttributes
  bool no_retry_on_failure = false;
  bool allocation_will_be_logged = false;


class CPUAllocator : public Allocator
  mutex mu_
  AllocatorStats stats_

class TrackingAllocator : public Allocator
  Allocator* allocator_;  // not owned.
  mutex mu_;
  int ref_ //初始化为 1
  size_t allocated_ //上一次调用 AllocateRaw 使用的空间大小
  size_t high_watermark_ //历史上分配过的最大的空间
  size_t total_bytes_ //历史使用总的 byte, 调用 AllocateRaw  分配的所有空间的大小
  const bool track_sizes_locally_;
  std::unordered_map<void*, Chunk> in_use_ //key : Allocator, value : 分配信息
  int64 next_allocation_id_ //只有 track_sizes 为 true, 并且 allocator_->TracksAllocationSizes() 为 0 时，每次分配在加一

目前的实现  allocated_ 和  high_watermark_, total_bytes_ 始终相同

bool cpu_allocator_collect_stats = true; //统计信息在 CPUAllocator 中统计
bool cpu_allocator_collect_full_stats = true; //统计信息在 TrackingAllocator 中统计

### 实例
``` cpp
  EnableCPUAllocatorStats(true);
  Allocator* a = cpu_allocator();
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a->AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  std::sort(ptrs.begin(), ptrs.end());

  float* t1 = a->Allocate<float>(1024);
  double* t2 = a->Allocate<double>(1048576);
  a->Deallocate(t1, 1024);
  a->Deallocate(t2, 1048576);

  //total_bytes_, high_watermark_, allocated_, AllocationId
  TrackingAllocator* ta = new TrackingAllocator(a, false);
  void* p1 = ta->AllocateRaw(4, 4); //4, 4, 4, 1
  ta->DeallocateRaw(p1); //4, 4, 0, 1

  void* p2 = ta->AllocateRaw(4, 12); //16, 12, 12, 2
  ta->DeallocateRaw(p2); //16, 12, 0, 2

  TestableSizeTrackingAllocator a = TestableSizeTrackingAllocator();
  ta = new TrackingAllocator(a, true);
  p1 = ta->AllocateRaw(4, 4); //4, 4, 4, 1
  ta->DeallocateRaw(p1); //4, 4, 0, 1
  p2 = ta->AllocateRaw(4, 12); //16, 12, 12, 2
  ta->DeallocateRaw(p2); //16, 12, 0, 2

  TrackingAllocator* ta = new TrackingAllocator(&a, false);
  void* p1 = ta->AllocateRaw(4, 12); //12, 12, 12, 1
  ta->DeallocateRaw(p1); //12, 12, 0, 1
  void* p2 = ta->AllocateRaw(4, 4); //16, 12, 4, 2
  ta->DeallocateRaw(p2); //16, 12, 0, 2

```

### 源码解析

void* CPUAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
1. 分配 num_bytes  的内存
2. 初始化更新 stats_

关于对齐

```
#ifdef EIGEN_VECTORIZE_AVX512
  // Align to 64 byte boundary.
  static constexpr size_t kAllocatorAlignment = 64;
#else
  // Align to 32 byte boundary.
  static constexpr size_t kAllocatorAlignment = 32;
#endif
```


void CPUAllocator::GetStats(AllocatorStats* stats)

stats 指向 stats_

void DeallocateRaw(void* ptr)

1. 更新 bytes_in_use
2. 释放 ptr 的内存

size_t AllocatedSizeSlow(void* ptr)

ptr 所需内存空间大小

TrackingAllocator::TrackingAllocator(Allocator* allocator, bool track_sizes)

Allocator* MakeCpuAllocator()
  构造一个 CPUAllocator

Allocator* cpu_allocator()
  从已经注册的 allocator 中获取一个 CPUAllocator

void* TrackingAllocator::AllocateRaw(size_t alignment, size_t num_bytes, const AllocationAttributes& allocation_attr)

1. 调用 allocator_->AllocateRaw
2. 初始化 allocated_, high_watermark_, total_bytes_

void TrackingAllocator::DeallocateRaw(void* ptr)

1. 调整 allocated_
2. 从 in_use_ 中删除 ptr
2. 减少引用计数，如果 引用计数为 1, 销毁当前对象

bool TrackingAllocator::TracksAllocationSizes()

是否记录分配空间大小

size_t TrackingAllocator::RequestedSize(void* ptr)

ptr  构造时请求的空间大小

size_t TrackingAllocator::AllocatedSize(void* ptr)

ptr 的 allocated_

int64 TrackingAllocator::AllocationId(void* ptr)

ptr 的 next_allocation_id_

void TrackingAllocator::GetStats(AllocatorStats* stats)

allocator_ 的状态信息保存在 stats

std::tuple<size_t, size_t, size_t> TrackingAllocator::GetSizesAndUnRef()

1. 引用计数减一
2. 返回统计信息

bool TrackingAllocator::UnRef()

引用计数减一，如果可以删除，就返回 true

Allocator* AllocatorRegistry::GetRegisteredAllocator(const string& name, int priority)

从  allocators_ 中找到  name, priority 都相同的 Allocator

void AllocatorRegistry::Register(const string& name, int priority, Allocator* allocator)

1. 将  name, priority, allocator 构造 AllocatorRegistryEntry 加入 allocators_
2. 设置 m_curr_allocator_ 为优先级最高的 Allocator

Allocator* AllocatorRegistry::GetAllocator()

返回优先级最高的 Allocator
