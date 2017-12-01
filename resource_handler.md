


### ResourceHandler

辅助创建 ResourceBase 对象

ResourceHandleProto -> ResourceHandle

  string device_;
  string container_;
  string name_;
  uint64 hash_code_ = 0;
  string maybe_type_name_;

### ResourceMgr

对 ResourceBase 的管理, CUDR

  typedef std::pair<uint64, string> Key
  typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container
  const string default_container_
  mutable mutex mu_
  std::unordered_map<string, Container*> containers_ //
  std::unordered_map<uint64, string> debug_type_names_

container_ 以 container 为 key, 每个 Container 中包含多个 ResourceBase

### ResourceBase

### ScopedStepContainer

  const string name_;
  const std::function<void(const string&)> cleanup_;

### ContainerInfo

  ResourceMgr* rmgr_ = nullptr;
  string container_; //NodeDef 的 container 属性或 ResourceMgr 的 default_container
  string name_; NodeDef 的 shared_name 或 name 或者`_NUM_node_name`
  bool resource_is_private_to_kernel_ = false; //当 NodeDef 的 shared_name 为空, 并且 use_node_name_as_default 为 false 时，才设置为 true

class IsResourceInitialized : public OpKernel
class ResourceHandleOp : public OpKernel

ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container, const string& name, const TypeIndex& type_index)

  构造并初始化 ResourceHandle

Status ResourceMgr::Create(const string& container, const string& name, T* resource)

创建一个 ResourceMgr

Status ResourceMgr::Lookup(const string& container, const string& name, T** resource)

查找 container, name 对应的 ResourceMgr，如果找到，resource 指向找到的
ResourceMgr

Status ResourceMgr::LookupOrCreate(const string& container, const string& name, T** resource, std::function<Status(T**)> creator)

1. 如果找到，resource 指向匹配的 ResourceMgr
2. 如果找不到，用 creator 创建之
3. 如果 creator 创建失败，调用 Create 创建之

Status ResourceMgr::Delete(const string& container, const string& name)

删除 container, name 对应的 ResourceMgr

Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name, T** resource)

1. 从 input_name 找到对应的 type
2. 如果 type 是 DT_RESOURCE,  查找对应的 resource, 并返回；否则继续
3. 从 input_name 找到对应的 tensor, 从 tensor 找到 container, name, 从 ctx->resource_manager() 查找  container, name 对应的 resource

ResourceHandle MakePerStepResourceHandle(OpKernelContext* ctx, const string& name)

TODO

Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value)

  return ctx->resource_manager()->Create(p.container(), p.name(), value);

Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value)

  return ctx->resource_manager()->Lookup(p.container(), p.name(), value);

Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p, T** value, std::function<Status(T**)> creator)

  return ctx->resource_manager()->LookupOrCreate(p.container(), p.name(), value, creator);

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p)

  return ctx->resource_manager()->Delete<T>(p.container(), p.name());

void IsResourceInitialized<T>::Compute(OpKernelContext* ctx)

TODO

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code, const string& type_name)

   将 hash_code, type_name 加入 debug_type_names_

const char* ResourceMgr::DebugTypeName(uint64 hash_code)

   从 debug_type_names_ 中获取 hash_code 对应的 type_name

Status ResourceMgr::DoCreate(const string& container, TypeIndex type, const string& name, ResourceBase* resource)

   以 container 为 key, value 为 type.hash_code(),name 组成的 Container, 保存到 containers_

Status ResourceMgr::DoLookup(const string& container, TypeIndex type, const string& name, ResourceBase** resource)

   1. 从 container_  中根据 container  找到 Container
   2. 从 Container 中根据 type.hash_code(), name 找到 resource

Status ResourceMgr::DoDelete(const string& container, uint64 type_hash_code, const string& resource_name, const string& type_name)

   1. 从 container_  中根据 container  找到 Container
   2. 从 Container 中根据 type_hash_code, resource_name 找到 resource, 从 Container 中删除之

Status ResourceMgr::Cleanup(const string& container)

   从 container_ 中删除 container 对应的 Container

Status ContainerInfo::Init(ResourceMgr* rmgr, const NodeDef& ndef, bool use_node_name_as_default)

   用 ndef 初始化 ContainerInfo

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p)
   从 cts->resource_manager() 中删除  p.container(), p.hash_code(), p.name()
   对应的 Container
