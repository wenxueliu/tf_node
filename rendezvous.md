
 主要的功能是在系统之家传递 Tensor, 发送端发送，接受端接受，每个 tensor
 以 key 为来标记该  Tensor 的源设备和目的设备

### 源文件

core/framework/rendezvous.h
core/framework/rendezvous.cc
core/common_runtime/rendezvous_mgr.h
core/common_runtime/rendezvous_mgr.cc
core/distributed_runtime/rendezvous_mgr_interface.h
core/distributed_runtime/base_rendezvous_mgr.h
core/distributed_runtime/base_rendezvous_mgr.cpp
core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc
contrib/mpi/mpi_rendezvous_mgr.h
contrib/mpi/mpi_rendezvous_mgr.cc
contrib/verbs/rdma_rendezvous_mgr.h
contrib/verbs/rdma_rendezvous_mgr.cc

## 数据结构

class Rendezvous
  struct Args
    DeviceContext* device_context = nullptr;
    AllocatorAttributes alloc_attrs;
  struct ParsedKey {
    StringPiece src_device;
    DeviceNameUtils::ParsedName src;
    uint64 src_incarnation = 0;
    StringPiece dst_device;
    DeviceNameUtils::ParsedName dst;
    StringPiece edge_name;
    string buf_;

class LocalRendezvousImpl : public Rendezvous
  typedef std::deque<Item*> ItemQueue;
  typedef gtl::FlatMap<uint64, ItemQueue> Table;
  mutex mu_;
  Table table_ GUARDED_BY(mu_);
  Status status_ GUARDED_BY(mu_);
  struct Item {
    DoneCallback waiter = nullptr; //为空时，IsSendValue 为 true
    Tensor value;
    bool is_dead = false;
    Args send_args; //发送端设置
    Args recv_args; //接受端设置

这里是一个非常巧妙的接受和发送者都不阻塞的的队列实现。 通过接受者没有收到元素
时设置接受者的回调函数，达到接受者非阻塞的目的。

table_[key_hash] 队列中存放元素，每个元素 IsSendValue 为 True 或 Fasle,
每个待发送的 Tensor 包装成 Item, 如果是 Sender 放入的元素就设置 IsSendValue 为
True， 如果是 Receiver 放入的元素就设置 IsSendValue 为 False

每个 Sender 取队列头的元素，如果是 Receiver 放进去的元素，会设置 waiter 回调函数，就调用
waiter 回调函数，如果是自己放进去的元素，说明 Receiver 没有取之前放置的元素，构建 Item
放到队列尾部

每个 Receiver 取队列头的元素，如果是 Sender 放进去的元素，就对该该元素调用 done
方法，如果是自己放进去的元素，说明 Sender 最近没有元素要发送，就构造 Item
放到队列尾部。


class SimpleRendezvous : public Rendezvous //不能跨主机
  typedef std::unordered_map<string, Tensor> Table;
  mutex mu_;
  Table table_ GUARDED_BY(mu_);

class IntraProcessRendezvous : public Rendezvous
  const DeviceMgr* device_mgr_;
  Rendezvous* local_;  // Owns a Ref on this object.
  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_); // Status given by StartAbort() if any.
  typedef std::function<void(const Status&)> StatusCallback;

class RemoteRendezvous : public Rendezvous
 public:
  // Fully construct the RemoteRendezvous.
  virtual Status Initialize(WorkerSession* session) = 0;

class BaseRemoteRendezvous : public RemoteRendezvous
  const WorkerEnv* const env_;  // Not owned.
  const int64 step_id_;
  Rendezvous* local_;  // NewLocalRendezvous()
  mutable mutex mu_;
  // Status given by StartAbort() if any.
  Status status_ GUARDED_BY(mu_);
  WorkerSession* session_ GUARDED_BY(mu_);  // Not owned.
  // Data structures to handle calls when partially initialized.
  struct DeferredCall
    const ParsedKey parsed; //Tensor 对应的 key
    DoneCallback done;      //对 key 对应的 Tensor 所做的处理
  std::vector<DeferredCall> deferred_calls_ GUARDED_BY(mu_);
  // Active outstanding RecvTensor calls.
  gtl::FlatSet<BaseRecvTensorCall*> active_ GUARDED_BY(mu_);

class RendezvousMgrInterface {
  virtual RemoteRendezvous* Find(int64 step_id) = 0;
  virtual void RecvLocalAsync(int64 step_id, const Rendezvous::ParsedKey& parsed, Rendezvous::DoneCallback done) = 0;
  virtual Status RecvLocal(int64 step_id, const Rendezvous::ParsedKey& parsed, Tensor* val, bool* is_dead) = 0;
  virtual void Cleanup(int64 step_id) = 0;
  virtual void CleanupAll() = 0;

class BaseRendezvousMgr : public RendezvousMgrInterface
  typedef gtl::FlatMap<int64, BaseRemoteRendezvous*> Table;
  const WorkerEnv* const worker_env_;
  mutex mu_;
  Table table_ GUARDED_BY(mu_); //step_id 和 BaseRemoteRendezvous 的映射关系


class RpcRendezvousMgr : public BaseRendezvousMgr

class RpcRemoteRendezvous : public BaseRemoteRendezvous

class BaseRecvTensorCall
  virtual void Start(std::function<void()> recv_done) = 0;
  virtual void StartAbort(const Status& s) = 0;
  virtual Status status() const = 0;

class RpcRecvTensorCall : public BaseRecvTensorCall
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;
  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

class RpcRecvTensorFreeList
  static const int kMaxObjects = 1000;
  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ GUARDED_BY(mu_);


## 例子

  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/CPU:0", 7890,
      "/job:mnist/replica:1/task:2/GPU:0", "var0", FrameAndIter(0, 0));

  EXPECT_EQ(key,
            "/job:mnist/replica:1/task:2/CPU:0;"
            "0000000000001ed2;"  // 7890 = 0x1ed2
            "/job:mnist/replica:1/task:2/GPU:0;"
            "var0;"
            "0:0");

class LocalRendezvousTest : public ::testing::Test {
 public:
  LocalRendezvousTest() : threads_(Env::Default(), "test", 16) {
    rendez_ = NewLocalRendezvous();
  }

  ~LocalRendezvousTest() override {
    rendez_->Unref();
  }

  void SchedClosure(std::function<void()> fn) {
    threads_.Schedule(std::move(fn));
  }

  Rendezvous::ParsedKey MakeKey(const string& name) {
    string s = Rendezvous::CreateKey("/job:mnist/replica:1/task:2/CPU:0", 7890,
                                     "/job:mnist/replica:1/task:2/GPU:0", name,
                                     FrameAndIter(0, 0));
    Rendezvous::ParsedKey k;
    Rendezvous::ParseKey(s, &k);
    return k;
  }

  Tensor V(const string& content) {
    Tensor tensor(DT_STRING, TensorShape({}));
    tensor.scalar<string>()() = content;
    return tensor;
  }

  void SendRecv() {
    Rendezvous::Args args;
    rendez_->Send(MakeKey("foo"), args, V("hello"), false);
    bool is_dead = false;
    rendez_->Recv(MakeKey("foo"), args, &val, &is_dead);
  }

  void RecvSend() {
    SchedClosure([this]() {
      Env::Default()->SleepForMicroseconds(10000);
      Rendezvous::Args args;
      rendez_->Send(MakeKey("foo"), args, V("hello"), false);
    });
    Tensor val(DT_STRING);
    bool is_dead = false;
    Rendezvous::Args args;
    rendez_->Recv(MakeKey("foo"), args, &val, &is_dead);
  }

  struct BlockingState {
    mutex lock;
    int counter = 0;
    Notification done;
  };

  void PingPong() {
    SchedClosure([this]() {
      Tensor t(DT_STRING);
      bool is_dead = false;
      Rendezvous::Args args;
      rendez_->Recv(MakeKey("foo"), args, &t, &is_dead);
      rendez_->Send(MakeKey("bar"), args, t, is_dead);
    });
    Env::Default()->SleepForMicroseconds(1000000);
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    rendez_->Send(MakeKey("foo"), args, V("secret msg"), val_dead);
    rendez_->Recv(MakeKey("bar"), args, &val, &val_dead);
  }

  void RandomSendRecv() {
    static const int N = 100;
    random::PhiloxRandom philox(testing::RandomSeed(), 17);
    random::SimplePhilox rnd(&philox);
    BlockingState state;
    state.counter = N;
    for (int i = 0; i < N; ++i) {
      int micros = 100 + rnd.Uniform(1000);
      SchedClosure([this, i, micros]() {
        Env::Default()->SleepForMicroseconds(micros);
        Rendezvous::Args args;
        TF_ASSERT_OK(rendez_->Send(MakeKey(strings::StrCat(i)), args,
                                   V(strings::StrCat(i)), false));
      });
      auto recv_done = [this, &state, i](const Status& status,
                                         const Rendezvous::Args& sender_args,
                                         const Rendezvous::Args& recver_args,
                                         const Tensor& val, const bool val_dead) {
        EXPECT_EQ(strings::StrCat(i), V(val));
        bool done = false;
        {
          mutex_lock l(state.lock);
          state.counter--;
          if (state.counter == 0) {
            done = true;
          }
        }
        if (done) {
          state.done.Notify();
        }
      };
      micros = 100 + rnd.Uniform(1000);
      SchedClosure([this, i, micros, recv_done]() {
        Env::Default()->SleepForMicroseconds(micros);
        rendez_->RecvAsync(MakeKey(strings::StrCat(i)), Rendezvous::Args(),
                           recv_done);
      });
    }

    state.done.WaitForNotification(); //done = true 时返回
  }

  void TransferDummyDeviceContext {
    Rendezvous::Args args;
    args.device_context = new DummyDeviceContext(123);

    rendez_->Send(MakeKey("foo"), args, V("hello"), false));

    Notification n;
    Rendezvous::Args args1;
    args1.device_context = new DummyDeviceContext(1);
    rendez_->RecvAsync(
        MakeKey("foo"), args1,
        [&n](const Status& s, const Rendezvous::Args& send_args,
             const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead) {
          CHECK_EQ(123, dynamic_cast<const DummyDeviceContext*>(
                            send_args.device_context)
                            ->stream_id());
          n.Notify();
        });

    n.WaitForNotification();
    args.device_context->Unref();
    args1.device_context->Unref();
  }

  void BM_SendRecv(int iters) {
    Tensor orig = V("val");
    Tensor val(DT_STRING, TensorShape({}));
    bool is_dead = false;
    Rendezvous::Args args;
    Status s;
    if (iters > 0) {
      while (iters--) {
        rendez_->Send(MakeKey("foo"), args, orig, is_dead);
        rendez_->Recv(MakeKey("foo"), args, &val, &is_dead);
      }
      CHECK_EQ(V(val), V(orig));
    }
    rendez->Unref();
  }

  void BM_PingPong(int iters) {
    CHECK_GT(iters, 0);
    // The main thread sends "foo" for iters times and receives "bar"
    // for iters times.  The other thread sends "bar" for iters times
    // and receives "foo" for iters times.
    Rendezvous* rendez = NewLocalRendezvous();
    thread_->Schedule([rendez, iters]() {
      Tensor bar = V("bar");
      Tensor foo(DT_STRING, TensorShape({}));
      bool is_dead = false;
      Rendezvous::Args args;
      Status s;
      for (int i = 0; i < iters; ++i) {
        rendez->Recv(KeyFoo(), args, &foo, &is_dead);
        rendez->Send(KeyBar(), args, bar, is_dead);
      }
      CHECK_EQ("foo", V(foo));
    });
    Tensor foo = V("foo");
    Tensor bar(DT_STRING, TensorShape({}));
    bool is_dead = false;
    Rendezvous::Args args;
    Status s;
    for (int i = 0; i < iters; ++i) {
      rendez->Send(KeyFoo(), args, foo, is_dead);
      rendez->Recv(KeyBar(), args, &bar, &is_dead);
    }
    CHECK_EQ("bar", V(bar));
  }

  Rendezvous* rendez_;

  private:
    thread::ThreadPool threads_;
};

class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {}
  ~DummyDeviceContext() override {}
  int stream_id() const { return stream_id_; }

 private:
  const int stream_id_;
};


## 源码分析

### Rendezvous

static string Rendezvous::CreateKey(const string& src_device, uint64 src_incarnation,
                          const string& dst_device, const string& name,
                          const FrameAndIter& frame_iter)

 返回 src_device";"strings::Uint64ToHexString(src_incarnation, buf)";"dst_device";"name";"frame_iter.frame_id":"frame_iter.iter_id

static StringPiece ConsumeNextPart(StringPiece* s, char delim)

1. 从 s 中删除 delim 之前的部分(包含 delim)
2. 返回 s 中 delim 之前的部分

Status Rendezvous::ParseKey(StringPiece key, ParsedKey* out)

解析 out(CreateKey 的格式相同), 并初始化 key

Status Rendezvous::Recv(const ParsedKey& key, const Args& recv_args, Tensor* val, bool* is_dead, int64 timeout_ms)

调用 RecvAsync 并设置超时时间，如果  timeout_ms 大于零就设置超市时间，如果小于等于 0，就一直阻塞

Status Rendezvous::Recv(const ParsedKey& key, const Args& args, Tensor* val, bool* is_dead)

调用 Recv  一直阻塞到完成

### LocalRendezvousImpl

Status LocalRendezvousImpl::Send(const ParsedKey& key, const Args& send_args, const Tensor& val, const bool is_dead)

1.  计算 key 的哈希值 key_hash，作为 table_ 的 key
2.  如果 table_[key_hash] 对应的队列为空，构造  Item, 加入该队列，返回
3.  如果 table_[key_hash] 对应的队列不为空， 从 queue 中取出一个 Item, 调用 Item 的 waiter 方法

void LocalRendezvousImpl::RecvAsync(const ParsedKey& key, const Args& recv_args, DoneCallback done)

1.  计算 key 的哈希值 key_hash，作为 table_ 的 key
2.  如果 table_[key_hash] 对应的队列为空，构造  Item, 加入该队列，返回
3.  如果 table_[key_hash] 对应的队列不为空， 从 queue 中取出一个 Item, 调用 done 方法

void StartAbort(const Status& status)

1. 将 table_ 中的元素保存在临时变量中
2. 调用每个元素的 waiter 方法

Rendezvous* NewLocalRendezvous()

    return new LocalRendezvousImpl();

### SimpleRendezvous

Status SimpleRendezvous::Send(const ParsedKey& parsed, const Args& send_args, const Tensor& val, const bool is_dead) //table_[parsed.edge_name.ToString()] = val;

void SimpleRendezvous::RecvAsync(const ParsedKey& parsed, const Args& recv_args, DoneCallback done)

1. tensor = table_[parsed.edge_name.ToString()];
2. done(status, Args{}, recv_args, tensor, false);

### IntraProcessRendezvous

Status IntraProcessRendezvous::Send(const ParsedKey& parsed, const Rendezvous::Args& args, const Tensor& val, const bool is_dead)

1.  计算 parsed 的哈希值 key_hash，作为 local_.table_ 的 key
2.  如果 local_.table_[key_hash] 对应的队列为空，构造 Item, 加入该队列，返回
3.  如果 local_.table_[key_hash] 对应的队列不为空，从 queue 中取出一个 Item, 调用 Item 的 waiter 方法

Status IntraProcessRendezvous::ParseKey(const string& key, bool is_src, Rendezvous::ParsedKey* parsed)

  Rendezvous::ParseKey(key, parsed)

void IntraProcessRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done)

1. 如果 send_args 和 recv_args 在同一类设备，直接调用 done，返回；否则继续
2. 将 in 通过 DMA 拷贝到 out 之后调用 done

void IntraProcessRendezvous::RecvAsync(const ParsedKey& parsed, const Rendezvous::Args& recv_args, DoneCallback done)

1.  计算 parsed 的哈希值 key_hash，作为 local_.table_ 的 key
2.  如果 local_.table_[key_hash] 对应的队列为空，构造  Item, 加入该队列，返回
3.  如果 local_.table_[key_hash] 对应的队列不为空， 从 queue 中取出一个 Item, 将 Item 中的 Tensor 拷贝到目标设备

void IntraProcessRendezvous::StartAbort(const Status& s) //local_->StartAbort(s);

### BaseRendezvousMgr

RemoteRendezvous* BaseRendezvousMgr::Find(int64 step_id) //从 this.table_ 中找到 step_id 对应的 RemoteRendezvous，如果找不到就创建之
BaseRemoteRendezvous* BaseRendezvousMgr::FindOrCreate(int64 step_id) //从 this.table_ 中找到 step_id 对应的 RemoteRendezvous，如果找不到就创建之
void BaseRendezvousMgr::RecvLocalAsync(int64 step_id, const Rendezvous::ParsedKey& parsed, Rendezvous::DoneCallback done)
找到  step_id 对应的 Rendezvous， 对收到的 Item 用 done 进行处理

Status BaseRendezvousMgr::RecvLocal(int64 step_id, const Rendezvous::ParsedKey& parsed, Tensor* val, bool* is_dead) //接受  parsed 对应的 Tensor, 将 其保存在 val 中

void BaseRendezvousMgr::Cleanup(int64 step_id)

如果找到 step_id 对应的 Rendezvous，从 table_ 中删除，调用对应的 StartAbort 和 UnRef 方法
如果找不到，什么也不做

void BaseRendezvousMgr::CleanupAll()

将 table_ 中的  Rendezvous 从 table_ 中删除，并依次调用对应的 StartAbort 和 UnRef 方法

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64 step_id, const WorkerEnv* worker_env) // return new RpcRemoteRendezvous(worker_env, step_id);

### BaseRemoteRendezvous

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env, int64 step_id)

static bool IsLocalDevice(const string& worker_name, const StringPiece device_name)

Status BaseRemoteRendezvous::Initialize(WorkerSession* session)

1. 用  session 初始化  session_
2. 遍历 deferred_calls_ 中的元素，接受对应的 Tensor，并处理之

WorkerSession* BaseRemoteRendezvous::session() //session_

Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& args, const Tensor& val, const bool is_dead) //调用 local_->Send(parsed, args, val, is_dead);

Status BaseRemoteRendezvous::ValidateDevices(const ParsedKey& parsed, bool is_src)

void BaseRemoteRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done)

1. 如果 send_args 和 recv_args 在同一类设备，直接调用 done，返回；否则继续
2. 将 in 通过 DMA 拷贝到 out 之后调用 done

bool BaseRemoteRendezvous::IsSameWorker(DeviceNameUtils::ParsedName src, DeviceNameUtils::ParsedName dst)

src 和 dst 是否相同，即 a.job == b.job && a.replica == b.replica && a.task == b.task)

void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed, const Rendezvous::Args& recv_args, DoneCallback done)

1. 如果 parsed 对应的  Tensor 在同一 Worker， 通过  DMA 从源设备拷贝到目标设备
2. 如果不在同一 work，调用 RecvFromRemoteAsync(parsed, recv_args, std::move(done)); TODO

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed, DoneCallback done) //local_->RecvAsync(parsed, Args(), std::move(done));

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed, DoneCallback done) //local_->RecvAsync(parsed, Args(), std::move(done));

void BaseRemoteRendezvous::StartAbort(const Status& s)

遍历  active_ 的每个元素，调用 StartAbort

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call)

如果 status_ 成功，加入 active_

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call)

将 call 从 active_ 中删除

### RpcRecvTensorFreeList

RpcRecvTensorCall* RpcRecvTensorFreeList::New()

如果 object_ 不为空, 从 object_ 中弹出一个 RpcRecvTensorCall
如果 object_ 为空，就新建一个

void Release(RpcRecvTensorCall* obj, WorkerCacheInterface* wc) //将  obj 用 wc 重置之后加入  object_


static RpcRecvTensorFreeList* get_call_freelist() //新建一个 RpcRecvTensorFreeList

### RpcRemoteRendezvous

void RpcRemoteRendezvous::RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args, DoneCallback done)

```
  RpcRecvTensorCall* call = get_call_freelist()->New();
  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->CreateWorker(call->src_worker_);
  sess->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device, recv_args, std::move(done));
  RegisterCall(call);
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
```

### RpcRecvTensorCall

void RpcRecvTensorCall::Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done)

初始化 RpcRecvTensorCall

void Reset(WorkerCacheInterface* wc) //用 wc 重置

void Start(std::function<void()> recv_done)

接受 Tensor 并调用 recv_done

void StartAbort(const Status& s) //opts_.StartCancel();
Status status() //return status_;
Tensor& tensor() //return resp_.tensor();
bool is_dead() //return resp_.metadata().is_dead();
Device* dst_device() //return dst_device_;
Rendezvous::Args& recv_args() //return recv_args_;
Rendezvous::DoneCallback& done() //return done_;
void StartRTCall(std::function<void()> recv_done) //

```
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
```
