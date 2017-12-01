

一个鲁棒性好的系统， 必须能够支持取消操作，尤其是某些操作非常耗时的时候。

## CancellationManager

typedef int64 CancellationToken;
typedef std::function<void()> CancelCallback;

class CancellationManager
  bool is_cancelling_;
  std::atomic_bool is_cancelled_;

  mutex mu_;
  Notification cancelled_notification_;
  CancellationToken next_cancellation_token_ GUARDED_BY(mu_);
  gtl::FlatMap<CancellationToken, CancelCallback> callbacks_ GUARDED_BY(mu_);

开始, is_cancelled_ 和 is_cancelling_ 都为 false，调用 StartCancel 之后，
is_cancelling_ 为  true, is_cancelled_ 为 false, 执行完之后，is_cancelled
为 true, is_cancelling_ 为 false

因此, DeregisterCallback 只能在 StartCancel 执行之前调用，之后就不行了。

而且 CancellationManager 是一次性的，一旦调用 StartCancel 之后，重新
RegisterCallback 也不行了

## 例子

```cpp
  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(token, [&is_cancelled]() { is_cancelled = true; });
  if (registered) {
    bool deregistered = manager->DeregisterCallback(token);
  }
  delete manager;

  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(token, [&is_cancelled]() { is_cancelled = true; });
  manager->StartCancel();
  delete manager;

  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  manager->StartCancel();
  bool registered = manager->RegisterCallback(token, nullptr);
  delete manager;

  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(token, [&is_cancelled]() { is_cancelled = true; });
  manager->StartCancel();
  bool deregistered = manager->DeregisterCallback(token);
  delete manager;


  bool is_cancelled_1 = false, is_cancelled_2 = false, is_cancelled_3 = false;
  CancellationManager* manager = new CancellationManager();
  auto token_1 = manager->get_cancellation_token();
  bool registered_1 = manager->RegisterCallback(
      token_1, [&is_cancelled_1]() { is_cancelled_1 = true; });
  EXPECT_TRUE(registered_1);
  auto token_2 = manager->get_cancellation_token();
  bool registered_2 = manager->RegisterCallback(
      token_2, [&is_cancelled_2]() { is_cancelled_2 = true; });


  CancellationManager* cm = new CancellationManager();
  thread::ThreadPool w(Env::Default(), "test", 4);
  std::vector<Notification> done(8);
  for (size_t i = 0; i < done.size(); ++i) {
    Notification* n = &done[i];
    w.Schedule([n, cm]() {
      while (!cm->IsCancelled()) {
      }
      n->Notify();
    });
  }
  Env::Default()->SleepForMicroseconds(1000000 /* 1 second */);
  cm->StartCancel();
  for (size_t i = 0; i < done.size(); ++i) {
    done[i].WaitForNotification();
  }
  delete cm;

```

## 源码实现

void CancellationManager::StartCancel()

1. 如果 is_cancelling_ 或 is_cancelled_ 为 true, 表明已经执行完所有的撤销操作，或者正在执行撤销操作，返回。
2. 设置 is_cancelling_ 为 true, 遍历 callbacks_，依次调用所有的撤销回调操作
3. 设置 is_cancelling_ 为 false, is_cancelled_  为 true

CancellationToken CancellationManager::get_cancellation_token()

next_cancellation_token_ 加 1

bool CancellationManager::RegisterCallback(CancellationToken token, CancelCallback callback)

如果 既没有正在执行，也没有已经执行完撤销操作，将 token, callback 加入 callbacks_

bool CancellationManager::DeregisterCallback(CancellationToken token)

1. 如果已经执行完所有撤销操作，立即返回
2. 如果正在进行撤销操作，阻塞，直到所有的取消操作都执行完成，返回
3. 将 token 从  callbacks_ 中删除
