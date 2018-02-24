
 跨系统之间相互交互选项



class CallOptions
  mutex mu_;
  CancelFunction cancel_func_ GUARDED_BY(mu_);
  int64 timeout_in_ms_ GUARDED_BY(mu_); // RPC operation timeout in milliseconds.

void CallOptions::StartCancel() // cancel_func_()
void CallOptions::SetCancelCallback(CancelFunction cancel_func)  //cancel_func_ = std::move(cancel_func);
void CallOptions::ClearCancelCallback() //cancel_func_ = nullptr;
int64 CallOptions::GetTimeout() // return timeout_in_ms_;
void CallOptions::SetTimeout(int64 ms) // timeout_in_ms_ = ms;
