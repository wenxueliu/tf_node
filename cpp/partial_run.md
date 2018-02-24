

class PartialRunMgr
  mutex mu_;
  std::unordered_map<int, std::unique_ptr<PartialRunState>> step_id_to_partial_run_ GUARDED_BY(mu_);
  struct PartialRunState {
    std::unique_ptr<CancellationManager> cancellation_manager;
    bool executor_done = false;
    StatusCallback final_callback = nullptr;
    Status final_status;


bool PartialRunMgr::FindOrCreate(int step_id, CancellationManager** cancellation_manager)

从 step_id_to_partial_run_ 中查找 step_id 对应的 PartialRunState
如果找到，保存到  cancellation_manager
如果没有找到，创建之，并加入  step_id_to_partial_run_

void PartialRunMgr::ExecutorDone(int step_id, const Status& executor_status)

1. 找到  step_id 对应的 PartialRunState run_it.
2. 没有找到直接返回，找到继续
3. run_it->second->final_callback(run_it->second->final_status); run_it->second->executor_done = true;
4. step_id_to_partial_run_.erase(step_id);

void PartialRunMgr::PartialRunDone(int step_id, StatusCallback done, const Status& status)

1. 找到  step_id 对应的 PartialRunState run_it.
2. 没有找到直接返回，找到继续
3. run_it->second->final_callback(run_it->second->final_status);
4. step_id_to_partial_run_.erase(step_id);
