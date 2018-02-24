
class ReaderBaseState
  mutable mutex mu_;
  const string name_;
  int64 work_started_ = 0;       //每次调用 OnWorkStartedLocked 加 1
  int64 work_finished_ = 0;      //已经完成的数量
  int64 num_records_produced_ = 0;
  string work_;

class ReaderInterface : public ResourceBase
class ReaderBase : public ReaderInterface

class ResourceOpKernel : public OpKernel
class ReaderOpKernel : public ResourceOpKernel<ReaderInterface>

### 实现

string& name()  返回 name_
bool work_in_progress() //返回 work_finished_ 是否小于 work_started_
string& current_work() const //返回 work_;
int64 ReaderBase::NumRecordsProduced() //  返回 num_records_produced_
int64 ReaderBase::NumWorkUnitsCompleted() //返回  work_finished_
Status ReaderBase::Reset() //重置所有变量
Status ReaderBase::ResetLocked() //重置所有变量
Status ReaderBase::SerializeState(string* state) //SerializeStateLocked 加锁
Status ReaderBase::SerializeStateLocked(string* state) //未实现
Status ReaderBase::RestoreState(const string& state) //RestoreStateLocked 加锁版
Status ReaderBase::RestoreStateLocked(const string& state) //未实现
int64 ReaderBase::ReadUpTo(const int64 num_records, QueueInterface* queue, std::vector<string>* keys,
    std::vector<string>* values, OpKernelContext* context)

1. 调用 ReadUpToLocked 读剩余的记录
2. 正常读完成调用 OnWorkFinishedLocked

string ReaderBase::GetNextWorkLocked(QueueInterface* queue, OpKernelContext* context)

从 queue 中取出一个元素返回

void ReaderBase::SaveBaseState(ReaderBaseState* state)

将当期的状态保存在 state

string ReaderBase::KeyName(const string& key)

 返回 "work_:key"

Status ReaderBase::RestoreBaseState(const ReaderBaseState& state)

从 state 恢复当前状态
