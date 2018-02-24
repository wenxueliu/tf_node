
Pack 和  Large 的区别在于  Pack 适用于数量比较少的情况

PackedCounts 最大支持计数到 8(3 位)
LargeCounts  最大支持计数到 31(2^31-1)


class PendingCounts
  const int num_bytes_;  // Just for bounds checking in debug mode
  char* bytes_;          // Array of num_bytes_ bytes
  enum NodeState
    // The pending count for the node > 0.
    PENDING_NOTREADY,
    // The pending count for the node == 0, but the node has not started executing.
    PENDING_READY,
    // The node has started executing.
    STARTED,
    // The node has finished executing.
    COMPLETED
  class Layout
    int next_offset_ = 0;  // Next byte offset to allocate

  static const int kMaxCountForPackedCounts = 7;
  struct PackedCounts {
    uint8 pending : 3;
    uint8 dead_count : 3;
    uint8 has_started : 1;

  struct LargeCounts {
    uint32 pending;
    uint32 dead_count : 31;
    uint8 has_started : 1;

  class Handle
    int byte_offset_ : 31;  // Byte offset of the rep in PendingCounts object
    bool is_large_ : 1;  // If true, rep is LargeCounts; otherwise PackedCounts

if (c->has_started) return (c->pending == 0) ? STARTED : COMPLETED;
else return (c->pending == 0) ? PENDING_READY : PENDING_NOTREADY;


## 源码分析

void set_initial_count(Handle h, size_t pending_count) //设置计数器的初始值
NodeState node_state(Handle h) //获取当前状态
int pending(Handle h) // 获取 h 对应的计数器，如果已经开始 pending 始终为 0
int decrement_pending(Handle h, int v) // h 对应的计数器减少 v  //BUG 应该检查 has_started
void mark_started(Handle h) //设置 has_started 为 1，表明开始
void mark_completed(Handle h) //设置 h 对应的计数器 has_started = 1, pending 为 1，表明完成
int dead_count(Handle h) //获取 h 对应的 dead_count
void increment_dead_count(Handle h) //增加 h 对应的 dead_count
void mark_live(Handle h) //TODO
void adjust_for_activation(Handle h, bool increment_dead, int* pending_result, int* dead_result) //pending 减 1, dead_count 加 1(如果 increment_dead 为  true)

TODO

inline PendingCounts::Handle PendingCounts::Layout::CreateHandle(size_t max_pending_count, size_t max_dead_count)
