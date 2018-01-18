
## 数据结构

### RunStepRequest

class RunStepRequestWrapper
  virtual const string& session_handle() const = 0;
  virtual const string& partial_run_handle() const = 0;
  virtual size_t num_feeds() const = 0;
  virtual const string& feed_name(size_t i) const = 0;
  virtual Status FeedValue(size_t i, Tensor* out_tensor) const = 0;
  virtual Status FeedValue(size_t i, TensorProto* out_tensor) const = 0;
  virtual size_t num_fetches() const = 0;
  virtual const string& fetch_name(size_t i) const = 0;
  virtual size_t num_targets() const = 0;
  virtual const string& target_name(size_t i) const = 0;
  virtual const RunOptions& options() const = 0;
  virtual string DebugString() const = 0;
  virtual const RunStepRequest& ToProto() const = 0;

class MutableRunStepRequestWrapper : public RunStepRequestWrapper

class InMemoryRunStepRequest : public MutableRunStepRequestWrapper //客户端和 Master 在不同的地址空间(address spaces)
  string session_handle_;
  string partial_run_handle_;
  gtl::InlinedVector<std::pair<string, Tensor>, 4> feeds_;
  gtl::InlinedVector<string, 4> fetches_;
  gtl::InlinedVector<string, 4> targets_;
  RunOptions options_;
  mutable std::unique_ptr<RunStepRequest> proto_version_;

class MutableProtoRunStepRequest : public MutableRunStepRequestWrapper //客户端和 Master 在不同的地址空间(address spaces)
  RunStepRequest request_;

class ProtoRunStepRequest : public RunStepRequestWrapper
  const RunStepRequest* const request_;  // Not owned.

### RunGraphRequestWrapper

class RunGraphRequestWrapper
  virtual const string& session_handle() const = 0;
  virtual const string& graph_handle() const = 0;
  virtual int64 step_id() const = 0;
  virtual const ExecutorOpts& exec_opts() const = 0;
  virtual size_t num_sends() const = 0;
  virtual const string& send_key(size_t i) const = 0;
  virtual Status SendValue(size_t i, Tensor* out_tensor) const = 0;
  virtual size_t num_recvs() const = 0;
  virtual const string& recv_key(size_t i) const = 0;
  virtual bool is_partial() const = 0;
  virtual bool is_last_partial_run() const = 0;
  virtual const RunGraphRequest& ToProto() const = 0;

class MutableRunGraphRequestWrapper : public RunGraphRequestWrapper

class InMemoryRunGraphRequest : public MutableRunGraphRequestWrapper
  string session_handle_;
  string graph_handle_;
  int64 step_id_;
  ExecutorOpts exec_opts_;
  gtl::InlinedVector<std::pair<string, Tensor>, 4> sends_;
  gtl::InlinedVector<string, 4> recvs_;
  bool is_partial_ = false;
  bool is_last_partial_run_ = false;
  mutable std::unique_ptr<RunGraphRequest> proto_version_;

class MutableProtoRunGraphRequest : public MutableRunGraphRequestWrapper

class ProtoRunGraphRequest : public RunGraphRequestWrapper
  const RunGraphRequest* const request_;  // Not owned.


### RunGraphResponse

class MutableRunGraphResponseWrapper
  virtual size_t num_recvs() const = 0;
  virtual const string& recv_key(size_t i) const = 0;
  virtual Status RecvValue(size_t i, TensorProto* out_tensor) = 0;
  virtual Status RecvValue(size_t i, Tensor* out_tensor) = 0;
  virtual void AddRecv(const string& key, const Tensor& value) = 0;

  // Submessages that store performance statistics about the subgraph execution, if necessary.
  virtual StepStats* mutable_step_stats() = 0;
  virtual CostGraphDef* mutable_cost_graph() = 0;
  virtual size_t num_partition_graphs() const = 0;
  virtual GraphDef* mutable_partition_graph(size_t i) = 0;
  virtual void AddPartitionGraph(const GraphDef& partition_graph) = 0;
  virtual RunGraphResponse* get_proto() = 0;

class InMemoryRunGraphResponse : public MutableRunGraphResponseWrapper
  RunGraphResponse* get_proto() override;
  gtl::InlinedVector<std::pair<string, Tensor>, 4> recvs_;
  StepStats step_stats_;
  CostGraphDef cost_graph_;
  std::vector<GraphDef> partition_graphs_;

  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;

class OwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper
  RunGraphResponse response_;

  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;
  RunGraphResponse* get_proto() override;


class NonOwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper
  RunGraphResponse* const response_;

  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;
  RunGraphResponse* get_proto() override;

class MutableRunStepResponseWrapper
  virtual size_t num_tensors() const = 0;
  virtual const string& tensor_name(size_t i) const = 0;
  virtual Status TensorValue(size_t i, Tensor* out_tensor) const = 0;
  virtual Status AddTensorFromRunGraphResponse(const string& name, MutableRunGraphResponseWrapper* run_graph_response, size_t i) = 0;
  virtual const RunMetadata& metadata() const = 0;
  virtual RunMetadata* mutable_metadata() = 0;

class InMemoryRunStepResponse : public MutableRunStepResponseWrapper
  gtl::InlinedVector<std::pair<string, Tensor>, 4> tensors_;
  RunMetadata metadata_;

class OwnedProtoRunStepResponse : public MutableRunStepResponseWrapper
  RunStepResponse response_;

class NonOwnedProtoRunStepResponse : public MutableRunStepResponseWrapper
  RunStepResponse* response_;  // Not owned.
