
在任何一个系统中调试的作用都是必须的，好的调试系统可以
让问题定位非常容易，从而增加用户使用粘度

## 数据结构

typedef std::function<std::unique_ptr<DebuggerStateInterface>(const DebugOptions& options)> DebuggerStateFactory;
DebuggerStateFactory* DebuggerStateRegistry::factory_ = new DebuggerStateFactory(factory);;

typedef std::function<std::unique_ptr<DebugGraphDecoratorInterface>(const DebugOptions& options)> DebugGraphDecoratorFactory;
DebugGraphDecoratorFactory* DebugGraphDecoratorRegistry::factory_ = nullptr;


class DebuggerStateInterface
  virtual Status PublishDebugMetadata(
      const int64 global_step, const int64 session_run_index,
      const int64 executor_step_index, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes) = 0;

class DebugGraphDecoratorInterface
  virtual Status DecorateGraph(Graph* graph, Device* device) = 0;
  virtual Status PublishGraph(const Graph& graph, const string& device_name) = 0;


class DebuggerStateRegistry
  static void RegisterFactory(const DebuggerStateFactory& factory);
  static Status CreateState(const DebugOptions& debug_options, std::unique_ptr<DebuggerStateInterface>* state);
  static DebuggerStateFactory* factory_;


class DebugGraphDecoratorRegistry
  static void RegisterFactory(const DebugGraphDecoratorFactory& factory);
  static Status CreateDecorator(const DebugOptions& options, std::unique_ptr<DebugGraphDecoratorInterface>* decorator);
  static DebugGraphDecoratorFactory* factory_;


## 源码分析

void DebuggerStateRegistry::RegisterFactory(const DebuggerStateFactory& factory) //factory_ = new DebuggerStateFactory(factory);

Status DebuggerStateRegistry::CreateState(const DebugOptions& debug_options, std::unique_ptr<DebuggerStateInterface>* state) //state = factory_(debug_options)

void DebugGraphDecoratorRegistry::RegisterFactory(const DebugGraphDecoratorFactory& factory) //factory_ = new DebugGraphDecoratorFactory(factory);

Status DebugGraphDecoratorRegistry::CreateDecorator(const DebugOptions& options, std::unique_ptr<DebugGraphDecoratorInterface>* decorator) //decorator = (*factory_)(options);
