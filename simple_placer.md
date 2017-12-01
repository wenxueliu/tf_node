
## 数据结构

class SimplePlacer {
  typedef std::unordered_map<string, int> NodeNameToIdMap;
  Graph* const graph_;
  const DeviceSet* const devices_;
  const SessionOptions* options_;
  const bool log_device_placement_;

class ColocationGraph
  Graph* const graph_;  // Not owned.
  std::vector<Member> members_;
  const DeviceSet* device_set_;  // Not owned.
  const std::vector<DeviceType> device_types_;
  const bool allow_soft_placement_;

  struct Member
    // The id of the node that is the parent of this one, or its own
    // id if it is a root. parent <= 0 indicates that this member is invalid.
    int parent = -1;
    // A proxy for the depth of the tree that is used to prefer
    // connecting smaller trees to larger trees when merging disjoint
    // sets.
    int rank = 0;

    // The intersection of all device types supported by this node,
    // and those of all of its children, in priority order
    // of the preferred device.
    DeviceTypeVector supported_device_types;

    // The merged form of the device requested for this node, with
    // those of all of its children.
    DeviceNameUtils::ParsedName device_name;

    // If this node is a root, stores a list of Devices to which this node
    // and all of its children have been assigned, or nullptr if this
    // has not yet been computed.
    std::vector<Device*> possible_devices; //存储某天节点及其父节点可以运行的 device 的集合

## 源码分析

Status SimplePlacer::Run()


std::vector<Device*> FilterSupportedDevices(const std::vector<Device*>& devices, const DeviceTypeVector& supported_device_types)

将 devices 中类型在 supported_device_types 中的 device 排序后返回

Status SimplePlacer::Run()

1. 创建 colocation_graph 对象
2. colocation_graph.InitializeMembers()
3. colocation_graph.ColocateAllNodes()


Status ColocationGraph::ColocateAllNodes()

Status ColocateNodeToGroup(
    std::unordered_map<StringPiece, const Node*, StringPiece::Hasher>* colocation_group_root,
    Node* node, StringPiece colocation_group)


Status ColocateNodes(const Node& x, const Node& y)

Status ColocateNodes(const Node& x, int x_root, const Node& y, int y_root)

Status InitializeMember(const Node& node, Member* member)

Status GetDevicesForNode(Node* node, std::vector<Device*>** possible_devices)

int FindRoot(int node_id)

递归找到 node_id 的父节点
