
static Status FeedInputs(Graph* g, const DeviceAttributes& device_info,
                         const gtl::ArraySlice<string>& fed_outputs, bool use_function_convention,
                         subgraph::NameIndex* name_index, DataTypeVector* out_feed_types)

fed_outputs：保存  Node 的 name 的列表
name_index : 保存  Node 的 name 和 Node 的映射关系
output_type : 保存 fed_outputs 中 name 在 name_index 中 对应 node 的 out_edge 的类型

1. 遍历 fed_outputs 的每一个元素 name
1.1 解析  name 为  id. 比如 abc:1 会解析为 id.first = abc, id.second=1
1.2 从 name_index 中找到 id.name 对应的节点 n
1.3
if use_function_convention 为 false
创建节点 recv_node
name:_recv_${id.first}_${id.second}_
op : `_Recv`
assign_device_name :device_info.name(
attr :
    tensor_type: BaseType(n->output_type(id.second))
    tensor_name: name
    send_device: device_info.name()
    recv_device: device_info.name()
    send_device_incarnation: static_cast<int64>(device_info.incarnation()))
    client_terminated: true
else
创建节点 recv_node
name:_arg_${id.first}_${id.second}_
op : `_Arg`
assign_device_name : device_info.name(
attr :
    T: n->output_type(id.second)
    index: static_cast<int32>(i)

2. name_index[recv_node->name()] = recv_node
3. 在 g 的 source 与  recv_node 建立 control edge
4. 遍历 n 的输出节点，将满足条件的 edge 删除，用  recv_node 替代


Status FetchOutputs(Graph* g, const DeviceAttributes& device_info,
                    const gtl::ArraySlice<string>& fetch_outputs, bool use_function_convention, NameIndex* name_index,
                    std::vector<Node*>* out_fetch_nodes, DataTypeVector* out_fetch_types)

out_fetch_nodes : fetch_outputs 中 name 对应的 send_node
out_feed_types : fetch_outputs 中 name 对应的 type

1. 遍历 fetch_outputs 的每一个元素 name
1.1 解析 name 为 id. 比如 abc:1 会解析为 id.first = abc, id.second=1
1.2 从 name_index 中找到 id.name 对应的节点 n
1.3
if use_function_convention 为 false
创建节点 send_node
name:_send_${id.first}_${id.second}_
op : `_Send`
input : n
assign_device_name :device_info.name(
attr :
    tensor_name: name
    send_device: device_info.name()
    recv_device: device_info.name()
    send_device_incarnation: static_cast<int64>(device_info.incarnation()))
    client_terminated: true
else
创建节点 send_node
name:_retval_${id.first}_${id.second}_
op : `_Retval`
assign_device_name : device_info.name(
attr :
    T: n->output_type(id.second)
    index: static_cast<int32>(i)

2. name_index[recv_node->name()] = send_node
3. 在 g 的 source 与 send_node 建立 control edge
4. out_fetch_nodes 加入 send_node
5. out_feed_types 加入 n 的 id.second 对应的 type

static bool AddNodeToTargets(const string& node_or_tensor_name, const subgraph::NameIndex& name_index, std::unordered_set<const Node*>* targets)

如果 node_or_tensor_name 在 name_index 中存在，加入 targets

static Status PruneForTargets(Graph* g, const subgraph::NameIndex& name_index, const std::vector<Node*>& fetch_nodes, const gtl::ArraySlice<string>& target_nodes)

1. 将 fed_outputs 和 target_nodes 中对应的节点加入 targets
2. 从 targets 中的节点开始从 g 反向遍历， 将不能遍历的节点从 g 中删除
3. 遍历 g 将 input 为空的  node 与 source node 建立 control edge, output 为空的  node 与 sink node 建立 control edge, 

Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs, const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names, const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata)

fetch_outputs 的元素在 fed_outputs 中不能存在
out_metadata->feed_types 保存 fed_outputs 对应的类型
out_metadata->fetch_types 保存 fetch_outputs 对应的类型

修改 g 中 fed_outputs 和 fetch_outputs  对应的节点，以 fetch_outputs 和 target_node_names 为基础更新 g

1. 将 g 的所有节点加入 name_index(key: node->name, value: node)
2. 将 fed_outputs 中的 name 对应的节点修改为 recv_node，更新 name_index，并删除部分 edge, 用新的节点替代。具体参考 FeedInputs
3. 将 fetch_outputs 中的 name 对应的节点修改为 send_node，更新 name_index，用新的节点替代， 将 send_node 加入 fetch_nodes
4. 将 fetch_nodes 和 target_node_names  组合之后，从 g 反向遍历，将不能遍历到的节点从 g 删除
5. 遍历 g 将 input 为空的 node 与 source node 建立 control edge, output 为空的  node 与 sink node 建立 control edge
