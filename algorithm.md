
void DFS(const Graph& g, const std::function<void(Node*)>& enter, const std::function<void(Node*)>& leave);

void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter, const std::function<void(Node*)>& leave);

void ReverseDFSFrom(const Graph& g, gtl::ArraySlice<Node*> start, const std::function<void(Node*)>& enter, const std::function<void(Node*)>& leave);

void GetPostOrder(const Graph& g, std::vector<Node*>* order);
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order);

bool PruneForReverseReachability(Graph* g, std::unordered_set<const Node*> nodes);

从 nodes 中的节点开始从 g 反向遍历，将不能遍历的节点从 g 中删除

bool FixupSourceAndSinkEdges(Graph* g);

遍历 g 将 input 为空的  node 与 source node 建立 control edge, output 为空的  node 与 sink node 建立 control edge
