

## 子表达式子消除

common-subexpression elimination


static void FillInputs(const Node* n, gtl::InlinedVector<Node*, 4>* control_edges, gtl::InlinedVector<std::pair<Node*, int>, 4>* in)

将  n 的 in_edges 初始化  control_edges, in, 并排序。

size_t OptimizerCSE::NodeHash(const Node* n)

计算 n 的哈希值，将 n 的各个属性转换为字符串，之后进行 Hash64

static bool HasRefInput(const Node* n)

n 的输入类型中是否有一个是 Ref 类型

bool OptimizerCSE::Equivalent(const Node* a, const Node* b, AttrSlice::Scratch* scratch)

a 和 b 要相同必须满足的条件
1. a 与  b 的 type_string 相同
2. a->op_def().is_stateful() == false
3. a 和 b 的  input 没有一个是 Ref 类型
4. a.attr 与 b.attr 相同
5. a.num_inputs == b.num_inputs
6. a 和  b 的  in_edges  相同

bool OptimizerCSE::Optimize( const std::function<bool(const Node*)>& consider_fn)


反向遍历 g_， 计算节点 n 的 hash 值 h
1. 如果在 available 中找不到，就加入 available
2. 如果找到了，并且 n 与 available 中对应的节点 candidate 相同，遍历 n 的所有
   out_edges, 用  candidate 替代 edge->src, 之后将  n 从 g_ 中删除
3. 如果找到了，n 与 available 中对应的节点 candidate 不同，什么也不做

其中  consider_fn 定义了是否考虑某个节点，返回 true, 表示考虑，返回 false 表示不考虑

比如

 A * B -> C
 A * B -> D

其中 C 和 D 满足  Equivalent 定义，那么经过优化之后就变为

A * B -> D

其中 C 和 D 满足  Equivalent 定义，那么经过优化之后(忽略 Mul 操作)就变为

 A * B -> C
 A * B -> D

例 2

 A * B -> C
 A * B -> D
 A * B -> E

其中 C 和 D  和 E 满足  Equivalent 定义，Optimize 那么经过优化之后就变为

A -> E
B -> E

例 3

 A * B -> C
 A * B -> D
 C * D -> E

其中 C 和 D  满足  Equivalent 定义，那么经过 Optimize 优化之后就变为

A * B -> D
D * D -> E

例 4

 A * B -> C
 A + B -> D

其中 C 和 D  满足  Equivalent 定义，那么经过 Optimize 优化之后就变为

A * B -> C
A + B -> D

例 5

 A * B -> C

其中 A 和 B  满足  Equivalent 定义，那么经过 Optimize 优化之后就变为

 B * B -> C
