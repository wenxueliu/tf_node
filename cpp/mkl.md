

mkl 的目的主要是对 Node 的输出是  MKL 格式，而接受该输出的节点
不支持  MKL 格式，因此，在两者之间要加一层转换。


Status MklToTfConversionPass::InsertConversionNodeOnEdge(std::unique_ptr<Graph>* g, Edge* e)

src = e->src
dst = e->dst
如果 src->attr("T") == dst.attr("T")，增加一个 Node , conversion_node

name :Mkl2Tf
op : `_Mkl2Tf`
input : src, e->src_output()
device : src->.device()
assigned_device_name : src->assigned_device_name()
Attr :  "T": src->attr("T")
        data_format : src->attr("data_format")
        `_kernel` : mkl_op_registry::kMklOpLabel

从 g 中增加 Edge {conversion_node, 0, dst, e->dst_input()}
从 g 中删除 e

bool MklToTfConversionPass::RunPass(std::unique_ptr<Graph>* g)

遍历 g 的所有 edge，找到  src 支持  MKL, dst 不支持  MKL 的  edge, 在它们之间加入 Mkl2Tf Node

bool InsertMklToTfConversionNodes(std::unique_ptr<Graph>* g)

遍历 g 的所有 edge，找到  src 支持  MKL, dst 不支持  MKL 的  edge, 在它们之间加入 Mkl2Tf Node

Status MklToTfConversionPass::Run(const GraphOptimizationPassOptions& options)

在  pre-partitioning 阶段，graph 保存在  options.graph 中，
遍历 g 的所有 edge，找到  src 支持  MKL, dst 不支持  MKL 的  edge, 在它们之间加入 Mkl2Tf Node

在  post-partitioning 阶段，graph 保存在  options.partition_graphs 中，
遍历 g 的所有 edge，找到  src 支持  MKL, dst 不支持  MKL 的  edge, 在它们之间加入 Mkl2Tf Node

