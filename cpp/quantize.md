
struct EdgeToConvert
  // edge is not owned here.
  const Edge* edge;
  int32 num_bits;
  bool signed_input;
  bool range_given;
  float input_min;
  float input_max;

inline bool IsGradientNode(const Graph* graph, const Node* node)

node->name 以 gradients  开头的 Node 为 Grantize 节点

bool FindType(const Graph* graph, const Node* node, bool* signed_input, bool* range_given, float* input_min, float* input_max)

对于 inode->type_string() :
if Const | Variable | VariableV2 : signed_input = true; range_given = false;
if Relu  : signed_input = false; range_given = false;
if Relu6 : signed_input = false; range_given = true; input_min = 0; input_max = 6;
if Sigmoid : signed_input = false; range_given = true; input_min = 0; input_max = 1;
if Tanh : signed_input = true; range_given = true; input_min = -1; input_max = 1;
if Reshape | ConcatV2 :
    for (const Edge* edge : node->in_edges())
      if (edge->src_output() != Graph::kControlSlot && edge->dst_input() == 0)
        FindType(graph, edge->src(), signed_input, range_given, input_min, input_max);
if Identity | MaxPool | MaxPool3D | AvgPool | AvgPool3D:
    for (const Edge* edge : node->in_edges())
      if (edge->src_output() != Graph::kControlSlot)
        FindType(graph, edge->src(), signed_input, range_given, input_min, input_max);
else
    signed_input = true;
    range_given = false;

Status FindSaveOp(const Graph* graph, Node** save_op, std::vector<const Edge*>* in_edges, bool* found)

遍历 graph 的所有 node, 找到 Op 为 SaveV2 的 node，保存在  save_op, 并设置
found, node->in_edges 保存在 in_edges

Node* FindRestoreAllOp(const Graph* graph, StringPiece save_prefix)

从 graph 的所有节点中找到名字为 save_prefix/restore_all 的节点

StringPiece GetNodeNamePrefix(const Node* node)

找到  node 的前缀, 比如 ab/bcc/c 为 ab

void FillStringTensor(Tensor* dst, const Tensor& src)

src->flat<string>() 设置 dst->flat<string>()

Status ConnectVariablesToSaveOp(Graph* graph, Node* save_op, const std::vector<const Edge*>& in_edges, const std::vector<Node*>& added_variables)

将 added_variables  加入 save_op 对应节点的 in_edges[1]->src 的属性 value 中

1. graph 增加 Node new_save_op

name: save_op->name()
op : save_op->type_string()
inputs : in_edges 的所有源节点，added_variables 中的所有节点

in_edges[0]->attr("value") 增加 added_variables 中每个元素的 name
in_edges[1]->attr("value") 增加 added_variables.size() 个元素，每个元素为 ""

2. 之后将 new_save_op + kControlSlot + save_op->out_edges() + kControlSlot 建立 control Edge 加入 graph
3. 从 graph 中删除 save_op 节点

Status AddRestoreVariableSubgraphs(Graph* graph, Node* save_op, const std::vector<const Edge*>& in_edges, const std::vector<Node*>& variables)

从 graph 中找到 op 为 name_prefix/restore_all 的 Node restore_all
遍历 variables  中每一个 var 创建如下图

    Const -> RestoreV2 -> Assign
    Const                  var
    Const -> RestoreV2 -> Assign -> restore_all
    Const                  var
    Const -> RestoreV2 -> Assign
    Const                  var

1. 找到 save_op 节点的前缀  name_prefix
2. 从 graph 中找到 op 为 name_prefix/restore_all 的 Node restore_all
3. 遍历 variables 每一个元素 var
3.1 创建节点 tensor_name
name : name_prefix/RestoreV2/tensor_names
op : Const
attr :
    dtype : DT_STRING
    value : DT_STRING, TensorShape({1})

3.2 创建节点 shape_and_slices
name : name_prefix/RestoreV2/shape_and_slices
op : Const
attr :
    dtype : DT_STRING
    value : DT_STRING, TensorShape({1})

3.3 创建节点 restore_op
name : name_prefix/RestoreV2
op : RestoreV2
input : in_edges[0]->src(), tensor_name, shape_and_slices
attr :
    dtype : DT_STRING

3.4 创建节点 assign_op
name : name_prefix/Assign
op : Assign
input : var, restore_op
attr :
    dtype : DT_STRING

3.5. 增加 control node  restore_all + kControlSlot + assign_op + kControlSlot

Status AddSaveAndRestore(Graph* graph, const std::vector<Node*>& variables)

1. 将 graph 中 SaveV2 的节点，将其与所有输出 edge 换成 control edge, 将
   variables 保存在 in_edges[1]->src 的 value 中
2. 遍历 variables  中每一个 var 创建如下图

    Const -> RestoreV2 -> Assign
    Const

Status MakeReductionAxes(Graph* graph, string name_prefix, Node* input, Node** output)

          start
input ->  rank   -> output
          delta

创建节点 start
name : name_prefix/ReductionAxes/RangeStart
op : Const
attr :  "dtype", DT_INT32
        "value", 0

创建节点 delta
name : name_prefix/ReductionAxes/RangeDelta
op : Const
attr :  "dtype", DT_INT32
        "value", 1

创建节点 rank
name : name_prefix/ReductionAxes/InputRank
op : Rank
input : input
attr :  "dtype", DT_INT32
        "value", 1

创建节点 output
name : name_prefix/ReductionAxes/ReductionAxes
op : Range
input : start, rank, delta


Status MakeExponentialMovingAverage(Graph* graph, string name_prefix, const NodeBuilder::NodeOut& input,
                                    Node* decay, Node* update_variable, Node** assign_value)

构建如下图

    one - decay             = decay_complement
    update_variable - input = value_diff
    value_diff * decay_complement = update_value
    update_variable - update_value = assign_value

    update_variable - (one-decay) * (update_variable - input) = assign_value

创建节点 one
name : name_prefix/EMA/OneConst
op : Const
attr :  "dtype", DT_FLOAT
        "value", {DT_FLOAT, TensorShape()} 1.0

创建节点 decay_complement
name : name_prefix/EMA/DecayComplement
op : Sub
input : one, decay

创建节点 value_diff
name : name_prefix/EMA/ValueDiff
op : Sub
input : update_variable, input

创建节点 update_value
name : name_prefix/EMA/UpdateValue
op : Mul
input : value_diff, decay_complement

创建节点 assign_value
name : name_prefix/EMA/EMAValue
op : Sub
input : update_variable, update_value


Status MakeInitializedEMAVariable(Graph* graph, const string& name, Node* decay, Node* init_val,
                                  std::vector<Node*>* added_variables, Node** var)


                               init_val
                                  |
      var --> is_initialized --> switch_node
       ^                        /       \
       |              output_true       output_false
       |                       |        |
       |                    ema_value   |
       |                        \      /
       +---------------------- assign_value

    one - decay             = decay_complement
    var - output_true = value_diff
    value_diff * decay_complement = update_value
    var - update_value = ema_value

    var - (one - decay) * (var - output_true) = ema_value

switch_node->output_edge[1]

创建节点 var
name : name/Variable
op : VariableV2
attr :  "dtype": DT_FLOAT
        "shape": TensorShape()


创建节点 is_initialized
name : name/IsInitialized
op : IsVariableInitialized
input : var

创建节点 switch_node
name : name/Switch
op : Switch
input : init_val, is_initialized


创建节点 assign_value
name : name/Merge
op : Merge
input : switch_node->output_edge[0], ema_value

创建节点 var
name : name/AssignValue
op : Assign
input : *var, assign_value

Status MakeEMAMinMaxVars(Graph* graph, const string& name_prefix, Node* input,
                         std::vector<Node*>* added_variables, Node** min_var, Node** max_var)

          start
input ->  rank   -> reduction_axes
          delta

min = min(reduction_axes, input)
max = max(reduction_axes, input)

                                 min
                                  |
      var --> is_initialized --> switch_node
       ^                        /       \
       |              output_true       output_false
       |                       |        |
       |                    ema_value   |
       |                        \      /
       +---------------------- assign_value

                                 max
                                  |
      var --> is_initialized --> switch_node
       ^                        /       \
       |              output_true       output_false
       |                       |        |
       |                    ema_value   |
       |                        \      /
       +---------------------- assign_value


创建节点 decay
name : name_prefix/Decay
op : Const
attr :  "dtype": DT_FLOAT
        "shape": {DT_FLOAT, TensorShape()};

创建节点 min
name : name_prefix/Min
op : Min
input : input, reduction_axes

创建节点 max
name : name_prefix/Max
op : Min
input : input, reduction_axes

Status MakeInputMinMax(Graph* graph, const string& name_prefix, const EdgeToConvert& edge,
                       std::vector<Node*>* added_variables, Node** input_min, Node** input_max)
if (edge.range_given)

创建节点 input_min
name : name_prefix/InputMin
op : Const
attr :  "dtype": DT_FLOAT
        "value": (DT_FLOAT, TensorShape()); edge.input_min;

创建节点 input_max
name : name_prefix/InputMax
op : Const
attr :  "dtype": DT_FLOAT
        "value": (DT_FLOAT, TensorShape()); edge.input_max;

else
    MakeEMAMinMaxVars(graph, name_prefix, edge.edge->src(), added_variables, input_min, input_max)

Status MakeQuantizeOp(Graph* graph, const string& name_prefix,
                      const string& quant_op_type, const EdgeToConvert& edge,
                      std::vector<Node*>* added_variables, Node** convert_node)

1. MakeInputMinMax(graph, name_prefix, edge, added_variables, &input_min, &input_max)
2.
if (quant_op_type == "QuantizeAndDequantizeV2")

创建节点 convert_node
name : name_prefix/quant_op_type
op : QuantizeAndDequantizeV2
input : edge->src(), input_min, input_max
attr :
    signed_input: edge.signed_input
    num_bits: edge.num_bits
    range_given: true

else if (quant_op_type == "FakeQuantWithMinMaxVars")

创建节点 convert_node
name : name_prefix/quant_op_type
op : QuantizeAndDequantizeV2
input : edge->src(), input_min, input_max
attr :
    num_bits: edge.num_bits

Status ProcessTargetEdges(Graph* graph, const string& quant_op_type, const std::vector<EdgeToConvert>& target_edges)
  for (const EdgeToConvert edge : target_edges)
    string name_prefix = edge.edge->src()->name();
    auto iter = name_index.find(name_prefix);
    if (iter == name_index.end()) {
      MakeQuantizeOp(graph, name_prefix, quant_op_type, edge, &added_variables, &convert_node)
      name_index[name_prefix] = convert_node;
    } else {
      convert_node = iter->second;
    }
    graph->AddEdge(convert_node, 0, edge.edge->dst(), edge.edge->dst_input());
    graph->RemoveEdge(edge.edge);
  AddSaveAndRestore(graph, added_variables)

Status DoQuantizeTraining(int32 num_bits, const string& quant_op_type, Graph* graph)

TODO

Status DoQuantizeTrainingOnGraphDef(const GraphDef& input_graphdef, int32 num_bits, const string& quant_op_type, GraphDef* result_graphdef)

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  ConvertGraphDefToGraph(opts, input_graphdef, &graph);
  DoQuantizeTraining(num_bits, quant_op_type, &graph);
  graph.ToGraphDef(result_graphdef);

Status DoQuantizeTrainingOnSerializedGraphDef(const string& input_graph_string, int32 num_bits,
                                              const string& quant_op_type, string* result_graph_string)

  GraphDef input_graphdef;
  ParseProtoUnlimited(&input_graphdef, input_graph_string)
  GraphDef output_graphdef;
  DoQuantizeTrainingOnGraphDef(input_graphdef, num_bits, quant_op_type, &output_graphdef);
  output_graphdef.SerializeToString(result_graph_string)
