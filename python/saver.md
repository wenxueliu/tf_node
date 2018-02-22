

checkpoint 主要用于训练模型参数的保存与恢复


## BaseSaverBuilder

def save_op(self, filename_tensor, saveables)

解析出 saveables 中的 tensor, tensor_name, tensor_slice
将 tensor, tensor_name, tensor_slice 写到 filename_tensor 中

```
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices, tensors)
```

参考 tensorflow/core/ops/io_ops.cc tensorflow/core/kernels/save_restore_v2_ops.cc

def restore_op(self, filename_tensor, saveable, preferred_shard)

遍历 saveable.specs，从 filename_tensor 读出，并加入 tensors， 返回 tensors

参考 tensorflow/core/ops/io_ops.cc tensorflow/core/kernels/save_restore_v2_ops.cc

def sharded_filename(self, filename_tensor, shard, num_shards)

返回 filename_tensor-shard-of-num_shards 字符串

参考 tensorflow/core/ops/io_ops.cc tensorflow/core/kernels/save_op.cc

def \_AddSaveOps(self, filename_tensor, saveables)

解析出 saveables 中的 tensor, tensor_name, tensor_slice
将 tensor, tensor_name, tensor_slice 写到 filename_tensor 中

def \_AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device)

1. 遍历 per_device，将 saveables 写入临时文件夹
2. 将临时文件夹合并，写入新的文件夹

```
for shard, (device, saveables) in enumerate(per_device)
    filename = "${checkpoint_prefix}_temp_${uuid}/part-${shard}-of-${num_shards})"
    sharded_prefixes.append(filename)
    sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
```

def \_AddShardedSaveOps(self, filename_tensor, per_device)

\_AddShardedSaveOpsForV2 的区别在于没有合并这一步操作


def \_AddRestoreOps(self, filename_tensor,
            saveables, restore_sequentially,
            reshape, preferred_shard=-1, name="restore_all")

遍历 saveables 从 filename_tensor 中读取对应的 Operation，之后返回所有 Operation 的 list

def \_AddShardedRestoreOps(self, filename_tensor, per_device, restore_sequentially, reshape)

遍历 per_device， 从 filename_tensor 读取每个 saveables 的内容加入列表，之后返回该列表

def \_GroupByDevices(self, saveables)

返回  saveables 对应的 per_device(key 为 pydev.canonical_name(saveable.spec.tensor.device), value 为o)

def OpListToDict(op_list)

解析 op_list 将其转为 dict

def \_ValidateAndSliceInputs(self, names_to_saveables)

遍历 names_to_saveables， 校验并将其 value 对应的 saveable 加入列表 saveables，返回该列表

def \_AddSaveable(self, saveables, seen_ops, saveable)

如果 saveable 在  seen_ops 中，抛出异常，否则加入  saveables 和 seen_ops

def build(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5,
        keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False,
        filename="model")

将  names_to_saveables 解析保存在  saveables 中
如果 sharded 为 True, 将 saveables 以 device 划分，之后构造 SaveDef
如果 sharded 为 False, 之后构造 SaveDef



def \_get_saver_or_default()

如果 ops.GraphKeys.SAVERS 中元素个数
1. 大于 1，抛异常
2. 等于 1，返回该 saver
3. 等于 0, 创建一个 Saver

def generate_checkpoint_state_proto(save_dir,
                                    model_checkpoint_path,
                                    all_model_checkpoint_paths=None)

获取相对于  save_dir 的文件路径，创建 CheckpointState 对象

def update_checkpoint_state(save_dir,
                            model_checkpoint_path,
                            all_model_checkpoint_paths=None,
                            latest_filename=None)


def \_update_checkpoint_state(save_dir,
                             model_checkpoint_path,
                             all_model_checkpoint_paths=None,
                             latest_filename=None,
                             save_relative_paths=False)

将 save_relative_paths 或 model_checkpoint_paths  构造 CheckpointState 对象之后, 写入 save_dir/latest_filename 文件

def get_checkpoint_state(checkpoint_dir, latest_filename=None)

从  checkpoint_dir/latest_filename 中读取 CheckpointState 并返回


### SaveSpec

self.tensor
self.slice_spec
self.name

### SaveableObject

self.op
self.spec
self.name

### VariableSaveable 继承自  SaveableObject

def restore(self, restored_tensors, restored_shapes)

self.op = restored_tensors

### ResourceVariableSaveable 继承自  SaveableObject

def restore(self, restored_tensors, restored_shapes)

调用 cpp 函数 AssignVariableOp

参考  core/kernels/resource_variable_ops.cc

## Saver

self._var_list = var_list
self._reshape = reshape
self._sharded = sharded
self._max_to_keep = max_to_keep
self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
self._name = name
self._restore_sequentially = restore_sequentially
self.saver_def  : SaveDef 对象，由  self._builder.build() 创建
self._builder : 创建  SaveDef 的 buider，默认  BaseSaverBuilder
self._builder = builder
self._is_built = False
self._allow_empty = allow_empty
self._is_empty = None
self._write_version = write_version
self._pad_step_number = pad_step_number
self._filename = filename
self._save_relative_paths = save_relative_paths
self._is_built : build() 是否调用过

def build()

初始化 self.save_def

def \_MetaGraphFilename(self, checkpoint_filename, meta_graph_suffix="meta")

返回 checkpoint_filename  对应的 meta 路径，比如 model.ckpt-123456-?????-of-00005 变为 model.ckpt-123456.meta

ops.GraphKeys.GLOBAL_VARIABLES 和 ops.GraphKeys.SAVEABLE_OBJECTS

def \_MaybeDeleteOldCheckpoints(self, latest_save_path, meta_graph_suffix="meta")

这里实现存疑? 貌似有问题

def as_saver_def(self)

返回  self.save_def

def to_proto(self, export_scope=None)

将 self.save_def 转为 SaveDef 对象

def from_proto(saver_def, import_scope=None)

用  save_def 创建 Saver 对象

def last_checkpoints(self)

返回 self._last_checkpoints 对应的 filename

def set_last_checkpoints(self, last_checkpoints)

last_checkpoints 加入  self._last_checkpoints

def set_last_checkpoints_with_time(self, last_checkpoints_with_time)

self._last_checkpoints = last_checkpoints_with_time

def recover_last_checkpoints(self, checkpoint_paths)

遍历 checkpoint_paths， 返回 (文件名,修改时间) 组成的列表

def save(self,
         sess,
         save_path,
         global_step=None,
         latest_filename=None,
         meta_graph_suffix="meta",
         write_meta_graph=True,
         write_state=True)

1. model_checkpoint_path = sess.run(self.saver_def.save_tensor_name)
2. if self._is_empty and write_state, 构造 SaveDef 对象并写入 save_path
3. if write_meta_graph, 将  meta_graph_def 写入 ${save_path}[-${global_step}]
4. 如果 self._is_empty 为 False,  返回 model_checkpoint_path，否则返回 None
5. 返回 model_checkpoint_path

def export_meta_graph(self,
                      filename=None,
                      collection_list=None,
                      as_text=False,
                      export_scope=None,
                      clear_devices=False,
                      clear_extraneous_savers=False)

def restore(self, sess, save_path)

sess.run(self.saver_def.restore_op_name )

def \_add_collection_def(meta_graph_def, key, export_scope=None)

meta_graph_def 加到 key



def get_checkpoint_mtimes(checkpoint_paths)

遍历 checkpoint_paths 中没有元素对应文件的修改时间，返回所有文件修改时间列表

对于 checkpoint_paths 中的元素 file, 首先查找 file.index 文件的修改时间，如果找不到，再查找 file 的修改时间

def latest_checkpoint(checkpoint_dir, latest_filename=None)

读  checkpoint_dir/latest_filename 文件，找到 CheckpointState 对象 chpt，ckpt.model_checkpoint_path

def import_meta_graph(meta_graph_or_file, clear_devices=False, import_scope=None, kwargs)

meta_graph.import_scoped_meta_graph(meta_graph_def, clear_devices=clear_devices, import_scope=import_scope, kwargs)

TODO
返回  meta_graph 对应的 Saver

def export_meta_graph(filename=None,
                      meta_info_def=None,
                      graph_def=None,
                      saver_def=None,
                      collection_list=None,
                      as_text=False,
                      graph=None,
                      export_scope=None,
                      clear_devices=False,
                      clear_extraneous_savers=False,
                      kwargs)
TODO
调用 meta_graph.export_scoped_meta_graph










## SavedModelBuilder

tag 用于标记 meta_graph_def

self._saved_model = saved_model_pb2.SavedModel()
self._export_dir

def \_save_and_write_assets(self, assets_collection_to_add=None)

1. 遍历 assets_collection_to_add 每个元素构造 AssetFileDef 对象，并加入 constants.ASSETS_KEY
2. 遍历 assets_collection_to_add 中每个 Tensor 对应的文件名，将其拷贝到 self._export_dir/constants.ASSETS_DIRECTORY

def \_maybe_add_legacy_init_op(self, legacy_init_op=None)

将 legacy_init_op 加入 constants.LEGACY_INIT_OP_KEY

def \_add_main_op(self, main_op)

将  main_op 加入 constants.MAIN_OP_KEY

def \_tag_and_add_meta_graph(self, meta_graph_def, tags, signature_def_map)

用  tags 和  signature_def_map 初始化 meta_graph_def，之后加入 self._saved_model.meta_graphs

def \_validate_tensor_info(self, tensor_info)

tensor_info 不为空，tensor_info.name 不为空，tensor_info.dtype 不为 types_pb2.DT_INVALID

def \_validate_signature_def_map(self, signature_def_map)

校验 signature_def_map 中的每个元素的  input 和  output


def add_meta_graph(self, tags, signature_def_map=None,
                assets_collection=None, legacy_init_op=None,
                clear_devices=False, main_op=None)

1. 校验 signature_def_map 的  input, output 都是合法的
2. 遍历 assets_collection 每个元素构造 AssetFileDef 对象，并加入 constants.ASSETS_KEY
3. 遍历 assets_collection 中每个 Tensor 对应的文件名，将其拷贝到 self._export_dir/constants.ASSETS_DIRECTORY
4. 如果 main_op 不为空，将 main_op 加入 constants.MAIN_OP_KEY，否则将 legacy_init_op 加入 constants.LEGACY_INIT_OP_KEY
5. 创建一个 Saver 对象，导出 meta_graph_def，用 tags 和 signature_def_map 初始化 meta_graph_def


def add_meta_graph_and_variables(self,
                                 sess,
                                 tags,
                                 signature_def_map=None,
                                 assets_collection=None,
                                 legacy_init_op=None,
                                 clear_devices=False,
                                 main_op=None)

1. 校验 signature_def_map 的  input, output 都是合法的
2. 遍历 assets_collection 每个元素构造 AssetFileDef 对象，并加入 constants.ASSETS_KEY
3. 遍历 assets_collection 中每个 Tensor 对应的文件名，将其拷贝到 self._export_dir/constants.ASSETS_DIRECTORY
4. 如果 main_op 不为空，将 main_op 加入 constants.MAIN_OP_KEY，否则将 legacy_init_op 加入 constants.LEGACY_INIT_OP_KEY
5. 创建一个 Saver 对象，self._export_dir/constants.VARIABLES_DIRECTORY/constants.VARIABLES_FILENAME
6. 导出 meta_graph_def，用 tags 和 signature_def_map 初始化 meta_graph_def

def save(self, as_text=False)

if as_text, 将 str(self._saved_model) 写入 self._export_dir/constants.SAVED_MODEL_FILENAME_PBTXT

def \_maybe_save_assets(assets_collection_to_add=None)

将 assets_collection_to_add 中每个元素构造一个 AssetFileDef 加入 constants.ASSETS_KEY，并返回每个元素对应的 filename

1. 校验 assets_collection_to_add 中每个元素
2. 对于每个元素 e, 将 e.op.get_attr("value").string_val 和 e 构建一个 AssetFileDef 加入 constants.ASSETS_KEY
3. 将 e.op.get_attr("value").string_val 加入 asset_source_filepath_list
返回 asset_source_filepath_list

对 assets_collection_to_add 的要求

1. 每个元素必须是 Tensor 对象
2. op.type 必须是 Const
3. dtype 必须是 string
4. 每个元素的 op.get_attr("value").string_val 必须只有一个

def \_add_asset_to_collection(asset_filename, asset_tensor)

用 asset_filename 和 asset_tensor 构造一个 asset_any_proto 加入 constants.ASSETS_KEY


```
message AssetFileDef {
  TensorInfo tensor_info = 1;
  string filename = 2;  // 不包含路径前缀，比如 /tmp/path/vocab.txt 时间为 vocab.txt
}

message TensorInfo {
  // For sparse tensors, The COO encoding stores a triple of values, indices,
  // and shape.
  message CooSparse {
    // The shape of the values Tensor is [?].  Its dtype must be the dtype of
    // the SparseTensor as a whole, given in the enclosing TensorInfo.
    string values_tensor_name = 1;

    // The indices Tensor must have dtype int64 and shape [?, ?].
    string indices_tensor_name = 2;

    // The dynamic logical shape represented by the SparseTensor is recorded in
    // the Tensor referenced here.  It must have dtype int64 and shape [?].
    string dense_shape_tensor_name = 3;
  }
  oneof encoding {
    // For dense `Tensor`s, the name of the tensor in the graph.
    string name = 1;
    // There are many possible encodings of sparse matrices
    // (https://en.wikipedia.org/wiki/Sparse_matrix).  Currently, TensorFlow
    // uses only the COO encoding.  This is supported and documented in the
    // SparseTensor Python class.
    CooSparse coo_sparse = 4;
  }
  DataType dtype = 2;
  // The static shape should be recorded here, to the extent that it can
  // be known in advance.  In the case of a SparseTensor, this field describes
  // the logical shape of the represented tensor (aka dense_shape).
  TensorShapeProto tensor_shape = 3;
}
```


## Loader

def \_parse_saved_model(export_dir)

1. 如果 export_dir/constants.SAVED_MODEL_FILENAME_PBTXT 存在，读取，初始化  SavedModel 并返回，否则继续步骤 2
2. 如果 export_dir/constants.SAVED_MODEL_FILENAME_PB 读取，初始化 SavedModel 并返回，否则继续步骤 3
3. 抛异常

def \_get_asset_tensors(export_dir, meta_graph_def_to_load)

遍历 collection_def[constants.ASSETS_KEY].any_list.value，初始化 AssetFileDef 对象 asset_proto，
将 asset_proto.tensor_info.name : asset_proto.filename 映射关系保存在 dict，之后返回该 dict

def \_get_main_op_tensor(meta_graph_def_to_load)

1. meta_graph_def_to_load.collection_def[constants.MAIN_OP_KEY] 只有一个元素
2. 返回 constants.MAIN_OP_KEY 中的元素

def \_get_legacy_init_op_tensor(meta_graph_def_to_load)

1. meta_graph_def_to_load.collection_def[constants.LEGACY_INIT_OP_KEY] 只有一个元素
2. 返回 constants.LEGACY_INIT_OP_KEY 中的元素

def maybe_saved_model_directory(export_dir)

export_dir/constants.SAVED_MODEL_FILENAME_PB 或 export_dir/constants.SAVED_MODEL_FILENAME_PBTXT

def load(sess, tags, export_dir, saver_kwargs)

1. 从 export_dir 读取  SavedModel 得到 saved_model
2. 遍历 saved_model.meta_graphs 找到与 tags 一致的 saved_model_def
3. 从 saved_model_def 初始化 saver，从 export_dir/constants.VARIABLES_DIRECTORY/constants.VARIABLES_FILENAME 恢复变量到 sess
4. 遍历 collection_def[constants.ASSETS_KEY].any_list.value，初始化 AssetFileDef 对象 asset_proto，
将 asset_proto.tensor_info.name : asset_proto.filename 映射关系保存在 asset_tensors_dictionary
5. 如果 main_op 存在，从 constants.MAIN_OP_KEY  读取 main_op, 调用 sess.run(main_op, feed_dict = asset_tensors_dictionary)，否则继续步骤 6
6. 从 constants.LEGACY_INIT_OP_KEY 读取 legacy_init_op, 调用 sess.run(legacy_init_op, feed_dict = asset_tensors_dictionary)


### signature_def


```
message SignatureDef {
  map<string, TensorInfo> inputs = 1;
  map<string, TensorInfo> outputs = 2;
  string method_name = 3;
}
```

def build_signature_def(inputs=None, outputs=None, method_name=None)

input, output, method_name 初始化 signature_def

def regression_signature_def(examples, predictions)

用 examples 作为 input, predictions 作为 output 初始化 signature_def

def classification_signature_def(examples, classes, scores)

examples 作为 input, classes, scores 作为 output 初始化 signature_def

def predict_signature_def(inputs, outputs)

inputs, outputs 初始化 signature_def


def build_tensor_info(tensor)

将  tensor 转化为 tensor_info

def get_tensor_from_tensor_info(tensor_info, graph=None, import_scope=None)

将  tensor_info 转化为 tensor
