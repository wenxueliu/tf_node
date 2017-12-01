### 常用 Operation

#### ParallelConcat

例子

```
 'x' is [[1, 4]]
 'y' is [[2, 5]]
 'z' is [[3, 6]]
parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
```
Concat 在开始之前需要所有的 input 被计算好，但是不需要 input 的 shape 已知
而 ParallelConcat 只有输入可用就会被拷贝到输出

#### Pack


例子

```
'x' is [1, 4]
'y' is [2, 5]
'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

TODO: 这里注释和例子还是没有看懂

#### Unpack

例子

```
'x' is [1, 4]
'y' is [2, 5]
'z' is [3, 6]
unpack([[1, 4], [2, 5], [3, 6]]) => [1,4] [2,3] [3,6] # Pack along first dim.
unpack([[1, 2, 3], [4, 5, 6]], axis=1) => [1,4] [2,5] [3,6]
```

#### Concat

#### ConcatV2

#### _MklConcatV2

#### ConcatOffset

例子
```
 'x' is [2, 2, 7]
 'y' is [2, 3, 7]
 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```

#### Split

#### SplitV

#### Const

#### ImmutableConst

#### ZerosLike

#### OnesLike

#### Diag

例子
```
 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```

#### Dig

#### DiagPart
```
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
```

#### Invert

对输入元素进行翻转

#### PopulationCount

计算输入元素中 bit 为 1 的个数

#### BitwiseAnd

两个元素的 bit 进行与操作

#### BitwiseOr

两个元素的 bit 进行或操作

#### BitwiseXor

两个元素的 bit 进行异或操作

#### UniformCandidateSampler

TODO

#### LogUniformCandidateSampler

TODO

#### LearnedUnigramCandidateSampler

TODO

#### ThreadUnsafeUnigramCandidateSampler

#### FixedUnigramCandidateSampler

#### AllCandidateSampler

#### ComputeAccidentalHits

### 控制流相关

#### Switch

Input
    data: T
    pred: bool

Output
    output_false: T
    output_true: T

如果 pred 为 True, T 到 output_true
如果 pred 为 False, T 到 output_false

#### RefSwitch

Input
    data: T
    pred: bool

Output
    output_false: T
    output_true: T

如果 pred 为 True, T 到 output_true
如果 pred 为 False, T 到 output_false

注：与 Switch 的区别在于允许没有初始化

#### RefSelect

Input
    index: int32
    inputs: Ref(N * T)

Output
    output: Ref(T)

将 inputs 中的第  index 赋值给 output

#### Merge

Input
    inputs: N * T

Ouput
    output: T
    value_index: int32

将 inputs 合并到  output, value_index 表示当前可用的  inputs 的索引

与 Switch 一起创建一个分支

#### RefMerge

Input
    inputs: N * T

Ouput
    output: T
    value_index: int32

将 inputs 合并到  output, value_index 表示当前可用的  inputs 的索引

与 Switch 一起创建一个分支

#### Enter

Input
    data: T

Ouput
    data: T

Attr
    frame_name: string
    is_constant: bool = false  //是否支持交换律，即 Add(a, b) 与  Add(b, a) 相同
    parallel_iterations: int = 10

找到或创建一个子帧(child frame), 让 data 对于子帧是可用的

什么是子帧: TODO

与 Exit 一起创建一个循环

#### RefEnter

Input
    data: T

Ouput
    data: T

Attr
    frame_name: string
    is_constant: bool = false //是否支持交换律，即 Add(a, b) 与  Add(b, a) 相同
    parallel_iterations: int = 10

找到或创建一个子帧(child frame), 让 data 对于子帧是可用的

什么是子帧: TODO

与 Exit 一起创建一个循环

#### Exit

Input
    data: T

Ouput
    output: T

从目前的帧回到母帧（parent frame）

与 Entry 一起创建一个循环

#### RefExit

Input
    data: T

Ouput
    output: T

从目前的帧回到母帧（parent frame）

与 Entry 一起创建一个循环

#### NextIteration

Input
    data: T

Ouput
    output: T

在下一次迭代中，data 可用

#### RefNextIteration

Input
    data: T

Ouput
    output: T

在下一次迭代中，data 可用

#### LoopCond

Input
    input: bool

Ouput
    output: bool

循环结束的条件

#### ControlTrigger

控制流触发

#### Abort

Attr
    error_msg: string = ''
    exit_without_error: bool = false


### CTC

#### CTCLoss

#### CTCGreedyDecoder

#### CTCBeamSearchDecoder

### dataFlow

DynamicPartition
DynamicStitch
RandomShuffleQueue
RandomShuffleQueueV2
FIFOQueue
FIFOQueueV2
PaddingFIFOQueue
PaddingFIFOQueueV2
PriorityQueue
PriorityQueueV2
FakeQueue
QueueEnqueue
QueueEnqueueV2
QueueEnqueueMany
QueueEnqueueManyV2
QueueDequeue
QueueDequeueV2
QueueDequeueMany
QueueDequeueManyV2
QueueDequeueUpTo
QueueDequeueUpToV2
QueueClose
QueueCloseV2
QueueIsClosed
QueueIsClosedV2
QueueSize
QueueSizeV2
AccumulatorNumAccumulated
AccumulatorSetGlobalStep
ConditionalAccumulator
AccumulatorApplyGradient
AccumulatorTakeGradient
SparseConditionalAccumulator
SparseAccumulatorApplyGradient
SparseAccumulatorTakeGradient
StackV2
StackPushV2
StackPopV2
StackCloseV2
Stack
StackPush
StackPop
StackClose
TensorArrayV3
TensorArrayGradV3
TensorArrayWriteV3
TensorArrayReadV3
TensorArrayGatherV3
TensorArrayScatterV3
TensorArrayConcatV3
TensorArraySplitV3
TensorArraySizeV3
TensorArrayCloseV3
TensorArray
TensorArrayV2
TensorArrayGrad
TensorArrayGradV2
TensorArrayWrite
TensorArrayWriteV2
TensorArrayRead
TensorArrayReadV2
TensorArrayPack
TensorArrayUnpack
TensorArrayGather
TensorArrayGatherV2
TensorArrayScatter
TensorArrayScatterV2
TensorArrayConcat
TensorArrayConcatV2
TensorArraySplit
TensorArraySplitV2
TensorArraySize
TensorArraySizeV2
TensorArrayClose
TensorArrayCloseV2
Barrier
BarrierInsertMany
BarrierTakeMany
BarrierClose
BarrierReadySize
BarrierIncompleteSize
GetSessionHandle
GetSessionHandleV2
GetSessionTensor
DeleteSessionTensor
Stage
Unstage
StagePeek
StageSize
StageClear
MapStage
MapPeek
MapUnstage
MapUnstageNoKey
MapSize
MapIncompleteSize
MapClear
OrderedMapStage
OrderedMapPeek
OrderedMapUnstage
OrderedMapUnstageNoKey
OrderedMapSize
OrderedMapIncompleteSize
OrderedMapClear
RecordInput

### dataset

TensorDataset
TensorSliceDataset
SparseTensorSliceDataset
ZipDataset
ConcatenateDataset
RepeatDataset
TakeDataset
SkipDataset
IgnoreErrorsDataset
MapDataset
ParallelMapDataset
FlatMapDataset
InterleaveDataset
GroupByWindowDataset
FilterDataset
BatchDataset
PaddedBatchDataset
DenseToSparseBatchDataset
RangeDataset
ShuffleDataset
CacheDataset
TextLineDataset
FixedLengthRecordDataset
TFRecordDataset
Iterator
MakeIterator
OneShotIterator
IteratorGetNext
IteratorDispose
IteratorToStringHandle
IteratorFromStringHandle

### Debug

Copy
CopyHost
DebugIdentity
DebugNanCount
DebugNumericSummary

### function

`_Arg`
`Retval`
`_ListToArray`
`_ArrayToList`
MapAccumulate
SymbolicGradient

### image

ResizeArea
ResizeBicubic
ResizeBilinear
QuantizedResizeBilinear
ResizeBilinearGrad
ResizeNearestNeighbor
ResizeNearestNeighborGrad
RandomCrop
DecodeJpeg
EncodeJpeg
AdjustContrast
AdjustContrastv2
AdjustHue
AdjustSaturation
DecodePng
EncodePng
DecodeBmp
DecodeGif
RGBToHSV
HSVToRGB
DrawBoundingBoxes
SampleDistortedBoundingBox
SampleDistortedBoundingBoxV2
ExtractGlimpse
CropAndResize
CropAndResizeGradImage
CropAndResizeGradBoxes
NonMaxSuppression
NonMaxSuppressionV2

### io

SaveV2
RestoreV2
MergeV2Checkpoints
Save
SaveSlices
Restore
RestoreSlice
ShardedFilename
ShardedFilespec
WholeFileReader
WholeFileReaderV2
TextLineReader
TextLineReaderV2
FixedLengthRecordReader
FixedLengthRecordReaderV2
TFRecordReader
TFRecordReaderV2
LMDBReader
IdentityReader
IdentityReaderV2
ReaderRead
ReaderReadV2
ReaderReadUpTo
ReaderReadUpToV2
ReaderNumRecordsProduced
ReaderNumRecordsProducedV2
ReaderNumWorkUnitsCompleted
ReaderNumWorkUnitsCompletedV2
ReaderSerializeState
ReaderSerializeStateV2
ReaderRestoreState
ReaderRestoreStateV2
ReaderReset
ReaderResetV2
ReadFile
WriteFile
MatchingFiles

### linalg

MatrixDeterminant
MatrixInverse
Cholesky
CholeskyGrad
SelfAdjointEig
SelfAdjointEigV2
MatrixSolve
MatrixTriangularSolve
MatrixSolveLs
Qr
Svd
BatchSelfAdjointEig
BatchMatrixDeterminant
BatchMatrixInverse
BatchCholesky
BatchCholeskyGrad
BatchSelfAdjointEigV2
BatchMatrixSolve
BatchMatrixTriangularSolve
BatchMatrixSolveLs
BatchSvd

### log

Assert
Print
TensorSummaryV2
TensorSummary
ScalarSummary
HistogramSummary
ImageSummary
AudioSummaryV2
AudioSummary
MergeSummary

### lookup

LookupTableFind
LookupTableFindV2
LookupTableInsert
LookupTableInsertV2
LookupTableSize
LookupTableSizeV2
LookupTableExport
LookupTableExportV2
LookupTableImport
LookupTableImportV2
HashTable
HashTableV2
MutableHashTable
MutableHashTableV2
MutableHashTableOfTensors
MutableHashTableOfTensorsV2
MutableDenseHashTable
MutableDenseHashTableV2
InitializeTable
InitializeTableV2
InitializeTableFromTextFile
InitializeTableFromTextFileV2

### math grad

Abs
Neg
Inv
Reciprocal
Square
Sqrt
Rsqrt
Exp
Expm1
Log
Log1p
Sinh
Cosh
Tanh
Asinh
Acosh
Atanh
Sigmoid
Sign
Sin
Cos
Acos
Asin
Atan
Tan
Real
Imag
Conj
Add
Sub
Mul
Div
RealDiv
Pow
Maximum
Minimum
Complex
Select
AddN
Sum
Mean
Prod
SegmentSum
SegmentMean
SparseSegmentSum
SparseSegmentMean
SparseSegmentSqrtN
SegmentMin
SegmentMax
UnsortedSegmentSum
UnsortedSegmentMax
Max
Min
MatMul
BatchMatMul
SparseMatMul
Less
LessEqual
Greater
GreaterEqual
Equal
NotEqual
LogicalAnd
LogicalOr
LogicalNot
Range
LinSpace
Floor
FloorDiv
TruncateDiv

### math

`_AddN`
BatchMatMul
Cast
`_HostCast`
Abs
ComplexAbs
Neg
Inv
InvGrad
Reciprocal
ReciprocalGrad
Square
Sqrt
SqrtGrad
Rsqrt
Round
RsqrtGrad
Exp
Expm1
Log
Log1p
Sinh
Cosh
Tanh
Asinh
Acosh
Atanh
TanhGrad
Lgamma
Digamma
Erf
Erfc
Sigmoid
SigmoidGrad
Sin
Cos
Tan
Asin
Acos
Atan
IsNan
IsInf
IsFinite
Sign
Floor
Ceil
Rint
Add
Sub
Mul
Div
FloorDiv
TruncateDiv
RealDiv
SquaredDifference
Maximum
Minimum
Mod
FloorMod
TruncateMod
Pow
Igammac
Igamma
Zeta
Polygamma
Atan2
Betainc
Less
LessEqual
Greater
GreaterEqual
Equal
NotEqual
ApproximateEqual
LogicalNot
LogicalAnd
LogicalOr
Select
MatMul
SparseMatMul
Sum
Mean
Prod
Min
Max
ArgMax
ArgMin
SegmentSum
SegmentMean
SegmentProd
SegmentMin
SegmentMax
UnsortedSegmentSum
UnsortedSegmentMax
SparseSegmentSum
SparseSegmentMean
SparseSegmentMeanGrad
SparseSegmentSqrtN
SparseSegmentSqrtNGrad
All
Any
Range
LinSpace
Complex
Real
Imag
Conj
Cross
Bincount
Cumsum
Cumprod
QuantizedMatMul
QuantizedMul
QuantizedAdd
QuantizeDownAndShrinkRange
Requantize
CompareAndBitpack
RequantizationRange
Bucketize

### neutral network gradient

Softmax
Relu
Relu6
CrossEntropy
Conv2D
MaxPool
AvgPool
MaxPoolGrad
BiasAdd

### neutral network

AvgPool
AvgPoolGrad
BatchNormWithGlobalNormalization
BatchNormWithGlobalNormalizationGrad
FusedBatchNorm
FusedBatchNormGrad
BiasAdd
BiasAddGrad
BiasAddV1
Conv2D
Conv2DBackpropInput
Conv2DBackpropFilter
FusedResizeAndPadConv2D
FusedPadConv2D
DepthwiseConv2dNative
DepthwiseConv2dNativeBackpropInput
DepthwiseConv2dNativeBackpropFilter
Conv3D
Conv3DBackpropInput
Conv3DBackpropFilter
Conv3DBackpropInputV2
Conv3DBackpropFilterV2
AvgPool3D
AvgPool3DGrad
MaxPool3D
MaxPool3DGrad
MaxPool3DGradGrad
L2Loss
LRN
LRNGrad
MaxPool
MaxPoolGrad
MaxPoolGradGrad
MaxPoolWithArgmax
MaxPoolGradWithArgmax
MaxPoolGradGradWithArgmax
Dilation2D
Dilation2DBackpropInput
Dilation2DBackpropFilter
Relu
ReluGrad
Relu6
Relu6Grad
Elu
EluGrad
Selu
SeluGrad
Softplus
SoftplusGrad
Softsign
SoftsignGrad
Softmax
LogSoftmax
SoftmaxCrossEntropyWithLogits
SparseSoftmaxCrossEntropyWithLogits
InTopK
InTopKV2
TopK
TopKV2
FractionalMaxPool
FractionalMaxPoolGrad
FractionalAvgPool
FractionalAvgPoolGrad
QuantizedAvgPool
QuantizedBiasAdd
QuantizedConv2D
QuantizedMaxPool
QuantizedRelu
QuantizedRelu6
QuantizedReluX
QuantizedBatchNormWithGlobalNormalization
`_MklConv2D`
`_MklConv2DWithBias`
`_MklConv2DBackpropFilter`
`_MklConv2DWithBiasBackpropBias`
`_MklConv2DBackpropInput`
`_MklRelu`
`_MklReluGrad`
`_MklMaxPool`
`_MklMaxPoolGrad`
`_MklAvgPool`
`_MklAvgPoolGrad`
`_MklLRN`
`_MklLRNGrad`
`_MklFusedBatchNorm`
`_MklFusedBatchNormGrad`
`_MklToTf`

### parse

DecodeRaw
ParseExample
ParseSingleSequenceExample
ParseTensor
DecodeJSONExample
DecodeCSV
StringToNumber

### random gradient

RandomUniform

### random

RandomUniform
RandomUniformInt
RandomStandardNormal
ParameterizedTruncatedNormal
TruncatedNormal
RandomShuffle
Multinomial
RandomGamma
RandomPoisson

#### script

PyFunc
PyFuncStateless

#### sdca

dcaOptimizer
SdcaShrinkL1
SdcaFprint

#### sendrecv

`_Send`
`_Recv`
`_HostSend`
`_HostRecv`

#### set

SetSize
DenseToDenseSetOperation
DenseToSparseSetOperation
SparseToSparseSetOperation

#### sparse

SparseAddGrad
SparseAdd
SparseTensorDenseMatMul
SerializeSparse
SerializeManySparse
DeserializeManySparse
SparseToDense
SparseConcat
SparseCross
SparseSplit
SparseSlice
SparseReorder
SparseReshape
SparseTensorDenseAdd
SparseReduceMax
SparseReduceMaxSparse
SparseReduceSum
SparseReduceSumSparse
SparseDenseCwiseMul
SparseDenseCwiseDiv
SparseDenseCwiseAdd
SparseSoftmax
SparseSparseMaximum
SparseSparseMinimum
AddSparseToTensorsMap
AddManySparseToTensorsMap
TakeManySparseFromTensorsMap
SparseFillEmptyRows
SparseFillEmptyRowsGrad

### spectral

FFT
IFFT
FFT2D
IFFT2D
FFT3D
IFFT3D
RFFT
IRFFT
RFFT2D
IRFFT2D
RFFT3D
IRFFT3D
BatchFFT
BatchIFFT
BatchFFT2D
BatchIFFT2D
BatchFFT3D
BatchIFFT3D

### state

VariableV2
Variable
IsVariableInitialized
TemporaryVariable
DestroyTemporaryVariable
Assign
AssignAdd
AssignSub
ScatterUpdate
ScatterAdd
ScatterSub
ScatterMul
ScatterDiv
ScatterNdUpdate
ScatterNdAdd
ScatterNdSub
ScatterNdMul
ScatterNdDiv
CountUpTo

### stateless

StatelessRandomUniform
StatelessRandomNormal
StatelessTruncatedNormal

### String

StringToHashBucketFast
StringToHashBucketStrong
StringToHashBucket
ReduceJoin
AsString
StringJoin
StringSplit
EncodeBase64
DecodeBase64
Substr

### training

ApplyGradientDescent
ResourceApplyGradientDescent
ApplyDelayCompensatedGradientDescent
ApplyProximalGradientDescent
SparseApplyProximalGradientDescent
ResourceApplyProximalGradientDescent
ResourceSparseApplyProximalGradientDescent
ApplyAdadelta
SparseApplyAdadelta
ResourceApplyAdadelta
ResourceSparseApplyAdadelta
ApplyAdagrad
ResourceApplyAdagrad
ApplyProximalAdagrad
ResourceApplyProximalAdagrad
SparseApplyAdagrad
ResourceSparseApplyAdagrad
ApplyAdagradDA
SparseApplyAdagradDA
SparseApplyProximalAdagrad
ResourceApplyAdagradDA
ResourceSparseApplyAdagradDA
ResourceSparseApplyProximalAdagrad
ApplyFtrl
SparseApplyFtrl
ResourceApplyFtrl
ResourceSparseApplyFtrl
ApplyFtrlV2
SparseApplyFtrlV2
ResourceApplyFtrlV2
ResourceSparseApplyFtrlV2
ApplyMomentum
SparseApplyMomentum
ResourceApplyMomentum
ResourceSparseApplyMomentum
ApplyAdam
ResourceApplyAdam
ApplyRMSProp
ApplyCenteredRMSProp
SparseApplyRMSProp
SparseApplyCenteredRMSProp
ResourceApplyRMSProp
ResourceApplyCenteredRMSProp
ResourceSparseApplyRMSProp
ResourceSparseApplyCenteredRMSProp

### word2vec

Skipgram
NegTrain


#### SymbolicGradient


通过 BP 算法计算 Gradient， 接受输入 N+M 输出 N

(dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N, dL/dy1, dL/dy2, ..., dL/dy_M)

Pack :

例子

x = [1, 4] y = [2, 5] z = [3, 6]
pack([x, y z]) = [[1, 4],[2, 5], [3, 6]]
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]




更多参考 ops.pbtxt



