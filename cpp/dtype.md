

tensorflow/core/framework/types.proto

  * `tf.float16`: 16-bit half-precision floating-point.
  * `tf.float32`: 32-bit single-precision floating-point.
  * `tf.float64`: 64-bit double-precision floating-point.
  * `tf.bfloat16`: 16-bit truncated floating-point.
  * `tf.complex64`: 64-bit single-precision complex.
  * `tf.complex128`: 128-bit double-precision complex.
  * `tf.int8`: 8-bit signed integer.
  * `tf.uint8`: 8-bit unsigned integer.
  * `tf.uint16`: 16-bit unsigned integer.
  * `tf.int16`: 16-bit signed integer.
  * `tf.int32`: 32-bit signed integer.
  * `tf.int64`: 64-bit signed integer.
  * `tf.bool`: Boolean.
  * `tf.string`: String.
  * `tf.qint8`: Quantized 8-bit signed integer.
  * `tf.quint8`: Quantized 8-bit unsigned integer.
  * `tf.qint16`: Quantized 16-bit signed integer.
  * `tf.quint16`: Quantized 16-bit unsigned integer.
  * `tf.qint32`: Quantized 32-bit signed integer.
  * `tf.resource`: Handle to a mutable resource.
  * `tf.variant`: Values of arbitrary types.


dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


resource = DType(types_pb2.DT_RESOURCE)
variant = DType(types_pb2.DT_VARIANT)
float16 = DType(types_pb2.DT_HALF)
half = float16
float32 = DType(types_pb2.DT_FLOAT)
float64 = DType(types_pb2.DT_DOUBLE)
double = float64
int32 = DType(types_pb2.DT_INT32)
uint8 = DType(types_pb2.DT_UINT8)
uint16 = DType(types_pb2.DT_UINT16)
int16 = DType(types_pb2.DT_INT16)
int8 = DType(types_pb2.DT_INT8)
string = DType(types_pb2.DT_STRING)
complex64 = DType(types_pb2.DT_COMPLEX64)
complex128 = DType(types_pb2.DT_COMPLEX128)
int64 = DType(types_pb2.DT_INT64)
bool = DType(types_pb2.DT_BOOL)
qint8 = DType(types_pb2.DT_QINT8)
quint8 = DType(types_pb2.DT_QUINT8)
qint16 = DType(types_pb2.DT_QINT16)
quint16 = DType(types_pb2.DT_QUINT16)
qint32 = DType(types_pb2.DT_QINT32)
resource_ref = DType(types_pb2.DT_RESOURCE_REF)
variant_ref = DType(types_pb2.DT_VARIANT_REF)
bfloat16 = DType(types_pb2.DT_BFLOAT16)
float16_ref = DType(types_pb2.DT_HALF_REF)
half_ref = float16_ref
float32_ref = DType(types_pb2.DT_FLOAT_REF)
float64_ref = DType(types_pb2.DT_DOUBLE_REF)
double_ref = float64_ref
int32_ref = DType(types_pb2.DT_INT32_REF)
uint8_ref = DType(types_pb2.DT_UINT8_REF)
uint16_ref = DType(types_pb2.DT_UINT16_REF)
int16_ref = DType(types_pb2.DT_INT16_REF)
int8_ref = DType(types_pb2.DT_INT8_REF)
string_ref = DType(types_pb2.DT_STRING_REF)
complex64_ref = DType(types_pb2.DT_COMPLEX64_REF)
complex128_ref = DType(types_pb2.DT_COMPLEX128_REF)
int64_ref = DType(types_pb2.DT_INT64_REF)
bool_ref = DType(types_pb2.DT_BOOL_REF)
qint8_ref = DType(types_pb2.DT_QINT8_REF)
quint8_ref = DType(types_pb2.DT_QUINT8_REF)
qint16_ref = DType(types_pb2.DT_QINT16_REF)
quint16_ref = DType(types_pb2.DT_QUINT16_REF)
qint32_ref = DType(types_pb2.DT_QINT32_REF)
bfloat16_ref = DType(types_pb2.DT_BFLOAT16_REF)
