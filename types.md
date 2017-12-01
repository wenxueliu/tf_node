


## 数据结构

enum DataType
  // Not a legal value for DataType.  Used to indicate a DataType field has not been set.
  DT_INVALID = 0;

  //Data types that all computation devices are expected to be capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types

  DT_FLOAT_REF = 101;
  DT_DOUBLE_REF = 102;
  DT_INT32_REF = 103;
  DT_UINT8_REF = 104;
  DT_INT16_REF = 105;
  DT_INT8_REF = 106;
  DT_STRING_REF = 107;
  DT_COMPLEX64_REF = 108;
  DT_INT64_REF = 109;
  DT_BOOL_REF = 110;
  DT_QINT8_REF = 111;
  DT_QUINT8_REF = 112;
  DT_QINT32_REF = 113;
  DT_BFLOAT16_REF = 114;
  DT_QINT16_REF = 115;
  DT_QUINT16_REF = 116;
  DT_UINT16_REF = 117;
  DT_COMPLEX128_REF = 118;
  DT_HALF_REF = 119;
  DT_RESOURCE_REF = 120;
  DT_VARIANT_REF = 121;

包含基本类型和引用类型，基本类型加 100 就得到引用类型，如果是引用类型，减 100 就得到基本类型

两个类型兼容的条件是都转为基本类型，类型相同


extern const char* const DEVICE_CPU;   // "CPU"
extern const char* const DEVICE_GPU;   // "GPU"
extern const char* const DEVICE_SYCL;  // "SYCL"

typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;

enum { kDataTypeRefOffset = 100 };

enum MemoryType
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1,

class DeviceType //这里讲类型字符串通过类，增加校验
  string type_;


class TypeIndex
  uint64 hash_;

在移动设备如安卓中为了减少二进制大小，禁止了 RTTI

## 源码分析

string DataTypeString(DataType dtype) // 返回  type 的 string 表达
bool DataTypeFromString(StringPiece sp, DataType* dt) //将 sp 转为 dt 类型
string DeviceTypeString(const DeviceType& device_type) //device_type.type()
string DataTypeSliceString(const DataTypeSlice types) //将各个类型以逗号分隔
DataTypeVector AllTypes() //所有类型的数组
DataTypeVector RealNumberTypes()
DataTypeVector QuantizedTypes()
DataTypeVector RealAndQuantizedTypes()
DataTypeVector NumberTypes()
bool DataTypeCanUseMemcpy(DataType dt)
bool DataTypeIsQuantized(DataType dt)
bool DataTypeIsInteger(DataType dt)
int DataTypeSize(DataType dt)

