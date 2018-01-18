
具体实现在 tensorflow/core/kernels/ 相关文件

* tensorflow/core/framework/reader_interface.h
* tensorflow/core/framework/reader_base.h
* tensorflow/core/framework/reader_base.cpp
* tensorflow/core/kernels/whole_file_read_op.cpp
* tensorflow/core/kernels/text_line_reader_op.cpp
* tensorflow/core/kernels/fixed_length_record_reader_op.cpp
* tensorflow/core/kernels/tf_record_reader_op.cpp
* tensorflow/core/kernels/lmdb_reader_op.cpp
* tensorflow/core/kernels/identity_reader_op.cpp

而这些类依赖

* tensorflow/core/lib/io/random_inputstream.cc //whole file read
* tensorflow/core/lib/io/inputbuffer.cc   //text_line_reader
* tensorflow/core/lib/io/buffered_inputstream.cc //fixed_length_record_reader_op
* tensorflow/core/lib/io/record_reader.cc //record_reader

