
class GPUDeviceContext : public DeviceContext
  int stream_id_;
  gpu::Stream* stream_;
  gpu::Stream* host_to_device_stream_; //拷贝数据从 host 到 device
  gpu::Stream* device_to_host_stream_; //拷贝数据从 device 到 host
  gpu::Stream* device_to_device_stream_; //拷贝数据从 device 到 device
