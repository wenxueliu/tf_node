

## Server defination

1. client no-stream and server no-stream

```
rpc SayHello(HelloRequest) returns (HelloResponse){
}
```
2. client no-stream and server stream

```
rpc LotsOfReplies(HelloRequest) returns (stream HelloResponse){
}
```

3. client stream and server no-stream

```
rpc LotsOfGreetings(stream HelloRequest) returns (HelloResponse) {
}
```

4. client stream and server stream

```
rpc BidiHello(stream HelloRequest) returns (stream HelloResponse){
}
```

注：只有 4 是异步的，连接复用的，1，2，3都是一问一答

## WireFormat

```
Request

HEADERS (flags = END_HEADERS)
:method = POST
:scheme = http
:path = /google.pubsub.v2.PublisherService/CreateTopic
:authority = pubsub.googleapis.com
grpc-timeout = 1S
content-type = application/grpc+proto
grpc-encoding = gzip
authorization = Bearer y235.wef315yfh138vh31hv93hv8h3v

DATA (flags = END_STREAM)
<Delimited Message>
```



