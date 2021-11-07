package cn.code.rpcserver;

import cn.code.rpcclient.RpcNetTransport;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class RemoteInvocationHandler implements InvocationHandler {

    private String host;
    private int port;

    public RemoteInvocationHandler(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        RpcRequest request = new RpcRequest();
        request.setParamters(args);
        request.setClassName(method.getDeclaringClass().getName());
        request.setTypes(method.getParameterTypes());
        request.setMethodName(method.getName());
        RpcNetTransport rpcNetTransport = new RpcNetTransport(host,port);
        return rpcNetTransport.send(request);
    }
}
