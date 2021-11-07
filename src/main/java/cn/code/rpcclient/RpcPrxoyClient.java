package cn.code.rpcclient;

import cn.code.rpcserver.RemoteInvocationHandler;

import java.lang.reflect.Proxy;

public class RpcPrxoyClient {

    public  <T>T clientProxy(Class interfacecls,String host, int port){

        return (T) Proxy.newProxyInstance(interfacecls.getClassLoader(), new Class[]{interfacecls},new RemoteInvocationHandler(host,port));

    }
}
