package cn.code.rpcclient;

import cn.code.rpcserver.HelloServer;

public class App {

    public static void main(String[] args) {

//        HelloServerImpl helloServer = new HelloServerImpl();
//        RpcProxyServer rpcProxyServer = new RpcProxyServer();
//        rpcProxyServer.publisher(8080,helloServer);
//
        RpcPrxoyClient rpcPrxoyClient = new RpcPrxoyClient();
        HelloServer helloServer1 = rpcPrxoyClient.clientProxy(HelloServer.class,"localhost",8080);
        String msg = helloServer1.sayHello("shenqi");
        System.out.println(msg);
;
    }
}
