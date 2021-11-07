package cn.code.rpcserver;

public class HelloServerImpl implements HelloServer{

    @Override
    public String sayHello(String text) {
        System.out.println("kaishi");
        return text;
    }
}
