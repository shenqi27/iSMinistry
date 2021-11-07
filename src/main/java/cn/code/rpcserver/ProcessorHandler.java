package cn.code.rpcserver;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.Socket;

public class ProcessorHandler implements Runnable{

    Socket socket;
    Object server = null;

    public ProcessorHandler(Socket socket,Object server){
        this.socket = socket;
        this.server =server;
    }

    public void run(){

        ObjectInputStream inputStream = null;
        ObjectOutputStream outputStream = null;

        try {
            inputStream = new ObjectInputStream(socket.getInputStream());
            RpcRequest rpcRequest = ((RpcRequest) inputStream.readObject());
            Object result = invoke(rpcRequest);
            outputStream = new ObjectOutputStream(socket.getOutputStream());
            outputStream.writeObject(result);
            outputStream.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public Object invoke(RpcRequest rpcRequest) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Object [] args = rpcRequest.getParamters();
        Class clazz = rpcRequest.getClass();
        Method method = clazz.getMethod(rpcRequest.getMethodName(),rpcRequest.getTypes());
        return method.invoke(server,args);
    }




}
