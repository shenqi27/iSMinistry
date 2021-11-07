package cn.code.data;

public class Adapter2 {
    public static void main(String[] args) {
        A2 a2 = new A2();
        a2.output5();
    }
}

class Adaptee{
    public int outPut220(){
        return 220;
    }
}

interface Target{
    int output5();
}

class A2 extends Adaptee implements Target{

    @Override
    public int output5() {
        int i = outPut220();
        System.out.println("转换");
        return 5;
    }
}
