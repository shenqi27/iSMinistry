//package cn.code.data;
//
//public class Adapter {
//    // 将一个类的接口转换成用户希望的另一种接口
//    public static void main(String[] args) {
//        Adaptee adaptee = new Adaptee();
//        Target target = new Ada2(adaptee);
//        target.output5();
//
//    }
//}
//
//class Adaptee{
//    public int outPut220(){
//        return 220;
//    }
//}
//
//interface Target{
//    int output5();
//}
//
//class Ada2 implements Target{
//    private Adaptee adaptee;
//    public Ada2(Adaptee adaptee){
//        this.adaptee = adaptee;
//    }
//
//    @Override
//    public int output5() {
//        int i = adaptee.outPut220();
//        System.out.println("转换");
//        return 5;
//    }
//}
