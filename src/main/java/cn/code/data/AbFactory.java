package cn.code.data;

public class AbFactory {

    public static void main(String[] args) {
        Util util = new A();
        Iconnet iconnet = util.getConnet();
        iconnet.connet();
        Isend isend = util.getIsend();
        isend.send();

    }


    interface Iconnet{
        void connet();

    }

    static  class Connet1 implements Iconnet{

        @Override
        public void connet() {
            System.out.println("connet111");
        }
    }

    interface Isend{
        void send();

    }

    static  class Send1 implements Isend{

        @Override
        public void send() {
            System.out.println("send111");
        }
    }

    interface Util{
        Iconnet getConnet();
        Isend getIsend();
    }

    static  class A implements  Util{

        @Override
        public Iconnet getConnet() {
            return new Connet1();
        }

        @Override
        public Isend getIsend() {
            return new Send1();
        }
    }






}
