package cn.code.data;

public class Facotry {

     public static void main(String[] args) {
          F1 f1 = new F11();

          Product product = f1.getProduct();
          product.method();
     }
}
      abstract class F1{
        abstract Product getProduct();
     }

      abstract class Product{
          abstract void method();
     }

       class P1 extends Product{
          @Override
          void method() {
               System.out.println("1");
          }
     }
      class F11 extends F1{

          @Override
          Product getProduct() {
               return new P1();
          }
     }




