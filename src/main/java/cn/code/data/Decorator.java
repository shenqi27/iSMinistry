package cn.code.data;

public class Decorator {

    // 在不改变原有对象的基础上，把功能附加到对象上

    public static void main(String[] args) {
           Component component1 =  new CreateCompnent();
            Component component = new Meiyan(new Lvjing(component1));
        component.operation();
    }

}

interface  Component{
    void operation();
}

class CreateCompnent implements  Component{

    @Override
    public void operation() {
        System.out.println("paizhao");
    }
}

abstract class Decorator22 implements  Component{
    Component component;
    public Decorator22(Component component){
        this.component = component;
    }
}

class Lvjing extends Decorator22{

    public Lvjing(Component component){
        super(component);
    }

    @Override
    public void operation() {

        System.out.println("lvjing");
        component.operation();
    }


}



class Meiyan extends Decorator22{

    public Meiyan(Component component){
        super(component);
    }

    @Override
    public void operation() {

        System.out.println("meiyan");
        component.operation();
    }


}
