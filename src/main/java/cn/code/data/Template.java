package cn.code.data;

public class Template {

    // 定义一个操作的算法骨架，而将一些步骤延迟到子类中，Template使得自雷可以不改变一个算法的结构即可重新定义该算法的某些特定步骤
    public static void main(String[] args) {
        AbstractAAA a = new AAA2();
        a.connet();
    }
}

abstract class AbstractAAA{
    public void connet(){
        System.out.println("1");
        System.out.println("2");
        System.out.println("3");
        templateMethod();
    }

    abstract protected void  templateMethod();
}

class AAA2 extends AbstractAAA{

    @Override
    protected void templateMethod() {
        System.out.println("A2");
    }
}
