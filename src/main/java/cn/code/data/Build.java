package cn.code.data;

public class Build {

    public static void main(String[] args) {
        Chanping cp = new Chanping();
        cp.setName("1");
        cp.setName2("2");
        Build1 build1 = new Build1();
        Director director = new Director(build1);
        Chanping chanping = director.makeChanping("1","2","3");
        System.out.println(chanping);

        Chanping chanping222 =new Chanping.Build2().buildName1("3").buildName3("3").buildName2("3").build();
        System.out.println(chanping222);
    }


}

interface ProductBuild{
    void buildName1(String name1);
    void buildName2(String name2);
    void buildName3(String name3);
    Chanping build();
}


class Build1 implements ProductBuild{
    private String name;
    private String name2;
    private String name3;
    @Override
    public void buildName1(String name1) {
        this.name = name1;

    }

    @Override
    public void buildName2(String name2) {
        this.name2 = name2;

    }

    @Override
    public void buildName3(String name3) {
        this.name3 = name3;
    }

    @Override
    public Chanping build() {
        return new Chanping(this.name,this.name2,this.name3);
    }



}

class Director{
    private Build1 build1;

    public Director(Build1 build1){
        this.build1 = build1;
    }

    public Chanping makeChanping(String name,String name2,String name3){
        build1.buildName1(name);
        build1.buildName2(name2);
        build1.buildName3(name3);
        return build1.build();
    }
}


class Chanping{

    private String name;
    private String name2;
    private String name3;

    public Chanping(){

    }
    public Chanping(String name,String name2,String name3){
        this.name = name;
        this.name2 = name2;
        this.name3 = name3;

    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName2() {
        return name2;
    }

    public void setName2(String name2) {
        this.name2 = name2;
    }

    public String getName3() {
        return name3;
    }

    public void setName3(String name3) {
        this.name3 = name3;
    }

    static class Build2{
        private String name;
        private String name2;
        private String name3;
        public Build2 buildName1(String name1) {
            this.name = name1;
            return this;
        }
        public Build2 buildName2(String name2) {
            this.name2 = name2;        return this;


        }
        public Build2 buildName3(String name3) {
            this.name3 = name3;
            return this;

        }

        Chanping build(){
            return new Chanping(name,name2,name3);
        }
    }
}

