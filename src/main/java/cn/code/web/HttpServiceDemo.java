package cn.code.web;


import cn.code.data.AAAA;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

/**
 Http服务端
// */
@RestController
@RequestMapping(value = "/httpService")
public class HttpServiceDemo {


    @RequestMapping(value = "/aaaa", method = RequestMethod.POST)
    public String getUserInfo(@RequestBody AAAA aa){
        System.out.println();
        return aa.getA();
    }






}
