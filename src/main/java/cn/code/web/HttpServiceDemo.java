package cn.code.web;


import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

/**
 Http服务端
// */
@RestController
@RequestMapping(value = "/httpService")
public class HttpServiceDemo {


    @RequestMapping(value = "/aaaa", method = RequestMethod.POST)
    public String getUserInfo(@RequestBody AAAA aa){

        return aa.getA();
    }






}