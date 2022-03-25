package com.basic_micro.user.controller;

import com.basic_micro.user.VO.Department;
import com.basic_micro.user.VO.ResponseTemplateVO;
import com.basic_micro.user.entity.User;
import com.basic_micro.user.error.UserNotFoundException;
import com.basic_micro.user.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import javax.validation.Valid;

@RestController
@RequestMapping("/users")
@Slf4j
public class UserController {

    @Autowired
    private UserService userService;

    @PostMapping("/")
    public User saveUser(@Valid @RequestBody User user){
        log.info("Entered saveUser function for UserController");
        return userService.saveUser(user);
    }

    @GetMapping("/{id}")
    public ResponseTemplateVO getUserWithDepartment(@PathVariable("id") Long userId) throws UserNotFoundException {
        log.info("Entered getUserWithDepartmentId function for UserController");
        return userService.getUserWithDepartment(userId);
    }
}

