package com.basic_micro.department.controller;

import com.basic_micro.department.entity.Department;
import com.basic_micro.department.error.DepartmentNotFoundException;
import com.basic_micro.department.service.DepartmentService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/departments")
@Slf4j
public class DepartmentController {

    @Autowired
    private DepartmentService departmentService;

    @PostMapping("/")
    public Department saveDepartment(@RequestBody Department department){
        log.info("Inside saveDepartment method of Department Controller");
        return departmentService.saveDepartment(department);
    }

    @GetMapping("/{id}")
    public Department printDepartment(@PathVariable("id") Long departmentId) throws DepartmentNotFoundException {
        log.info("Inside printDepartment method of Department Controller");
        return departmentService.printDepartment(departmentId);
    }
}
