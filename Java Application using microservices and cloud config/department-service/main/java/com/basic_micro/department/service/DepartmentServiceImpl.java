package com.basic_micro.department.service;

import com.basic_micro.department.entity.Department;
import com.basic_micro.department.error.DepartmentNotFoundException;
import com.basic_micro.department.repository.DepartmentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@Slf4j
public class DepartmentServiceImpl implements DepartmentService{

    @Autowired
    private DepartmentRepository departmentRepository;

    @Override
    public Department saveDepartment(Department department) {
        log.info("Inside saveDepartment method of Department Service Implementer");
        return departmentRepository.save(department);
    }

    @Override
    public Department printDepartment(Long departmentId) throws DepartmentNotFoundException {
        log.info("Inside printDepartment method of Department Service Implementer");
        Department department = departmentRepository.findByDepartmentId(departmentId);

        if (department == null) throw new DepartmentNotFoundException("Department does not exist for Id: " + departmentId);
        return departmentRepository.findByDepartmentId(departmentId);
    }
}
