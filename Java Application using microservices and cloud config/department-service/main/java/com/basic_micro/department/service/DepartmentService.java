package com.basic_micro.department.service;

import com.basic_micro.department.entity.Department;
import com.basic_micro.department.error.DepartmentNotFoundException;

public interface DepartmentService {
    public Department saveDepartment(Department department);

    public Department printDepartment(Long departmentId) throws DepartmentNotFoundException;
}
