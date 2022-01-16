package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.Department;
import com.amansans.BasicSBProject.error.DepartmentNotFoundException;

import java.util.List;

public interface DepartmentService {
    public Department saveDepartment(Department department);

    public List<Department> fetchDeparmentList();

    public Department getDepartmentById(Long departmentId) throws DepartmentNotFoundException;

    public void deleteDepartmentById(Long departmentId);

    public Department updateDepartment(Long departmentID, Department department);

    public Department getDepartmentByName(String departmentName);
}
