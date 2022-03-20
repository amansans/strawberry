package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.Department;
import com.amansans.BasicSBProject.repository.DepartmentRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class DepartmentServiceTest {

    @Autowired
    private DepartmentService departmentService;

    @MockBean
    private DepartmentRepository departmentRepository;

    @BeforeEach
    void setUp() {
        Department department = Department.builder().
                departmentName("IT").
                departmentAddress("Test").
                departmentCode("TT").
                departmentId(1L).
                build();

        Mockito.when(departmentRepository.findByDepartmentName("IT")).thenReturn(department);
    }

    @Test
    @DisplayName("Get Data Based on Valid department Name")
    public void whenValidDepartmentName_theDepartmentWillBeFound() {
        String departmentName = "IT";

        Department found = departmentService.getDepartmentByName(departmentName);
        assertEquals(departmentName,found.getDepartmentName());
    }
}