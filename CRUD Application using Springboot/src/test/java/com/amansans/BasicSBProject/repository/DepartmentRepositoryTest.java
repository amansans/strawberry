package com.amansans.BasicSBProject.repository;

import com.amansans.BasicSBProject.entity.Department;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;

import static org.junit.jupiter.api.Assertions.*;

    @DataJpaTest
class DepartmentRepositoryTest {

    @Autowired
    private TestEntityManager testEntityManager;

    @Autowired
    private DepartmentRepository departmentRepository;

    @BeforeEach
    void setUp() {
        Department department = Department.builder().
                departmentName("IT").
                departmentAddress("Test").
                departmentCode("TT").
                build();

        testEntityManager.persist(department);
    }

    @Test
    @DisplayName("Get Department Name based on ID")
    public void whenFindById_thenReturnDepartment(){
        String departmentName = "IT";

        Department department = departmentRepository.findById(1L).get();
        assertEquals(department.getDepartmentName(),departmentName);

    }
}