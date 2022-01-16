package com.amansans.BasicSBProject.controller;

import com.amansans.BasicSBProject.entity.Department;
import com.amansans.BasicSBProject.error.DepartmentNotFoundException;
import com.amansans.BasicSBProject.service.DepartmentService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;

import static org.junit.jupiter.api.Assertions.*;

@WebMvcTest(DepartmentController.class)
class DepartmentControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private DepartmentService departmentService;

    private Department department;

    @BeforeEach
    void setUp() {
        Department department = Department.builder().
                departmentName("IT").
                departmentAddress("Test").
                departmentCode("TT").
                departmentId(1L).
                build();
    }

    @Test
    public void saveDepartment() throws Exception {
        Department inputDepartment = Department.builder().
                departmentName("IT").
                departmentAddress("Test").
                departmentCode("TT").
                build();

        Mockito.when(departmentService.saveDepartment(inputDepartment)).thenReturn(department);

        mockMvc.perform(MockMvcRequestBuilders.post("/departments")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\n" +
                        "    \"departmentName\":\"IT\",\n" +
                        "    \"departmentAddress\":\"Test\",\n" +
                        "    \"departmentCode\":\"TT\"\n" +
                        "}")).andExpect(MockMvcResultMatchers.status().isOk());
    }

    //@Test
    public void getDepartmentById() throws Exception {
        Mockito.when(departmentService.getDepartmentById(1L)).thenReturn(department);

        mockMvc.perform(MockMvcRequestBuilders.get("/departments/1").contentType(MediaType.APPLICATION_JSON))
                .andExpect(MockMvcResultMatchers.status().isOk())
                .andExpect(MockMvcResultMatchers.jsonPath("$.departmentName")
                        .value(department.getDepartmentName()));
    }
}