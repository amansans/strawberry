package com.amansans.BasicSBProject.repository;

import com.amansans.BasicSBProject.entity.Course;
import com.amansans.BasicSBProject.entity.CourseMaterial;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import java.util.List;

@SpringBootTest
public class CourseMaterialRepositoryTest {

    @Autowired
    private CourseMaterialRepository courseMaterialRepository;

    @Test
    public void saveCourseMaterial(){
        Course course = Course.builder()
                .title("Sex Ed")
                .credits(4)
                .build();

        CourseMaterial courseMaterial = CourseMaterial.builder()
                .url("www.3.com")
                .course(course)
                .build();

        courseMaterialRepository.save(courseMaterial);
    }



//    @Test
    public void printAllCourseMaterials(){
        List<CourseMaterial> courseMaterialList = courseMaterialRepository.findAll();
        System.out.println("Course Material = " + courseMaterialList);
    }
}
