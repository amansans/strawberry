package com.amansans.BasicSBProject.repository;

import com.amansans.BasicSBProject.entity.B_Country;
import com.amansans.BasicSBProject.entity.B_UserDetails;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;
import java.util.Optional;

@SpringBootTest
public class B_UserDetailsRepositoryTest {

    @Autowired
    private B_UserDetailsRepository userDetailsRepository;

//    @Test
    public void saveUser(){
        B_Country country = B_Country.builder()
                .countryName("Canada")
                .build();

        B_UserDetails userDetails =  B_UserDetails.builder()
                .firstName("Aman")
                .lastName("Sansowa")
                .emailId("amansansowa@gmail.com")
                .country(country)
                .build();

        userDetailsRepository.save(userDetails);
    }

//    @Test
    public void printUser(){
        var users = userDetailsRepository.findAll();
        System.out.println("Users: " + users);
    }

}
