package com.amansans.BasicSBProject.repository;

import com.amansans.BasicSBProject.entity.B_UserDetails;
import com.amansans.BasicSBProject.error.UserNotFoundException;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface B_UserDetailsRepository extends JpaRepository<B_UserDetails,Long> {
    public Long findByEmailId(String email);
    public B_UserDetails findByFirstNameAndLastName(String firstName, String lastName);
}
