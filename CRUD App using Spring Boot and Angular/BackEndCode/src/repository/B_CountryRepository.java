package com.amansans.BasicSBProject.repository;

import com.amansans.BasicSBProject.entity.B_Country;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface B_CountryRepository extends JpaRepository<B_Country,Long> {
    B_Country findByCountryName(String country);
}
