package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.B_Country;
import com.amansans.BasicSBProject.repository.B_CountryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class B_CountryServiceImpl implements  B_CountryService{

    @Autowired
    private B_CountryRepository countryRepository;

    @Override
    public List<B_Country> printCountryList() {
        return  countryRepository.findAll();
    }

    @Override
    public Long printCountryId(String countryName) {
        return countryRepository.findByCountryName(countryName).getCountryID();
    }
}
