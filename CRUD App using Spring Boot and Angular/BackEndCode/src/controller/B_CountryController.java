package com.amansans.BasicSBProject.controller;

import com.amansans.BasicSBProject.entity.B_Country;
import com.amansans.BasicSBProject.service.B_CountryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class B_CountryController {

    @Autowired
    private B_CountryService countryService;

    @GetMapping("/country/GetCountryList")
    public List<B_Country> printCountryList(){
        return countryService.printCountryList();
    }

    @GetMapping("/country/GetCountryId/{country}")
    public Long printCountryId(@PathVariable("country") String countryName) {
        return countryService.printCountryId(countryName);
    }

}
