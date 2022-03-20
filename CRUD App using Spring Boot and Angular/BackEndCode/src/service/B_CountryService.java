package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.B_Country;

import java.util.List;

public interface B_CountryService {
    List<B_Country> printCountryList();

    Long printCountryId(String countryName);
}
