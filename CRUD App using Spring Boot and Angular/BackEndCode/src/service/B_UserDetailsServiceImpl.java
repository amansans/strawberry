package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.B_Country;
import com.amansans.BasicSBProject.entity.B_UserDetails;
import com.amansans.BasicSBProject.error.UserNotFoundException;
import com.amansans.BasicSBProject.repository.B_CountryRepository;
import com.amansans.BasicSBProject.repository.B_UserDetailsRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Objects;

@Service
public class B_UserDetailsServiceImpl implements B_UserDetailsService{

    @Autowired
    private B_UserDetailsRepository userDetailsRepository;

    @Autowired
    private B_CountryRepository countryRepository;

    @Override
    public B_UserDetails saveUser(B_UserDetails userDetails) {

        B_Country country = countryRepository.findByCountryName(userDetails.getCountry().getCountryName());

        if(country == null) return userDetailsRepository.save(userDetails);
        else {
            String countryName = userDetails.getCountry().getCountryName();
            B_Country countryDetails = countryRepository.findByCountryName(countryName);
            B_UserDetails userDetails1 = new B_UserDetails();

            userDetails1.setCountry(countryDetails);
            userDetails1.setEmailId(userDetails.getEmailId());
            userDetails1.setFirstName(userDetails.getFirstName());
            userDetails1.setLastName(userDetails.getLastName());

            return userDetailsRepository.save(userDetails1);
        }
    }

    @Override
    public List<B_UserDetails> getUserList() {
        return userDetailsRepository.findAll();
    }

    @Override
    public B_UserDetails updateUser(Long id, B_UserDetails userDetails) throws UserNotFoundException {

        var user = userDetailsRepository.findById(id);

        if(user.isEmpty()) throw new UserNotFoundException("Cannot update User: User with Id " +id+ " Does not exist");

        if(Objects.nonNull(userDetails.getFirstName()) && !"".equalsIgnoreCase(userDetails.getFirstName())) {
            user.get().setFirstName(userDetails.getFirstName());
        }

        if(Objects.nonNull(userDetails.getLastName()) && !"".equalsIgnoreCase(userDetails.getLastName()) ){
            user.get().setLastName(userDetails.getLastName());
        }

        if(Objects.nonNull(userDetails.getEmailId()) && !"".equalsIgnoreCase(userDetails.getEmailId()) ){
            user.get().setEmailId(userDetails.getEmailId());
        }

        if(Objects.nonNull(userDetails.getCountry()) && !"".equalsIgnoreCase(String.valueOf(userDetails.getCountry())) ){
            B_Country country = countryRepository.findByCountryName(userDetails.getCountry().getCountryName());

            if(country == null){
                user.get().setCountry(userDetails.getCountry());
            } else {
                String countryName = userDetails.getCountry().getCountryName();
                B_Country countryDetails = countryRepository.findByCountryName(countryName);
                user.get().setCountry(countryDetails);
            }
        }

        return userDetailsRepository.save(user.get());
    }

    @Override
    public void removeUser(Long id) throws UserNotFoundException {

        var user = userDetailsRepository.findById(id);
        if(user.isEmpty()) throw new UserNotFoundException("Cannot delete User: User with Id " +id+ " Does not exist");

        userDetailsRepository.delete(user.get());
    }

    @Override
    public B_UserDetails getUsersByFirstAndLastName(String firstName, String lastName) throws UserNotFoundException {
        var user = userDetailsRepository.findByFirstNameAndLastName(firstName,lastName);
        if(user == null) throw new UserNotFoundException(firstName + " " + lastName + " Does not exist");
        return user;
    }
}
