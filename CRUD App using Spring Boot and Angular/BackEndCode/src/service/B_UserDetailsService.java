package com.amansans.BasicSBProject.service;

import com.amansans.BasicSBProject.entity.B_UserDetails;
import com.amansans.BasicSBProject.error.UserNotFoundException;

import java.util.List;

public interface B_UserDetailsService {
    B_UserDetails saveUser(B_UserDetails userDetails);

    public List<B_UserDetails> getUserList();

    public B_UserDetails updateUser(Long id, B_UserDetails userDetails) throws UserNotFoundException;

    public void removeUser(Long id) throws UserNotFoundException;

    public B_UserDetails getUsersByFirstAndLastName(String firstName, String lastName) throws UserNotFoundException;
}
