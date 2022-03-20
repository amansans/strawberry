package com.amansans.BasicSBProject.controller;

import com.amansans.BasicSBProject.entity.B_UserDetails;
import com.amansans.BasicSBProject.entity.User;
import com.amansans.BasicSBProject.error.UserNotFoundException;
import com.amansans.BasicSBProject.service.B_UserDetailsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;

@RestController
public class B_UserDetailsController {

    @Autowired
    private B_UserDetailsService userDetailsService;

    @PostMapping("/adduser")
    public B_UserDetails saveUser(@Valid @RequestBody B_UserDetails userDetails){
        return userDetailsService.saveUser(userDetails);
    }

//    @CrossOrigin(origins = "http://localhost:4200")
    @GetMapping("/getUserList")
    public List<B_UserDetails> getUserList(){
        return userDetailsService.getUserList();
    }

    @PutMapping("/updateUser/{id}")
    public B_UserDetails updateUser(@PathVariable("id") Long id,
                                    @Valid @RequestBody B_UserDetails userDetails) throws UserNotFoundException {

        return userDetailsService.updateUser(id,userDetails);
    }

    @DeleteMapping("/removeUser/{id}")
    public void  removeUser(@PathVariable("id") Long id) throws UserNotFoundException {userDetailsService.removeUser(id);}

    @GetMapping("/getUsersByFirstAndLastName/{firstName}/{lastName}")
    public B_UserDetails getUsersByFirstAndLastName(@PathVariable("firstName") String firstName,
                                    @PathVariable("lastName") String lastName)  throws UserNotFoundException {

        return userDetailsService.getUsersByFirstAndLastName(firstName,lastName);
    }
}
