import { Component, NgModule, OnInit } from '@angular/core';
import { user_details } from '../interface/user_details';
import { UserService } from '../service/user.service';
import { HttpErrorResponse } from '@angular/common/http';
import { country } from '../interface/country';
import { Form, FormControl, FormGroup, NgForm, Validators } from '@angular/forms';
import { Observable } from 'rxjs';
import { error } from '@angular/compiler/src/util';
import { not } from '@angular/compiler/src/output/output_ast';

@Component({
  selector: 'app-users',
  templateUrl: './users.component.html',
  styleUrls: ['./users.component.css']
})
export class UsersComponent implements OnInit {

  hideGetUserList = true
  hideAddUserList = true
  hideUpdateUserList = true
  hideDeleteUserList = true
  userList: user_details[];

  deleteUserForm = new FormGroup({
    id : new FormControl('',
    [
      Validators.required
    ])
  })

  userForm = new FormGroup({
    id : new FormControl(0,
    [
      Validators.required
    ]),
    firstName : new FormControl('',
    [
      Validators.required,  
      Validators.minLength(2)
    ]),
    lastName : new FormControl('',
    [
      Validators.required,
      Validators.minLength(2)
    ]),
    emailId : new FormControl('',
    [
      Validators.required,
      Validators.email
    ]),
    countryName : new FormControl('',[
      Validators.required
    ])
  })

  constructor(private userService: UserService) {}

  ngOnInit(): void {}
  
  get f() { return this.userForm.controls }
  get d() { return this.deleteUserForm.controls }
  
  public unhideDiv(division : string): void {

    if(division == 'getUserDiv') {
      this.hideAddUserList = true;
      this.hideUpdateUserList = true;
      this.hideDeleteUserList = true;
      this.hideGetUserList = !this.hideGetUserList
      this.getUserList()
    }
    if(division == 'addUserDiv'){
      this.hideGetUserList = true;
      this.hideUpdateUserList = true;
      this.hideDeleteUserList = true;
      this.hideAddUserList = !this.hideAddUserList
    }
    if(division == 'updateUserDiv'){
      this.hideGetUserList = true;
      this.hideAddUserList = true;
      this.hideDeleteUserList = true;
      this.hideUpdateUserList = !this.hideUpdateUserList
    }
    if(division == 'deleteUserDiv'){
      this.hideGetUserList = true;
      this.hideAddUserList = true;
      this.hideUpdateUserList = true;
      this.hideDeleteUserList = !this.hideDeleteUserList
    }
  }

  public getUserList() : void {
    this.userService.getUserList().subscribe(
      (response : user_details[]) => {

        this.userList = response;
      },
      (error : HttpErrorResponse) => {
        alert(error.message);
      }
    );
  }

  public addUser(userForm:FormGroup) : void {
    
    const obj = {
      "firstName":userForm.value.firstName,
      "lastName":userForm.value.lastName,
      "emailId":userForm.value.emailId,
      "country": {
          "countryName":userForm.value.countryName
      }
  }

    this.userService.addUser(obj).subscribe(
      (response: any) => { 
        console.log("AddedUser_Response ", response)
        this.userForm.reset();
      },
      (error: HttpErrorResponse) => {
        alert(error.message);
      }
    );
  }

  // ideally we want to check if the id exists or not when updating the user details
  // If it doesnt exist we should send an error message to the user
  // However this is out of scope for now
  // Also, we can create a new form with updated validation conditions to allow null values for certain parameters

  public updateUser(userForm:FormGroup) : void {
    
    const obj = {
      "firstName":userForm.value.firstName,
      "lastName":userForm.value.lastName,
      "emailId":userForm.value.emailId,
      "country": {
          "countryName":userForm.value.countryName
      }
  }

    this.userService.updateUser(obj,userForm.value.id).subscribe(
      (response: any) => { 
        console.log("updateUser ", response);
        this.userForm.reset();
      },
      (error: HttpErrorResponse) => {
        alert(error.message);
      }
    );
  }

  public deleteUser(deleteUserForm:FormGroup) : void {
    
    this.userService.deleteUser(deleteUserForm.value.id).subscribe(
      (response: any) => { 
        console.log("deleteUser ", response)
        this.deleteUserForm.reset();
      },
      (error: HttpErrorResponse) => {
        alert(error.message);
      }
    );
  }

}

