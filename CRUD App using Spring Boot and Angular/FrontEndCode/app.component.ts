import { HttpErrorResponse } from '@angular/common/http';
import { BuiltinFunctionCall } from '@angular/compiler/src/compiler_util/expression_converter';
import { Component, OnInit } from '@angular/core';
import { country } from './interface/country';
import { user_details } from './interface/user_details';
import { CountryService } from './service/country.service';
import {Router} from '@angular/router'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent {
  
  constructor(private router: Router) {}

  goToPage(pagename:string) : void{
    this.router.navigate([`${pagename}`])
    console.log('Clicked on',`${pagename}`)
  }
}