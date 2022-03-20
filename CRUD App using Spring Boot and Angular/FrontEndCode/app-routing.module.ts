import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import {UsersComponent} from '../app/users/users.component';
import { AppComponent } from './app.component';
import { HomepageComponent } from './homepage/homepage.component';

const routes: Routes = [
  {path:'', component: HomepageComponent},
  {path:'Homepage', component: HomepageComponent},
  {path:'Users',component: UsersComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes),FormsModule],
  exports: [RouterModule]
})
export class AppRoutingModule { }
