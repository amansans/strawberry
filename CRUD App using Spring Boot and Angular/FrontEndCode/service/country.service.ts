import { HttpClient } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { Observable, throwError } from "rxjs";
import { catchError,tap  } from "rxjs/operators";
import { environment } from "src/environments/environment";
import { country } from "../interface/country";
import { user_details } from "../interface/user_details";

@Injectable({
    providedIn:'root'
})

export class CountryService {

    apiServerUrl = environment.apiBaseUrl;

    constructor(private http: HttpClient ) { }

    public getCountryList(): Observable<country[]> {
        return this.http.get<country[]>(`${this.apiServerUrl}/country/GetCountryList/`)
    }
    
    public addUser(user: user_details): Observable<user_details> {
        return this.http.post<user_details>(`${this.apiServerUrl}/adduser/`, user )
    }
} 