import { HttpClient, HttpHeaders } from "@angular/common/http";
import { HtmlParser } from "@angular/compiler";
import { Injectable } from "@angular/core";
import { Observable } from "rxjs";
import { environment } from "src/environments/environment";
import { user_details } from "../interface/user_details";

@Injectable({
    providedIn:'root'
})

export class UserService {

    apiServerUrl = environment.apiBaseUrl;

    constructor(private http: HttpClient) { }

    public getUserList(): Observable<user_details[]> {
        return this.http.get<user_details[]>(`${this.apiServerUrl}/getUserList`)
    }

    public addUser(user:any): Observable<any> {
        const httpHeaders = new HttpHeaders();
        httpHeaders.append('content-type','application/json');
        return this.http.post<any>(`${this.apiServerUrl}/adduser`,user, {headers:httpHeaders})
    }

    public updateUser(user:any,id:number): Observable<any> {
        const httpHeaders = new HttpHeaders();
        httpHeaders.append('content-type','application/json');
        return this.http.put<any>(`${this.apiServerUrl}/updateUser/ + ${id}`,user, {headers:httpHeaders})
    }

    public deleteUser(id:number): Observable<any> {
        const httpHeaders = new HttpHeaders();
        httpHeaders.append('content-type','application/json');
        return this.http.delete<any>(`${this.apiServerUrl}/removeUser/ + ${id}`, {headers:httpHeaders})
    }
} 