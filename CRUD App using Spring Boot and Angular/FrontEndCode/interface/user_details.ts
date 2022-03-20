import { country } from "./country";

export interface user_details{
    ID: number;
    firstName: string;
    lastName: string;
    emailId: string;
    country : country;
}