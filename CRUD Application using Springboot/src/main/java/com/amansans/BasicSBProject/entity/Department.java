package com.amansans.BasicSBProject.entity;

import lombok.*;
import org.hibernate.validator.constraints.Length;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.validation.constraints.*;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Department {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long departmentId;

    //@Length(max = 5,min = 1,message = "length should be between 1 to 5")
    //@Positive
    //@PositiveOrZero
    //@Negative
    //@NegativeOrZero
    //@Past
    //@PastOrPresent
    //@Future
    //@FutureOrPresent
    //@Email

    @NotBlank(message = "Please add Department Name")
    private String departmentName;
    private String departmentAddress;
    private String departmentCode;

}
