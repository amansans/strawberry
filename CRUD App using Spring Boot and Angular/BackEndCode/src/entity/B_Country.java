package com.amansans.BasicSBProject.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.NotEmpty;
import java.util.Set;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class B_Country {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long countryID;

    @NotEmpty
    private String countryName;

    @OneToMany(orphanRemoval = true,cascade = CascadeType.PERSIST)
    private Set<B_UserDetails> userDetails;

}
