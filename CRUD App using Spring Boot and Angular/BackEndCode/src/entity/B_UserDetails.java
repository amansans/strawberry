package com.amansans.BasicSBProject.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import java.util.List;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class B_UserDetails {

    @Id
    @SequenceGenerator(
            name = "user_id_sequence_g",
            sequenceName = "user_id_sequence_g",
            allocationSize = 1
    )

    @GeneratedValue (
            strategy = GenerationType.SEQUENCE,
            generator = "user_id_sequence_g"
    )
    private Long userId;

    @NotBlank
    private String firstName;

    @NotBlank
    private String lastName;

    @Email
    @Column(unique = true)
    private String emailId;

    @ManyToOne (
            cascade = CascadeType.PERSIST,
            fetch = FetchType.EAGER,
            optional = false
    )
    @JoinColumn(
            name = "country_id",
            referencedColumnName = "countryID"
    )
    private B_Country country;

}
