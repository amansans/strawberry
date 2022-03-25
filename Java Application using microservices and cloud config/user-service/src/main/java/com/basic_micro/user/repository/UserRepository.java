package com.basic_micro.user.repository;

import com.basic_micro.user.entity.User;
import org.hibernate.query.criteria.internal.ValueHandlerFactory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, ValueHandlerFactory.LongValueHandler> {
    public User findByUserId(Long userId);
}
