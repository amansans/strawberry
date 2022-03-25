package com.basic_micro.user.service;

import com.basic_micro.user.VO.ResponseTemplateVO;
import com.basic_micro.user.entity.User;
import com.basic_micro.user.error.UserNotFoundException;

public interface UserService {
    public User saveUser(User user);

    public ResponseTemplateVO getUserWithDepartment(Long userId) throws UserNotFoundException;
}
