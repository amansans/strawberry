package com.basic_micro.user.service;

import com.basic_micro.user.VO.Department;
import com.basic_micro.user.VO.ResponseTemplateVO;
import com.basic_micro.user.entity.User;
import com.basic_micro.user.error.UserNotFoundException;
import com.basic_micro.user.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
@Slf4j
public class UserServiceImpl implements UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private RestTemplate restTemplate;

    @Override
    public User saveUser(User user) {
        log.info("Entered saveUser function for UserService Implementer");
        return userRepository.save(user);
    }

    @Override
    public ResponseTemplateVO getUserWithDepartment(Long userId) throws UserNotFoundException {
        ResponseTemplateVO vo = new ResponseTemplateVO();
        User user = userRepository.findByUserId(userId);

        if(user == null) throw new UserNotFoundException("User with userId " + userId + " does not exist");

        Department department = restTemplate.getForObject("http://DEPARTMENT-SERVICE/departments/"+ user.getDepartmentId(),Department.class);

        vo.setUser(user);
        vo.setDepartment(department);

        return vo;
    }
}
