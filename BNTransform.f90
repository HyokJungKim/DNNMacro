SUBROUTINE BNTransform(Vec, VecSize, bn_gamma, bn_beta, bn_mean, bn_var)
    INTEGER, intent(in) :: VecSize
    REAL, intent(in) :: bn_gamma(VecSize), bn_beta(VecSize)
    REAL, intent(in) :: bn_mean(VecSize), bn_var(VecSize)
    
    REAL :: bn_std(VecSize)
    REAL :: epsilon_bn = 0.0001
    
    REAL, intent(inout) :: Vec(VecSize)
    
    bn_std = SQRT(bn_var + epsilon_bn)
    
    Vec = bn_gamma*(Vec - bn_mean)/bn_std + bn_beta
END SUBROUTINE BNTransform