SUBROUTINE LeakyReLU(Vec, VecSize)
    INTEGER, intent(in) :: VecSize
    REAL, intent(inout) :: Vec(VecSize)
    
    REAL :: alpha = 2.0
    INTEGER :: ii
    
    DO ii = 1,VecSize
        IF (Vec(ii) < 0) THEN
            Vec(ii) = alpha*Vec(ii)
        END IF
    END DO
END SUBROUTINE LeakyReLU
