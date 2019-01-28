SUBROUTINE ReLU(Vec, VecSize)
    INTEGER, intent(in) :: VecSize
    REAL, intent(inout) :: Vec(VecSize)
    
    INTEGER :: ii
    
    DO ii = 1,VecSize
        IF (Vec(ii) < 0) THEN
            Vec(ii) = 0
        END IF
    END DO
END SUBROUTINE ReLU