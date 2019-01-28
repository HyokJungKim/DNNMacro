SUBROUTINE Pct2Val(gridChoice, FunExog, Vals)
    REAL, intent(in) :: gridChoice(2)
    REAL, intent(in) :: FunExog(4)
    REAL, intent(out) :: Vals(3)
    
    ! C(t) = ( Y(t) + D(t-1) )*c(t)
    Vals(1) = (FunExog(1) + FunExog(4))*gridChoice(1)
    ! A(t) = ( Y(t) + D(t-1) + R(t)*A(t-1) - C(t) )*a(t)
    Vals(2) = (FunExog(1) + FunExog(4) + FunExog(2)*FunExog(3) - Vals(1))*gridChoice(2)
    ! D(t) = ( Y(t) + D(t-1) + R(t)*A(t-1) - C(t) )*(1-a(t))
    Vals(3) = (FunExog(1) + FunExog(4) + FunExog(2)*FunExog(3) - Vals(1))*(1-gridChoice(2))
END SUBROUTINE Pct2Val