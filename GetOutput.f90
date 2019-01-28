SUBROUTINE GetOutput(AADD,Weight,NNwidth,NNdepth,NNinput,bias,VV)
    ! PART (0): VARIABLE DECLARATION
    ! -----------------------------------------------------------------------------
    ! Inputs
    ! -----------------------------------------------------------------------------
    INTEGER, intent(in) :: NNwidth   ! Width of neural network
    INTEGER, intent(in) :: NNdepth   ! Depth of neural network
    INTEGER, intent(in) :: NNinput   ! Inputs # of neural network
    REAL, intent(in) :: bias       ! Bias parameter for output
    REAL, intent(in) :: AADD(2)
    REAL, intent(in) :: Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1, NNwidth)
    
    ! -----------------------------------------------------------------------------
    ! Interim variables
    ! -----------------------------------------------------------------------------
    REAL :: tempvec(NNwidth)
    REAL :: bn_gamma(NNwidth)
    REAL :: bn_beta(NNwidth)
    REAL :: bn_mean(NNwidth)
    REAL :: bn_var(NNwidth)
    
    INTEGER :: ii, STA_IDX, END_IDX
    
    ! -----------------------------------------------------------------------------
    ! Outputs
    ! -----------------------------------------------------------------------------
    REAL, intent(out) :: VV
    
    ! STEP 1: Initial stage
    tempvec = MATMUL(AADD, Weight(1:NNinput,:))
    CALL LeakyReLU(tempvec, NNwidth)
    
    bn_gamma = Weight(NNinput+1,:)
    bn_beta  = Weight(NNinput+2,:)
    bn_mean  = Weight(NNinput+3,:)
    bn_var   = Weight(NNinput+4,:)
    
    CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
    
    ! STEP 2: 
    DO ii = 1,NNdepth
        IF (ii == NNdepth) THEN
            VV = DOT_PRODUCT(tempvec, Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1,:))
            VV = VV + bias
            
            ! Last ReLU transformation for output vector
            IF (VV < 0) THEN
                VV = 0
            END IF
        ELSE
            ! ReLU activation
            STA_IDX = NNinput+4+1+(NNwidth*2+4*2)*(ii-1)
            END_IDX = STA_IDX + NNwidth - 1
            tempvec = MATMUL(tempvec, Weight(STA_IDX:END_IDX,:))
            CALL ReLU(tempvec, NNwidth)
            
            ! Batch normalization
            STA_IDX = END_IDX + 1
            bn_gamma = Weight(STA_IDX,:)
            bn_beta  = Weight(STA_IDX+1,:)
            bn_mean  = Weight(STA_IDX+2,:)
            bn_var   = Weight(STA_IDX+3,:)
            CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
            
            ! Leaky ReLU normalization
            STA_IDX = END_IDX + 5
            END_IDX = STA_IDX + NNwidth-1
            tempvec = MATMUL(tempvec, Weight(STA_IDX:END_IDX,:))
            CALL LeakyReLU(tempvec, NNwidth)
            
            ! Batch normalization
            STA_IDX = END_IDX + 1
            bn_gamma = Weight(STA_IDX,:)
            bn_beta  = Weight(STA_IDX+1,:)
            bn_mean  = Weight(STA_IDX+2,:)
            bn_var   = Weight(STA_IDX+3,:)
            CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
        END IF
    END DO
END SUBROUTINE GetOutput