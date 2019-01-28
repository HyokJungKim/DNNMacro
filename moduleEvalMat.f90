MODULE moduleEvalMat
CONTAINS
    ! ATTRIBUTES(device) SUBROUTINE ReLU(Vec, VecSize)
    !     INTEGER, intent(in) :: VecSize
    !     REAL, intent(inout) :: Vec(VecSize)

    !     INTEGER :: ii

    !     DO ii = 1,VecSize
    !         IF (Vec(ii) < 0) THEN
    !             Vec(ii) = 0
    !         END IF
    !     END DO
    ! END SUBROUTINE ReLU

    ! ATTRIBUTES(device) SUBROUTINE LeakyReLU(Vec, VecSize)
    !     INTEGER, intent(in) :: VecSize
    !     REAL, intent(inout) :: Vec(VecSize)

    !     REAL :: alpha = 2.0
    !     INTEGER :: ii

    !     DO ii = 1,VecSize
    !         IF (Vec(ii) < 0) THEN
    !             Vec(ii) = alpha*Vec(ii)
    !         END IF
    !     END DO
    ! END SUBROUTINE LeakyReLU

    ! ATTRIBUTES(device) SUBROUTINE BNTransform(Vec, VecSize, bn_gamma, bn_beta, bn_mean, bn_var)
    !     INTEGER, intent(in) :: VecSize
    !     REAL, intent(in) :: bn_gamma(VecSize), bn_beta(VecSize)
    !     REAL, intent(in) :: bn_mean(VecSize), bn_var(VecSize)
        
    !     REAL :: bn_std(VecSize)
    !     REAL :: epsilon_bn = 0.0001
        
    !     REAL, intent(inout) :: Vec(VecSize)
        
    !     bn_std = SQRT(bn_var + epsilon_bn)
        
    !     Vec = bn_gamma*(Vec - bn_mean)/bn_std + bn_beta
    ! END SUBROUTINE BNTransform

    ! ATTRIBUTES(device) SUBROUTINE GetOutput(AADD,Weight,NNwidth,NNdepth,NNinput,bias,VV)
    !     ! PART (0): VARIABLE DECLARATION
    !     ! -----------------------------------------------------------------------------
    !     ! Inputs
    !     ! -----------------------------------------------------------------------------
    !     INTEGER, intent(in) :: NNwidth   ! Width of neural network
    !     INTEGER, intent(in) :: NNdepth   ! Depth of neural network
    !     INTEGER, intent(in) :: NNinput   ! Inputs # of neural network
    !     REAL, intent(in) :: bias       ! Bias parameter for output
    !     REAL, intent(in) :: AADD(2)
    !     REAL, intent(in) :: Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1, NNwidth)
        
    !     ! -----------------------------------------------------------------------------
    !     ! Interim variables
    !     ! -----------------------------------------------------------------------------
    !     REAL :: tempvec(NNwidth)
    !     REAL :: bn_gamma(NNwidth)
    !     REAL :: bn_beta(NNwidth)
    !     REAL :: bn_mean(NNwidth)
    !     REAL :: bn_var(NNwidth)
        
    !     INTEGER :: ii, STA_IDX, END_IDX
        
    !     ! -----------------------------------------------------------------------------
    !     ! Outputs
    !     ! -----------------------------------------------------------------------------
    !     REAL, intent(out) :: VV
        
    !     ! STEP 1: Initial stage
    !     tempvec = MATMUL(AADD, Weight(1:NNinput,:))
    !     CALL LeakyReLU(tempvec, NNwidth)
        
    !     bn_gamma = Weight(NNinput+1,:)
    !     bn_beta  = Weight(NNinput+2,:)
    !     bn_mean  = Weight(NNinput+3,:)
    !     bn_var   = Weight(NNinput+4,:)
        
    !     CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
        
    !     ! STEP 2: 
    !     DO ii = 1,NNdepth
    !         IF (ii == NNdepth) THEN
    !             VV = DOT_PRODUCT(tempvec, Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1,:))
    !             VV = VV + bias
                
    !             ! Last ReLU transformation for output vector
    !             IF (VV < 0) THEN
    !                 VV = 0
    !             END IF
    !         ELSE
    !             ! ReLU activation
    !             STA_IDX = NNinput+4+1+(NNwidth*2+4*2)*(ii-1)
    !             END_IDX = STA_IDX + NNwidth - 1
    !             tempvec = MATMUL(tempvec, Weight(STA_IDX:END_IDX,:))
    !             CALL ReLU(tempvec, NNwidth)
                
    !             ! Batch normalization
    !             STA_IDX = END_IDX + 1
    !             bn_gamma = Weight(STA_IDX,:)
    !             bn_beta  = Weight(STA_IDX+1,:)
    !             bn_mean  = Weight(STA_IDX+2,:)
    !             bn_var   = Weight(STA_IDX+3,:)
    !             CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
                
    !             ! Leaky ReLU normalization
    !             STA_IDX = END_IDX + 5
    !             END_IDX = STA_IDX + NNwidth-1
    !             tempvec = MATMUL(tempvec, Weight(STA_IDX:END_IDX,:))
    !             CALL LeakyReLU(tempvec, NNwidth)
                
    !             ! Batch normalization
    !             STA_IDX = END_IDX + 1
    !             bn_gamma = Weight(STA_IDX,:)
    !             bn_beta  = Weight(STA_IDX+1,:)
    !             bn_mean  = Weight(STA_IDX+2,:)
    !             bn_var   = Weight(STA_IDX+3,:)
    !             CALL BNTransform(tempvec, NNwidth, bn_gamma, bn_beta, bn_mean, bn_var)
    !         END IF
    !     END DO
    ! END SUBROUTINE GetOutput

    ATTRIBUTES(global) SUBROUTINE routineEvalMat(evalmat_cuda, gridsizeC, gridsizeA, FunExog, Weight, lb_C, lb_A, NNwidth, NNdepth, NNinput, bias, ssigma, discnt)
        IMPLICIT NONE
        ! Import CUDA matrix
        REAL, intent(inout) :: evalmat_cuda(:,:)
        
        ! Inputs as values
        INTEGER, value :: NNwidth, NNdepth, NNinput
        REAL, value :: gridsizeC, gridsizeA, bias, lb_C, lb_A, ssigma, discnt

        REAL, device :: FunExog(4)
        REAL, device :: Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1,NNwidth)
        
        ! Other variables
        INTEGER :: jj, kk, nn(2), gridChoice(2)
        REAL :: Vals(3), AADD(2), VV, CC, UU
        
        ! Replace the do-loop
        jj = blockDim%x*(blockIdx%x-1) + threadIdx%x
        kk = blockDim%y*(blockIdx%y-1) + threadIdx%y

        nn(1) = size(evalmat_cuda,1)
        nn(2) = size(evalmat_cuda,2)

        IF (jj <= nn(1) .and. kk<=nn(2)) THEN
            gridChoice(1) = lb_C + (jj-1)*gridsizeC
            gridChoice(2) = lb_A + (kk-1)*gridsizeA

            ! C(t) = ( Y(t) + D(t-1) )*c(t)
            Vals(1) = (FunExog(1) + FunExog(4))*gridChoice(1)
            ! A(t) = ( Y(t) + D(t-1) + R(t)*A(t-1) - C(t) )*a(t)
            Vals(2) = (FunExog(1) + FunExog(4) + FunExog(2)*FunExog(3) - Vals(1))*gridChoice(2)
            ! D(t) = ( Y(t) + D(t-1) + R(t)*A(t-1) - C(t) )*(1-a(t))
            Vals(3) = (FunExog(1) + FunExog(4) + FunExog(2)*FunExog(3) - Vals(1))*(1-gridChoice(2))

            CC = Vals(1)
            AADD = Vals(2:3)

            UU = (CC**(1-ssigma))/(1-ssigma)
            VV = 0
            !CALL GetOutput(AADD,Weight,NNwidth,NNdepth,NNinput,bias,VV)

            evalmat_cuda(jj,kk) = UU + discnt*VV
        END IF

    END SUBROUTINE routineEvalMat
END MODULE moduleEvalMat