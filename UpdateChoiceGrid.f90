!  UpdateChoiceGrid.f90 
!
!  FUNCTIONS/SUBROUTINES exported from FortranUpdateChoiceGrid.dll:
!  FortranUpdateChoiceGrid - subroutine 
!
subroutine FortranUpdateChoiceGrid(cWeight,bias,cFunExog,cparamUC,cGridC,cpaddingC,cGridA,cpaddingA,NNwidth,NNdepth,NNinput,NNstep,outChoice,outVal) BIND(C, name='FortranUpdateChoiceGrid')
    !use, intrinsic :: ISO_C_BINDING, only: c_associated, c_loc, c_ptr, C_INT, C_FLOAT
USE, intrinsic :: ISO_C_BINDING
    ! Expose subroutine FortranUpdateChoiceGrid to users of this DLL
    !DEC$ ATTRIBUTES DLLEXPORT::FortranUpdateChoiceGrid
    IMPLICIT NONE
    
    ! PART (0): VARIABLE DECLARATION
    ! -----------------------------------------------------------------------------
    ! Inputs: elementwise
    ! -----------------------------------------------------------------------------
    INTEGER (C_INT), intent(in), value :: NNwidth   ! Width of neural network
    INTEGER (C_INT), intent(in), value :: NNdepth   ! Depth of neural network
    INTEGER (C_INT), intent(in), value :: NNinput   ! Inputs # of neural network
    INTEGER (C_INT), intent(in), value :: NNstep    ! Steps # to search
    REAL (C_FLOAT), intent(in), value :: bias       ! Bias parameter for output
    
    ! -----------------------------------------------------------------------------
    ! Inputs: vectors and matrices
    ! -----------------------------------------------------------------------------
    TYPE(C_PTR), target, intent(in), value :: cWeight   ! Weights of neural network
    
    TYPE(C_PTR), target, intent(in), value :: cFunExog  ! Exogenous variables
    TYPE(C_PTR), target, intent(in), value :: cparamUC  ! Collection of parameters
    
    TYPE(C_PTR), target, intent(in), value :: cGridC    ! Grid for choice variable c(t)
    TYPE(C_PTR), target, intent(in), value :: cpaddingC ! Padding for choice variable c(t)
    TYPE(C_PTR), target, intent(in), value :: cGridA    ! Grid for choice variable a(t)
    TYPE(C_PTR), target, intent(in), value :: cpaddingA ! Padding for choice variable a(t)
    
    ! Pointers for vectors and matrices
    INTEGER, POINTER :: pGridC(:), pGridA(:)
    REAL, POINTER :: pWeight(:), pFunExog(:), pparamUC(:), ppaddingC(:), ppaddingA(:)
    
    REAL :: Weight(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1,NNwidth)
    REAL :: FunExog(4)
    REAL :: paramUC(2)
    INTEGER :: GridC(NNstep)
    REAL :: paddingC(NNstep)
    INTEGER :: GridA(NNstep)
    REAL :: paddingA(NNstep)
    
    ! -----------------------------------------------------------------------------
    ! Interim Variables
    ! -----------------------------------------------------------------------------
    REAL :: lb_C, ub_C, lb_A, ub_A, ssigma, discnt
    REAL :: gridsizeC, gridsizeA, Ccenter, Acenter
    REAL :: gridChoice(2)
    REAL :: Vals(3)
    REAL :: AADD(2)
    REAL :: CC, VV, UU
    REAL, ALLOCATABLE :: evalmat(:,:)
    
    INTEGER :: temprow, tempcol, ii, jj, kk
    INTEGER :: MAXIDX(2)
    
    ! -----------------------------------------------------------------------------
    ! Outputs
    ! -----------------------------------------------------------------------------
    REAL (C_FLOAT), intent(out) :: outChoice(2)
    REAL (C_FLOAT), intent(out) :: outVal
    
    ! -----------------------------------------------------------------------------
    ! Allocate pointers
    ! -----------------------------------------------------------------------------
    CALL C_F_POINTER(cWeight, pWeight, [(NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1)*NNwidth])
    Weight = RESHAPE(pWeight, (/NNinput+4*(2*NNdepth-1)+NNwidth*2*(NNdepth-1)+1, NNwidth/))
    
    CALL C_F_POINTER(cFunExog, pFunExog, [4])
    FunExog = RESHAPE(pFunExog, (/4/))
    
    CALL C_F_POINTER(cparamUC, pparamUC, [2])
    paramUC = RESHAPE(pparamUC, (/2/))
    
    CALL C_F_POINTER(cGridC, pGridC, [NNstep])
    GridC = RESHAPE(pGridC, (/NNstep/))
    
    CALL C_F_POINTER(cpaddingC, ppaddingC, [NNstep])
    paddingC = RESHAPE(ppaddingC, (/NNstep/))
    
    CALL C_F_POINTER(cGridA, pGridA, [NNstep])
    GridA = RESHAPE(pGridA, (/NNstep/))
    
    CALL C_F_POINTER(cpaddingA, ppaddingA, [NNstep])
    paddingA = RESHAPE(ppaddingA, (/NNstep/))
    
    ! PART (1): Draw Grid and Search for New Optimal Policy    
    IF (FunExog(1) + FunExog(3) + FunExog(4) < 0.00000001) THEN
        outChoice(1) = 1
        outChoice(2) = 0
        outVal = 0
    ELSE
        lb_C = 0
        ub_C = 1
        lb_A = 0
        ub_A = 1
        
        ssigma = paramUC(1)
        discnt = paramUC(2)
        
        DO ii = 1,NNstep
            gridsizeC = (ub_C - lb_C)/(GridC(ii) - 1)
            gridsizeA = (ub_A - lb_A)/(GridA(ii) - 1)
            
            ALLOCATE( evalmat(GridC(ii),GridA(ii)) )
            
            DO jj = 1,GridC(ii)
                gridChoice(1) = lb_C + (jj-1)*gridsizeC
			DO kk = 1,GridA(ii)
                gridChoice(2) = lb_A + (kk-1)*gridsizeA
                
				CALL Pct2Val(gridChoice, FunExog, Vals)
                
                CC = Vals(1)
                
                AADD = Vals(2:3)
                
                CALL GetOutput(AADD,Weight,NNwidth,NNdepth,NNinput,bias,VV)
                
                UU = (CC**(1-ssigma))/(1-ssigma)
                
                evalmat(jj,kk) = UU + discnt*VV
			END DO
            END DO
            
            outVal = MAXVAL(evalmat)
            MAXIDX = MAXLOC(evalmat)
            
            temprow = MAXIDX(1)
            tempcol = MAXIDX(2)
            
            Ccenter = lb_C + (temprow-1)*gridsizeC
            Acenter = lb_A + (tempcol-1)*gridsizeA
            
            ub_C = MIN(Ccenter + paddingC(ii)*gridsizeC, 1.0)
            lb_C = MAX(Ccenter - paddingC(ii)*gridsizeC, 0.0)
        
            ub_A = MIN(Acenter + paddingA(ii)*gridsizeA, 1.0)
            lb_A = MAX(Acenter - paddingA(ii)*gridsizeA, 0.0)
            
            DEALLOCATE(evalmat)
        END DO
	END IF
	
    outChoice(1) = Ccenter
    outChoice(2) = Acenter
END SUBROUTINE FortranUpdateChoiceGrid