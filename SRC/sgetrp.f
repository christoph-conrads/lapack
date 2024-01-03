*> \brief \b SGETRP computes the transpose of a matrix.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download SGETRP + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgetrp.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgetrp.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgetrp.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE SGETRP( M, N, S, LDS, D, LDD, INFO )
*
*       .. Scalar Arguments ..
*       INTEGER            M, N, LDD, LDS, INFO
*       ..
*       .. Array Arguments ..
*       REAL               S( LDS, * ), D( LDD, * )
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> SGETRP computes the transpose of the real M-by-N source matrix S and
*> stores the result in the real N-by-M destination matrix D.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix S and the number of columns
*>          of the matrix D.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrix S and the number of rows
*>          of the matrix D.
*> \endverbatim
*>
*> \param[in] S
*> \verbatim
*>          S is REAL array, dimension ( LDS, N )
*>          The matrix to be transposed.
*> \endverbatim
*>
*> \param[in] LDS
*> \verbatim
*>          LDS is INTEGER
*>          The leading dimension of S. LDS >= MAX(1,M).
*> \endverbatim
*>
*> \param[out] D
*> \verbatim
*>          D is REAL array, dimension
*>          On exit, D contains the transpose of S.
*> \endverbatim
*>
*> \param[in] LDD
*> \verbatim
*>          LDD is INTEGER
*>          The leading dimension of D. LDD >= MAX(1,N).
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0: successful exit
*>          < 0: if INFO = -i, the i-th argument has an illegal value
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Christoph Conrads
*
*  =====================================================================
      SUBROUTINE SGETRP( M, N, S, LDS, D, LDD, INFO )
*
*  -- LAPACK computational routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd. --
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            M, N, LDS, LDD, INFO
*     ..
*     .. Array Arguments ..
      REAL               S( LDS, * ), D( LDD, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            B
*     Assumed 64 byte cache line size divided by four bytes per REAL
      PARAMETER          ( B = 16 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J, K, L
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDS.LT.MAX( 1, M ) ) THEN
         INFO = -4
      ELSE IF( LDD.LT.MAX( 1, N ) ) THEN
         INFO = -6
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'SGETRP', -INFO )
         RETURN
      END IF
*
*     Transpose matrix
*
      DO J = 1, N + B - 1, B
         DO I = 1, M + B - 1, B
            DO L = J, MIN( J + B, N )
               DO K = I, MIN( I + B, M )
                  D( L, K ) = S( K, L )
               END DO
            END DO
         END DO
      END DO
*
      RETURN
*
*     End of SGETRP
*
      END
