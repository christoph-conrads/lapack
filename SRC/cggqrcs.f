*> \brief <b> CGGQRCS computes the singular value decomposition (SVD) for OTHER matrices</b>
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download CGGQRCS + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cggqrcs.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cggqrcs.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cggqrcs.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE CGGQRCS( JOBU1, JOBU2, JOBX, M, N, P, L, SWAPPED,
*                           A, LDA, B, LDB,
*                           ALPHA, BETA,
*                           U1, LDU1, U2, LDU2
*                           WORK, LWORK, RWORK, LRWORK, IWORK, INFO )
*
*       .. Scalar Arguments ..
*       CHARACTER          JOBU1, JOB2, JOBX
*       INTEGER            INFO, LDA, LDB, LDU1, LDU2, M, N, P, L,
*                          LWORK, LRWORK
*       ..
*       .. Array Arguments ..
*       INTEGER            IWORK( * )
*       REAL               ALPHA( N ), BETA( N ), RWORK( * )
*       COMPLEX            A( LDA, * ), B( LDB, * ),
*      $                   U1( LDU1, * ), U2( LDU2, * ),
*      $                   WORK( * )
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> CGGQRCS computes the generalized singular value decomposition (GSVD)
*> of an M-by-N complex matrix A and P-by-N complex matrix B:
*>
*>       A = U1 * D1 * X,           B = U2 * D2 * X
*>
*> where U1 and U2 are unitary matrices. CGGQRCS uses the QR
*> factorization with column pivoting and the 2-by-1 CS decomposition to
*> compute the GSVD.
*>
*> Let L be the effective numerical rank of the matrix (A**H,B**H)**H,
*> then X is a L-by-N nonsingular matrix, D1 and D2 are M-by-L and
*> P-by-L "diagonal" matrices. If SWAPPED is false, then D1 and D2 are
*> of the of the following structures, respectively:
*>
*>                 K1  K
*>            K1 [ I   0   0 ]
*>       D1 = K  [ 0   C   0 ]
*>               [ 0   0   0 ]
*>
*>                     K   K2
*>               [ 0   0   0 ]
*>       D2 = K  [ 0   S   0 ]
*>            K2 [ 0   0   I ]
*>
*> where
*>
*>   K  = MIN(M, P, L, M + P - L),
*>   K1 = MAX(L - P, 0),
*>   K2 = MAX(L - M, 0),
*>   C  = diag( ALPHA(1), ..., ALPHA(K) ),
*>   S  = diag( BETA(1), ..., BETA(K) ), and
*>   C^2 + S^2 = I.
*>
*> If SWAPPED is true, then D1 and D2 are of the of the following
*> structures, respectively:
*>
*>                     K   K1
*>               [ 0   0   0 ]
*>       D1 = K  [ 0   S   0 ]
*>            K1 [ 0   0   I ]
*>
*>                 K2  K
*>            K2 [ I   0   0 ]
*>       D2 = K  [ 0   C   0 ]
*>               [ 0   0   0 ]
*>
*> where
*>
*>   S  = diag( ALPHA(1), ..., ALPHA(K) ),
*>   C  = diag( BETA(1), ..., BETA(K) ), and
*>   C^2 + S^2 = I.
*>
*> The routine computes C, S and optionally the matrices U1, U2, and X.
*> On exit, X is stored in WORK( 2:L*N+1 ).
*>
*> If B is an N-by-N nonsingular matrix, then the GSVD of the matrix
*> pair (A, B) implicitly gives the SVD of A*inv(B):
*>
*>       A*inv(B) = U1*(D1*inv(D2))*U2**H.
*>
*> If (A**H,B**H)**H  has orthonormal columns, then the GSVD of A and B
*> is also equal to the CS decomposition of A and B. Furthermore, the
*> GSVD can be used to derive the solution of the eigenvalue problem:
*>
*>       A**H*A x = lambda * B**H*B x.
*>
*> In some literature, the GSVD of A and B is presented in the form
*>
*>       A = U1*D1*( 0 R )*Q**H,    B = U2*D2*( 0 R )*Q**H
*>
*> where U1, U2, and Q are unitary matrices. This latter GSVD form is
*> computed directly by DGGSVD3. It is possible to convert between the
*> two representations by calculating the RQ decomposition of X but this
*> is not recommended for reasons of numerical stability.
*>
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] JOBU1
*> \verbatim
*>          JOBU1 is CHARACTER*1
*>          = 'Y':  Orthogonal matrix U1 is computed;
*>          = 'N':  U1 is not computed.
*> \endverbatim
*>
*> \param[in] JOBU2
*> \verbatim
*>          JOBU2 is CHARACTER*1
*>          = 'Y':  Orthogonal matrix U2 is computed;
*>          = 'N':  U2 is not computed.
*> \endverbatim
*>
*> \param[in] JOBX
*> \verbatim
*>          JOBX is CHARACTER*1
*>          = 'Y':  Matrix X is computed;
*>          = 'N':  X is not computed.
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix A.  M >= 1.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrices A and B.  N >= 1.
*> \endverbatim
*>
*> \param[in] P
*> \verbatim
*>          P is INTEGER
*>          The number of rows of the matrix B.  P >= 1.
*> \endverbatim
*>
*> \param[out] L
*> \verbatim
*>          L is INTEGER
*>          On exit, the effective numerical rank of the matrix
*>          (A**H, B**H)**H.
*> \endverbatim
*>
*> \param[out] SWAPPED
*> \verbatim
*>          L is LOGICAL
*>          On exit, SWAPPED is true if CGGQRCS swapped the input
*>          matrices A, B and computed the GSVD of (B, A); false
*>          otherwise.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is COMPLEX array, dimension (LDA,N)
*>          On entry, the M-by-N matrix A.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A. LDA >= max(1,M).
*> \endverbatim
*>
*> \param[in,out] B
*> \verbatim
*>          B is COMPLEX array, dimension (LDB,N)
*>          On entry, the P-by-N matrix B.
*> \endverbatim
*>
*> \param[in] LDB
*> \verbatim
*>          LDB is INTEGER
*>          The leading dimension of the array B. LDB >= max(1,P).
*> \endverbatim
*>
*> \param[out] ALPHA
*> \verbatim
*>          ALPHA is REAL array, dimension (N)
*> \endverbatim
*>
*> \param[out] BETA
*> \verbatim
*>          BETA is REAL array, dimension (N)
*>
*>          On exit, ALPHA and BETA contain the K generalized singular
*>          value pairs of A and B.
*> \endverbatim
*>
*> \param[out] U1
*> \verbatim
*>          U1 is COMPLEX array, dimension (LDU1,M)
*>          If JOBU1 = 'Y', U1 contains the M-by-M unitary matrix U1.
*>          If JOBU1 = 'N', U1 is not referenced.
*> \endverbatim
*>
*> \param[in] LDU1
*> \verbatim
*>          LDU1 is INTEGER
*>          The leading dimension of the array U1. LDU1 >= max(1,M) if
*>          JOBU1 = 'Y'; LDU1 >= 1 otherwise.
*> \endverbatim
*>
*> \param[out] U2
*> \verbatim
*>          U2 is COMPLEX array, dimension (LDU2,P)
*>          If JOBU2 = 'Y', U2 contains the P-by-P unitary matrix U2.
*>          If JOBU2 = 'N', U2 is not referenced.
*> \endverbatim
*>
*> \param[in] LDU2
*> \verbatim
*>          LDU2 is INTEGER
*>          The leading dimension of the array U2. LDU2 >= max(1,P) if
*>          JOBU2 = 'Y'; LDU2 >= 1 otherwise.
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is COMPLEX array, dimension (MAX(1,LWORK))
*>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*> \endverbatim
*>
*> \param[in] LWORK
*> \verbatim
*>          LWORK is INTEGER
*>          The dimension of the array WORK.
*>
*>          If LWORK = -1, then a workspace query is assumed; the
*>          routine only calculates the optimal size of the WORK array,
*>          returns this value as the first entry of the WORK array, and
*>          no error message related to LWORK is issued by XERBLA.
*> \endverbatim
*>
*> \param[out] RWORK
*> \verbatim
*>          RWORK is REAL array, dimension (MAX(1,LRWORK))
*> \endverbatim
*>
*> \param[in] LRWORK
*> \verbatim
*>          LRWORK is INTEGER
*>          The dimension of the array RWORK.
*>
*>          If LRWORK = -1, then a workspace query is assumed; the routine
*>          only calculates the optimal size of the RWORK array, returns
*>          this value as the first entry of the work array, and no error
*>          message related to LRWORK is issued by XERBLA.
*> \endverbatim
*>
*> \param[out] IWORK
*> \verbatim
*>          IWORK is INTEGER array, dimension (M + N + P)
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0:  successful exit.
*>          < 0:  if INFO = -i, the i-th argument had an illegal value.
*>          > 0:  CBBCSD did not converge. For further details, see
*>                subroutine CUNCSDBY1.
*> \endverbatim
*
*> \par Internal Parameters:
*  =========================
*>
*> \param[out] W
*> \verbatim
*>          W is REAL
*>          W is a radix power chosen such that the Frobenius norm of A
*>          and W*B are with SQRT(RADIX) and 1/SQRT(RADIX) of each
*>          other.
*> \endverbatim
*>
*> \verbatim
*>  TOL     REAL
*>          Let G = (A**H,B**H)**H. TOL is the threshold to determine
*>          the effective rank of G. Generally, it is set to
*>                   TOL = MAX( M + P, N ) * norm(G) * MACHEPS,
*>          where norm(G) is the Frobenius norm of G.
*>          The size of TOL may affect the size of backward error of the
*>          decomposition.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Christoph Conrads (https://christoph-conrads.name)
*
*> \date October 2019, May 2020
*
*> \ingroup realGEsing
*
*> \par Contributors:
*  ==================
*>
*>     Christoph Conrads (https://christoph-conrads.name)
*>
*
*> \par Further Details:
*  =====================
*>
*>  CGGQRCS should be significantly faster than DGGSVD3 for large
*>  matrices because the matrices A and B are reduced to a pair of
*>  well-conditioned bidiagonal matrices instead of pairs of upper
*>  triangular matrices. On the downside, CGGQRCS requires a much larger
*>  workspace whose dimension must be queried at run-time. CGGQRCS also
*>  offers no guarantees which of the two possible diagonal matrices
*>  is used for the matrix factorization.
*>
*  =====================================================================
      RECURSIVE SUBROUTINE CGGQRCS( JOBU1, JOBU2, JOBX, M, N, P, L,
     $                              SWAPPED,
     $                              A, LDA, B, LDB,
     $                              ALPHA, BETA,
     $                              U1, LDU1, U2, LDU2,
     $                              WORK, LWORK, RWORK, LRWORK, IWORK,
     $                              INFO )
*
*  -- LAPACK driver routine (version 3.7.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd. --
*     September 2016
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      LOGICAL            SWAPPED
      CHARACTER          JOBU1, JOBU2, JOBX
      INTEGER            INFO, LDA, LDB, LDU1, LDU2, L, M, N, P, LWORK,
     $                   LRWKOPT, LRWORK, LRWORK2BY1
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      REAL               ALPHA( N ), BETA( N ), RWORK( * )
      COMPLEX            A( LDA, * ), B( LDB, * ),
     $                   U1( LDU1, * ), U2( LDU2, * ),
     $                   WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      COMPLEX            CONE, CZERO
      PARAMETER          ( CONE = ( 1.0E0, 0.0E0 ),
     $                   CZERO = ( 0.0E0, 0.0E0 ) )
*     .. Local Scalars ..
      LOGICAL            WANTU1, WANTU2, WANTX, LQUERY
      INTEGER            I, J, K, K1, LMAX, Z, LDG, LDX, LDVT, LWKOPT
      REAL               BASE, NAN, NORMA, NORMB, NORMG, TOL, ULP, UNFL,
     $                   THETA, IOTA, W
      COMPLEX            CNAN
*     .. Local Arrays ..
      COMPLEX            G( M + P, N ), VT( N, N )
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      REAL               SLAMCH, CLANGE
      EXTERNAL           LSAME, SLAMCH, CLANGE
*     ..
*     .. External Subroutines ..
      EXTERNAL           CGEMM, CGEQP3, CLACPY, CLAPMT, CLASCL,
     $                   CLASET, CUNGQR, CUNCSD2BY1, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          COS, MAX, MIN, SIN, SQRT
*     ..
*     .. Executable Statements ..
*
*     Decode and test the input parameters
*
      WANTU1 = LSAME( JOBU1, 'Y' )
      WANTU2 = LSAME( JOBU2, 'Y' )
      WANTX = LSAME( JOBX, 'Y' )
      LQUERY = LWORK.EQ.-1 .OR. LRWORK.EQ.-1
      LWKOPT = 1
*
*     Test the input arguments
*
      INFO = 0
      IF( .NOT.( WANTU1 .OR. LSAME( JOBU1, 'N' ) ) ) THEN
         INFO = -1
      ELSE IF( .NOT.( WANTU2 .OR. LSAME( JOBU2, 'N' ) ) ) THEN
         INFO = -2
      ELSE IF( .NOT.( WANTX .OR. LSAME( JOBX, 'N' ) ) ) THEN
         INFO = -3
      ELSE IF( M.LT.1 ) THEN
         INFO = -4
      ELSE IF( N.LT.1 ) THEN
         INFO = -5
      ELSE IF( P.LT.1 ) THEN
         INFO = -6
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -9
      ELSE IF( LDB.LT.MAX( 1, P ) ) THEN
         INFO = -11
      ELSE IF( LDU1.LT.1 .OR. ( WANTU1 .AND. LDU1.LT.M ) ) THEN
         INFO = -15
      ELSE IF( LDU2.LT.1 .OR. ( WANTU2 .AND. LDU2.LT.P ) ) THEN
         INFO = -17
      ELSE IF( LWORK.LT.1 .AND. .NOT.LQUERY ) THEN
         INFO = -19
      END IF
*
*     Make sure A is the matrix smaller in norm
*
      IF( INFO.EQ.0 ) THEN
         NORMA = CLANGE( 'F', M, N, A, LDA, RWORK )
         NORMB = CLANGE( 'F', P, N, B, LDB, RWORK )
*
         IF( NORMA.GT.SQRT( 2.0E0 ) * NORMB ) THEN
            CALL CGGQRCS( JOBU2, JOBU1, JOBX, P, N, M, L,
     $                    SWAPPED,
     $                    B, LDB, A, LDA,
     $                    BETA, ALPHA,
     $                    U2, LDU2, U1, LDU1,
     $                    WORK, LWORK, RWORK, LRWORK, IWORK, INFO )
            SWAPPED = .TRUE.
            RETURN
         ENDIF
*
*     Past this point, we know that
*     * NORMA <= NORMB (almost)
*     * W >= 1
*     * ALPHA will contain cosine values at the end
*     * BETA will contain sine values at the end
*
      END IF
*
*     Initialize variables
*
*     Computing 0.0 / 0.0 directly causes compiler errors
      NAN = 1.0E0
      NAN = 0.0 / (NAN - 1.0E0)
      CNAN = CMPLX( NAN, NAN )
*
      SWAPPED = .FALSE.
      L = 0
      LMAX = MIN( M + P, N )
      Z = ( M + P ) * N
      G = WORK( 1 )
      LDG = M + P
      VT = 0
      LDVT = N
      THETA = NAN
      IOTA = NAN
      W = NAN
*
*     Compute workspace
*
      IF( INFO.EQ.0 ) THEN
         LWKOPT = 0
*
         CALL CGEQP3( M + P, N, G, LDG, IWORK, WORK, WORK, -1, RWORK,
     $                INFO )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
         LWKOPT = INT( WORK( 1 ) )
*
         CALL CUNGQR( M + P, LMAX, LMAX, G, LDG, WORK, WORK, -1, INFO )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
*
         CALL CUNCSD2BY1( JOBU1, JOBU2, JOBX, M + P, M, LMAX,
     $                    G, LDG, G, LDG,
     $                    ALPHA,
     $                    U1, LDU1, U2, LDU2, VT, LDVT,
     $                    WORK, -1, RWORK, LRWORK, IWORK, INFO )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
*        The matrix (A, B) must be stored sequentially for CUNGQR
         LWKOPT = LWKOPT + Z
*        2-by-1 CSD matrix V1 must be stored
         IF( WANTX ) THEN
            LWKOPT = LWKOPT + LDVT*N
         END IF
*        Adjust CUNCSD2BY1 LRWORK for case with maximum memory
*        consumption
         LRWORK2BY1 = INT( RWORK(1) )
*        Select safe xUNCSD2BY1 IBBCSD value
     $                - 9 * MAX( 0, MIN( M, P, N, M+P-N-1 ) )
     $                + 9 * MAX( 1, MIN( M, P, N ) )
*        Select safe xUNCSD2BY1 LBBCSD value
     $                - 8 * MAX( 0, MIN( M, P, N, M+P-N ) )
     $                + 8 * MIN( M, P, N )
         LRWKOPT = MAX( 2*N, LRWORK2BY1 )
*
         WORK( 1 ) = CMPLX( REAL( LWKOPT ), 0.0E0 )
         RWORK( 1 ) = REAL( LRWKOPT )
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'CGGQRCS', -INFO )
         RETURN
      END IF
      IF( LQUERY ) THEN
         RETURN
      ENDIF
*     Finish initialization
      IF( WANTX ) THEN
         VT = WORK( Z + 1 )
      ELSE
         LDVT = 0
      END IF
*
*     Scale matrix A such that norm(A) \approx norm(B)
*
      IF( NORMA.EQ.0.0E0 ) THEN
         W = 1.0E0
      ELSE
         BASE = SLAMCH( 'B' )
         W = BASE ** INT( LOG( NORMB / NORMA ) / LOG( BASE ) )
*
         CALL CLASCL( 'G', -1, -1, 1.0E0, W, M, N, A, LDA, INFO )
         IF ( INFO.NE.0 ) THEN
            RETURN
         END IF
      END IF
*
*     Copy matrices A, B into the (M+P) x N matrix G
*
      CALL CLACPY( 'A', M, N, A, LDA, G( 1, 1 ), LDG )
      CALL CLACPY( 'A', P, N, B, LDB, G( M + 1, 1 ), LDG )
*
*     DEBUG
*
      CALL CLASET( 'A', M, N, CNAN, CNAN, A, LDA )
      CALL CLASET( 'A', P, N, CNAN, CNAN, B, LDB )
*
*     Compute the Frobenius norm of matrix G
*
      NORMG = NORMB * SQRT( 1.0E0 + ( ( W * NORMA ) / NORMB )**2 )
*
*     Get machine precision and set up threshold for determining
*     the effective numerical rank of the matrix G.
*
      ULP = SLAMCH( 'Precision' )
      UNFL = SLAMCH( 'Safe Minimum' )
      TOL = MAX( M + P, N ) * MAX( NORMG, UNFL ) * ULP
*
*     IWORK stores the column permutations computed by CGEQP3.
*     Columns J where IWORK( J ) is non-zero are permuted to the front
*     so we set the all entries to zero here.
*
      IWORK( 1:N ) = 0
*
*     Compute the QR factorization with column pivoting GΠ = Q1 R1
*
      CALL CGEQP3( M + P, N, G, LDG, IWORK, WORK( Z + 1 ),
     $             WORK( Z + LMAX + 1 ), LWORK - Z - LMAX, RWORK, INFO )
      IF( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     Determine the rank of G
*
      DO I = 1, MIN( M + P, N )
         IF( ABS( G( I, I ) ).LE.TOL ) THEN
            EXIT
         END IF
         L = L + 1
      END DO
*
*     Handle rank=0 case
*
      IF( L.EQ.0 ) THEN
         IF( WANTU1 ) THEN
            CALL CLASET( 'A', M, M, CZERO, CONE, U1, LDU1 )
         END IF
         IF( WANTU2 ) THEN
            CALL CLASET( 'A', P, P, CZERO, CONE, U2, LDU2 )
         END IF
*
         WORK( 1 ) = CMPLX( REAL ( LWKOPT ), 0.0E0 )
         RWORK( 1 ) = REAL( LRWKOPT )
         RETURN
      END IF
*
*     Copy R1( 1:L, : ) into A, B and set lower triangular part to zero
*
      IF( WANTX ) THEN
         IF( L.LE.M ) THEN
             CALL CLACPY( 'U', L, N, G, LDG, A, LDA )
             CALL CLASET( 'L', L - 1, N, CZERO, CZERO, A( 2, 1 ), LDA )
         ELSE
             CALL CLACPY( 'U', M, N, G, LDG, A, LDA )
             CALL CLACPY( 'U', L - M, N - M, G( M+1,M+1 ), LDG, B, LDB )
*
             CALL CLASET( 'L', M - 1, N, CZERO, CZERO, A( 2, 1 ), LDA )
             CALL CLASET( 'L', L-M-1, N, CZERO, CZERO, B( 2, 1 ), LDB )
         END IF
      END IF
*
*     Explicitly form Q1 so that we can compute the CS decomposition
*
      CALL CUNGQR( M + P, L, L, G, LDG, WORK( Z + 1 ),
     $             WORK( Z + L + 1 ), LWORK - Z - L, INFO )
      IF ( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     DEBUG
*
      ALPHA( 1:N ) = CNAN
      BETA( 1:N ) = CNAN
*
*     Compute the CS decomposition of Q1( :, 1:L )
*
      K = MIN( M, P, L, M + P - L )
      K1 = MAX( L - P, 0 )
      CALL CUNCSD2BY1( JOBU1, JOBU2, JOBX, M + P, M, L,
     $                 G( 1, 1 ), LDG, G( M + 1, 1 ), LDG,
     $                 ALPHA,
     $                 U1, LDU1, U2, LDU2, VT, LDVT,
     $                 WORK( Z + LDVT*N + 1 ), LWORK - Z - LDVT*N,
     $                 RWORK, LRWORK,
     $                 IWORK( N + 1 ), INFO )
      IF( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     DEBUG
*
      WORK( 1:LDG*N ) = CNAN
      RWORK( 1:2*N ) = NAN
*
*     Compute X = V^T R1( 1:L, : ) and adjust for matrix scaling
*
      IF( WANTX ) THEN
         LDX = L
         IF ( L.LE.M ) THEN
            CALL CGEMM( 'N', 'N', L, N, L,
     $                  CONE, VT, LDVT, A, LDA,
     $                  CZERO, WORK( 2 ), LDX )
         ELSE
            CALL CGEMM( 'N', 'N', L, N, M,
     $                  CONE, VT( 1, 1 ), LDVT, A, LDA,
     $                  CZERO, WORK( 2 ), LDX )
            CALL CGEMM( 'N', 'N', L, N - M, L - M,
     $                  CONE, VT( 1, M + 1 ), LDVT, B, LDB,
     $                  CONE, WORK( L*M + 2 ), LDX )
         END IF
*        Revert column permutation Π by permuting the columns of X
         CALL CLAPMT( .FALSE., L, N, WORK( 2 ), LDX, IWORK )
      END IF
*
*     Adjust generalized singular values for matrix scaling
*     Compute sine, cosine values
*     Prepare row scaling of X
*
      DO I = 1, K
         THETA = ALPHA( I )
*        Do not adjust singular value if THETA is greater
*        than pi/2 (infinite singular values won't change)
         IF( COS( THETA ).LE.0.0E0 ) THEN
            ALPHA( I ) = 0.0E0
            BETA( I ) = 1.0E0
            IF( WANTX ) THEN
               RWORK( I ) = 1.0E0
            END IF
         ELSE
*           iota comes in the greek alphabet after theta
            IOTA = ATAN( W * TAN( THETA ) )
*           ensure sine, cosine divisor is far away from zero
*           w is a power of two and will cause no trouble
            IF( SIN( IOTA ) .GE. COS( IOTA ) ) THEN
               ALPHA( I ) =  ( SIN( IOTA ) / TAN( THETA ) ) / W
               BETA( I ) = SIN( IOTA )
               IF( WANTX ) THEN
                  RWORK( I ) = SIN( THETA ) / SIN( IOTA )
               END IF
            ELSE
               ALPHA( I ) = COS( IOTA )
               BETA( I ) = SIN( IOTA )
               IF( WANTX ) THEN
                  RWORK( I ) = COS( THETA ) / COS( IOTA ) / W
               END IF
            END IF
         END IF
      END DO
*     Adjust rows of X for matrix scaling
      IF( WANTX ) THEN
         DO J = 0, N-1
            DO I = 1, K1
               WORK( LDX*J + I + 1 ) = WORK( LDX*J + I + 1 ) / W
            END DO
            DO I = 1, K
               WORK( LDX*J + I + K1 + 1 ) =
     $         WORK( LDX*J + I + K1 + 1 ) * RWORK( I )
            END DO
         END DO
      END IF
*
      WORK( 1 ) = CMPLX( REAL( LWKOPT ), 0.0E0 )
      RWORK( 1 ) = REAL( LRWKOPT )
      RETURN
*
*     End of CGGQRCS
*
      END
