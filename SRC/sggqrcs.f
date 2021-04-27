*> \brief <b> SGGQRCS computes the singular value decomposition (SVD) for OTHER matrices</b>
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download SGGQRCS + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sggqrcs.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sggqrcs.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sggqrcs.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE SGGQRCS( JOBU1, JOBU2, JOBX, M, N, P, L, SWAPPED,
*                           A, LDA, B, LDB,
*                           ALPHA, BETA,
*                           U1, LDU1, U2, LDU2,
*                           TOL,
*                           WORK, LWORK, IWORK, INFO )
*
*       .. Scalar Arguments ..
*       CHARACTER          JOBU1, JOB2, JOBX
*       INTEGER            INFO, LDA, LDB, LDU1, LDU2, M, N, P, L, LWORK
*       REAL               TOL
*       ..
*       .. Array Arguments ..
*       INTEGER            IWORK( * )
*       REAL               A( LDA, * ), B( LDB, * ),
*      $                   ALPHA( N ), BETA( N ),
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
*> SGGQRCS computes the generalized singular value decomposition (GSVD)
*> of an M-by-N real matrix A and P-by-N real matrix B:
*>
*>       A = U1 * D1 * X,           B = U2 * D2 * X
*>
*> where U1 and U2 are orthogonal matrices. SGGQRCS uses the QR
*> factorization with column pivoting and the 2-by-1 CS decomposition to
*> compute the GSVD.
*>
*> Let L be the effective numerical rank of the matrix (A**T,B**T)**T,
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
*>       A*inv(B) = U1*(D1*inv(D2))*U2**T.
*>
*> If (A**T,B**T)**T  has orthonormal columns, then the GSVD of A and B
*> is also equal to the CS decomposition of A and B. Furthermore, the
*> GSVD can be used to derive the solution of the eigenvalue problem:
*>
*>       A**T*A x = lambda * B**T*B x.
*>
*> In some literature, the GSVD of A and B is presented in the form
*>
*>       A = U1*D1*( 0 R )*Q**T,    B = U2*D2*( 0 R )*Q**T
*>
*> where U1, U2, and Q are orthogonal matrices. This latter GSVD form is
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
*>          (A**T, B**T)**T.
*> \endverbatim
*>
*> \param[out] SWAPPED
*> \verbatim
*>          L is LOGICAL
*>          On exit, SWAPPED is true if SGGQRCS swapped the input
*>          matrices A, B and computed the GSVD of (B, A); false
*>          otherwise.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is REAL array, dimension (LDA,N)
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
*>          B is REAL array, dimension (LDB,N)
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
*>          U1 is REAL array, dimension (LDU1,M)
*>          If JOBU1 = 'Y', U1 contains the M-by-M orthogonal matrix U1.
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
*>          U2 is REAL array, dimension (LDU2,P)
*>          If JOBU2 = 'Y', U2 contains the P-by-P orthogonal matrix U2.
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
*> \param[in,out] TOL
*> \verbatim
*>          TOL is REAL
*>          This user-provided tolerance is used for the rank determination
*>          of the matrix G = (A**T, W*B**T)**T, see the documentation
*>          of ABSTOL for details.
*>
*>          If TOL < 0, then the tolerance will be determined
*>          automatically and this should be the default choice for most
*>          users. Otherwise, the user must provide a value in the
*>          closed interval [0, 1].
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is REAL array, dimension (MAX(1,LWORK))
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
*>          > 0:  SBBCSD did not converge. For further details, see
*>                subroutine SORCSDBY1.
*> \endverbatim
*
*> \par Internal Parameters:
*  =========================
*>
*> \verbatim
*>  W       REAL
*>          W is a radix power chosen such that the Frobenius norm of A
*>          and W*B are within SQRT(RADIX) and 1/SQRT(RADIX) of each
*>          other.
*>
*>  ABSTOL  REAL
*>          Let G = (A**T, W*B**T)**T. ABSTOL is the threshold to determine
*>          the effective rank of G. Generally, it is set to
*>                   ABSTOL = TOL * MAX( M + P, N ) * norm(G),
*>          where norm(G) is the Frobenius norm of G.
*>          The size of ABSTOL may affect the size of backward error of the
*>          decomposition.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Christoph Conrads (https://christoph-conrads.name)
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
*>  SGGQRCS should be significantly faster than SGGSVD3 for large
*>  matrices because the matrices A and B are reduced to a pair of
*>  well-conditioned bidiagonal matrices instead of pairs of upper
*>  triangular matrices. On the downside, SGGQRCS requires a much larger
*>  workspace whose dimension must be queried at run-time. SGGQRCS also
*>  offers no guarantees which of the two possible diagonal matrices
*>  is used for the matrix factorization.
*>
*  =====================================================================
      RECURSIVE SUBROUTINE SGGQRCS( JOBU1, JOBU2, JOBX, M, N, P, L,
     $                              SWAPPED,
     $                              A, LDA, B, LDB,
     $                              ALPHA, BETA,
     $                              U1, LDU1, U2, LDU2,
     $                              TOL,
     $                              WORK, LWORK, IWORK, INFO )
*
*  -- LAPACK driver routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd. --
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      LOGICAL            SWAPPED
      CHARACTER          JOBU1, JOBU2, JOBX
      INTEGER            INFO, LDA, LDB, LDU1, LDU2, L, M, N, P, LWORK
      REAL               TOL
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      REAL               A( LDA, * ), B( LDB, * ),
     $                   ALPHA( N ), BETA( N ),
     $                   U1( LDU1, * ), U2( LDU2, * ),
     $                   WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      LOGICAL            PREPA, PREPB, WANTU1, WANTU2, WANTX, LQUERY
      INTEGER            I, J, K, K1, K2,KP, LMAX, IG, IG11, IG21, IG22,
     $                   K2P, K1P,
     $                   IVT, IVT12, LDG, LDX, LDVT, LWKMIN, LWKOPT,
     $                   ROWSA, ROWSB
      REAL               BASE, NORMA, NORMB, NORMG, ABSTOL, ULP, UNFL,
     $                   THETA, IOTA, W
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      REAL               SLAMCH, SLANGE
      EXTERNAL           LSAME, SLAMCH, SLANGE
*     ..
*     .. External Subroutines ..
      EXTERNAL           SGEMM, SGEQP3, SLACPY, SLAPMT, SLASCL,
     $                   SLASET, SORGQR, SORCSD2BY1, SORMQR, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          COS, MAX, MIN, SIN, SQRT
*     ..
*     .. Executable Statements ..
*
*     IWORK stores the column permutations computed by xGEQP3.
*     Columns J where IWORK( J ) is non-zero are permuted to the front
*     so IWORK must be set to zero before every call to xGEQP3.
*
*
*     Decode and test the input parameters
*
      WANTU1 = LSAME( JOBU1, 'Y' )
      WANTU2 = LSAME( JOBU2, 'Y' )
      WANTX = LSAME( JOBX, 'Y' )
      LQUERY = ( LWORK.EQ.-1 )
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
      ELSE IF( M.LT.0 ) THEN
         INFO = -4
      ELSE IF( N.LT.0 ) THEN
         INFO = -5
      ELSE IF( P.LT.0 ) THEN
         INFO = -6
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -10
      ELSE IF( LDB.LT.MAX( 1, P ) ) THEN
         INFO = -12
      ELSE IF( LDU1.LT.1 .OR. ( WANTU1 .AND. LDU1.LT.M ) ) THEN
         INFO = -16
      ELSE IF( LDU2.LT.1 .OR. ( WANTU2 .AND. LDU2.LT.P ) ) THEN
         INFO = -18
      ELSE IF( ISNAN(TOL) .OR. TOL.GT.1.0E0 ) THEN
         INFO = -19
      ELSE IF( LWORK.LT.1 .AND. .NOT.LQUERY ) THEN
         INFO = -21
      END IF
*
*     Make sure A is the matrix smaller in norm
*
      IF( INFO.EQ.0 ) THEN
         NORMA = SLANGE( 'F', M, N, A, LDA, WORK )
         NORMB = SLANGE( 'F', P, N, B, LDB, WORK )
*
         IF( NORMA.GT.SQRT( 2.0E0 ) * NORMB ) THEN
            CALL SGGQRCS( JOBU2, JOBU1, JOBX, P, N, M, L,
     $                    SWAPPED,
     $                    B, LDB, A, LDA,
     $                    BETA, ALPHA,
     $                    U2, LDU2, U1, LDU1,
     $                    TOL,
     $                    WORK, LWORK, IWORK, INFO )
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
      SWAPPED = .FALSE.
*     TODO
      PREPA = m.GT.n .OR. ( m.GT.0 )
      PREPB = p.GT.n .OR. ( p.GT.0 )
*      PREPA = .FALSE.
      PREPB = .FALSE.
      L = 0
*     The leading dimension must never be zero
      ROWSA = M
      IF( PREPA ) THEN
         ROWSA = MIN( M, N )
      ENDIF
      ROWSB = P
      IF( PREPB ) THEN
         ROWSB = MIN( P, N )
      ENDIF
      LDG = MAX( M + P, 1 )
      LDVT = N
      LMAX = MIN( ROWSA + ROWSB, N )
      IG = 1
*     IGxx are blocks of the full rank, upper triangular factor of QR
*     factorization of G to be copied into A, B
*     G11 is upper left M X M block
      IG11 = IG
      IG21 = IG11 + M
*     Rank of A is ignored because only its dimension matters
      IG22 = LDG * M + M + 1
      IVT = LDG * N + 2
      IVT12 = IVT + LDVT * M
      THETA = -1
      IOTA = -1
      W = -1
      ULP = SLAMCH( 'Precision' )
      UNFL = SLAMCH( 'Safe Minimum' )
      IF( TOL.LT.0.0E0 .AND. .NOT.LQUERY ) THEN
         TOL = ULP
      ENDIF
*
*     Compute workspace
*
      IF( INFO.EQ.0 ) THEN
         LWKMIN = 0
         LWKOPT = 0
*
         CALL SGEQP3( ROWSA + ROWSB, N, WORK( IG ), LDG, IWORK, BETA,
     $                WORK, -1, INFO )
         LWKMIN = MAX( LWKMIN, 3 * N + 1 )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
*
         CALL SORGQR( ROWSA + ROWSB, LMAX, LMAX, WORK( IG ), LDG, BETA,
     $                WORK, -1, INFO )
         LWKMIN = MAX( LWKMIN, LMAX )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
*
         CALL SORCSD2BY1( JOBU1, JOBU2, JOBX, ROWSA + ROWSB, ROWSA,LMAX,
     $                    WORK( IG ), LDG, WORK( IG ), LDG,
     $                    BETA,
     $                    U1, LDU1, U2, LDU2, WORK( IVT ), LDVT,
     $                    WORK, -1, IWORK, INFO )
         LWKMIN = MAX( LWKMIN, INT( WORK( 1 ) ) )
         LWKOPT = MAX( LWKOPT, INT( WORK( 1 ) ) )
*        The matrix (A, B) must be stored sequentially for SORGQR
         LWKMIN = LWKMIN + IVT
         LWKOPT = LWKOPT + IVT
*        2-by-1 CSD matrix V1 must be stored
         IF( WANTX ) THEN
            LWKMIN = LWKMIN + LDVT*N
            LWKOPT = LWKOPT + LDVT*N
         END IF
*        Check workspace size
         IF( LWORK.LT.LWKMIN .AND. .NOT.LQUERY ) THEN
            INFO = -20
         END IF
*
         WORK( 1 ) = REAL( LWKOPT )
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'SGGQRCS', -INFO )
         RETURN
      END IF
      IF( LQUERY ) THEN
         RETURN
      ENDIF
*     Finish initialization
      IF( .NOT.WANTX ) THEN
         LDVT = 0
      END IF
*
*     Select scaling factor W such that norm(A) \approx norm(B)
*
      IF( NORMA.EQ.0.0E0 ) THEN
         W = 1.0E0
      ELSE
         BASE = SLAMCH( 'B' )
         W = BASE ** INT( LOG( NORMB / NORMA ) / LOG( BASE ) )
      END IF
*
*     Attempt to remove unnecessary matrix rows
*     Copy matrices A, B or their full-rank factors, respectively, into
*     the LDG x N matrix G
*
      IF( PREPA ) THEN
         IWORK( 1:N ) = 0
         CALL SGEQP3( M, N, A, LDA, IWORK, ALPHA, WORK, LWORK, INFO )
         IF( INFO.NE.0 ) THEN
            RETURN
         END IF
*        Determine rank of A
         ROWSA = 0
         ABSTOL = TOL * MAX( M, N ) * MAX( NORMA, UNFL )
         DO I = 1, MIN( M, N )
            IF( ABS( A( I, I ) ).LE.ABSTOL ) THEN
               EXIT
            END IF
            ROWSA = ROWSA + 1
         END DO
         IG21 = IG11 + ROWSA
*        Scale, copy full rank part into G
         CALL SLASCL( 'U', -1, -1, 1.0E0, W, ROWSA, N, A, LDA, INFO )
         IF ( INFO.NE.0 ) THEN
            RETURN
         END IF
         CALL SLASET( 'L', ROWSA, N, 0.0E0, 0.0E0, WORK( IG ), LDG )
         CALL SLACPY( 'U', ROWSA, N, A, LDA, WORK( IG ), LDG )
         CALL SLAPMT( .FALSE., ROWSA, N, WORK( IG ), LDG, IWORK )
*        Initialize U1 although xORCSDB2BY1 will partially overwrite this
         IF( WANTU1 ) THEN
             CALL SLASET( 'A', M, M, 0.0E0, 1.0E0, U1, LDU1 )
         ENDIF
      ELSE
         CALL SLASCL( 'G', -1, -1, 1.0E0, W, M, N, A, LDA, INFO )
         IF ( INFO.NE.0 ) THEN
            RETURN
         END IF
         CALL SLACPY( 'A', M, N, A, LDA, WORK( IG11 ), LDG )
      END IF
*
      IF( PREPB ) THEN
         IWORK( 1:N ) = 0
         CALL SGEQP3( P, N, B, LDB, IWORK, BETA,
     $                WORK( IVT ), LWORK - IVT + 1, INFO )
         IF( INFO.NE.0 ) THEN
            RETURN
         END IF
*        Determine rank
         ROWSB = 0
         ABSTOL = TOL * MAX( P, N ) * MAX( NORMB, UNFL )
         DO I = 1, MIN( P, N )
            IF( ABS( B( I, I ) ).LE.ABSTOL ) THEN
               EXIT
            END IF
            ROWSB = ROWSB + 1
         END DO
*        Copy full rank part into G
         CALL SLASET( 'L', ROWSB, N, 0.0E0, 0.0E0, WORK( IG21 ), LDG )
         CALL SLACPY( 'U', ROWSB, N, B, LDB, WORK( IG21 ), LDG )
         CALL SLAPMT( .FALSE., ROWSB, N, WORK( IG21 ), LDG, IWORK )
*        Initialize U2 although xORCSDB2BY1 will partially overwrite this
*        Copy scalar factors because BETA is re-used later
*        Copy into last column of B because it is never used
         IF( WANTU2 ) THEN
            CALL SLACPY( 'A', ROWSB, 1, BETA, 1, B( 1, N ), LDB )
            CALL SLASET( 'A', P, P, 0.0E0, 1.0E0, U2, LDU2 )
         ENDIF
      ELSE
         CALL SLACPY( 'A', P, N, B, LDB, WORK( IG21 ), LDG )
      END IF
*
*     Compute the Frobenius norm of matrix G
*
      NORMG = NORMB * SQRT( 1.0E0 + ( ( W * NORMA ) / NORMB )**2 )
*
*     Compute the QR factorization with column pivoting GΠ = Q1 R1
*
      IWORK( 1:N ) = 0
      CALL SGEQP3( ROWSA + ROWSB, N, WORK( IG ), LDG, IWORK, BETA,
     $             WORK( IVT ), LWORK - IVT + 1, INFO )
      IF( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     Determine the rank of G
*
      ABSTOL = TOL * MAX( ROWSA + ROWSB, N ) * MAX( NORMG, UNFL )
      DO I = 1, MIN( ROWSA + ROWSB, N )
         IF( ABS( WORK( (I-1) * LDG + I ) ).LE.ABSTOL ) THEN
            EXIT
         END IF
         L = L + 1
      END DO
*
*     Handle rank=0 case
*
      IF( L.EQ.0 ) THEN
         IF( WANTU1 ) THEN
            CALL SLASET( 'A', M, M, 0.0E0, 1.0E0, U1, LDU1 )
         END IF
         IF( WANTU2 ) THEN
            CALL SLASET( 'A', P, P, 0.0E0, 1.0E0, U2, LDU2 )
         END IF
*
         WORK( 1 ) = REAL( LWKOPT )
         RETURN
      END IF
*
*     Copy R1( 1:L, : ) into A, B and set lower triangular part to zero
*
      IF( WANTX ) THEN
         IF( L.LE.M ) THEN
            CALL SLACPY( 'U', L, N, WORK( IG ), LDG, A, LDA )
         ELSE
            CALL SLACPY( 'U', M, N, WORK( IG ), LDG, A, LDA )
            CALL SLACPY( 'U', L - M, N - M, WORK( IG22 ), LDG, B, LDB )
         END IF
      END IF
*
*     Explicitly form Q1 so that we can compute the CS decomposition
*
      CALL SORGQR( ROWSA + ROWSB, L, L, WORK( IG ), LDG, BETA,
     $             WORK( IVT ), LWORK - IVT + 1, INFO )
      IF ( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     Compute the CS decomposition of Q1( :, 1:L )
*
      CALL SORCSD2BY1( JOBU1, JOBU2, JOBX, ROWSA + ROWSB, ROWSA, L,
     $                 WORK( IG11 ), LDG, WORK( IG21 ), LDG,
     $                 BETA,
     $                 U1, LDU1, U2, LDU2, WORK( IVT ), LDVT,
     $                 WORK( IVT + LDVT*N ), LWORK - IVT - LDVT*N + 1,
     $                 IWORK( N + 1 ), INFO )
      IF( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     Apply orthogonal factors of QR decomposition of A, B to U1, U2
*
      IF( PREPA.AND.WANTU1 ) THEN
         CALL SORMQR( 'L', 'N', M, M, ROWSA, A, LDA, ALPHA, U1, LDU1,
     $           WORK, IVT, INFO )
         IF( INFO.NE.0 ) THEN
            RETURN
         ENDIF
      ENDIF
*
      IF( PREPB.AND.WANTU2 ) THEN
         CALL SORMQR( 'L', 'N', P, P, ROWSB, B, LDB, B( 1, N ),
     $           U2, LDU2, WORK, IVT, INFO )
         IF( INFO.NE.0 ) THEN
            RETURN
         ENDIF
      ENDIF
*
*     Compute X = V^T R1( 1:L, : ) and adjust for matrix scaling
*
      IF( WANTX ) THEN
         LDX = L
         IF ( L.LE.M ) THEN
            CALL SLASET( 'L', L - 1, N, 0.0E0, 0.0E0, A( 2, 1 ), LDA )
            CALL SGEMM( 'N', 'N', L, N, L,
     $                  1.0E0, WORK( IVT ), LDVT, A, LDA,
     $                  0.0E0, WORK( 2 ), LDX )
         ELSE
            CALL SLASET( 'L', M - 1, N, 0.0E0, 0.0E0, A( 2, 1 ), LDA )
            CALL SGEMM( 'N', 'N', L, N, M,
     $                  1.0E0, WORK( IVT ), LDVT, A, LDA,
     $                  0.0E0, WORK( 2 ), LDX )
*
            CALL SLASET( 'L', L-M-1, N, 0.0E0, 0.0E0, B( 2, 1 ), LDB )
            CALL SGEMM( 'N', 'N', L, N - M, L - M,
     $                  1.0E0, WORK( IVT12 ), LDVT, B, LDB,
     $                  1.0E0, WORK( L*M + 2 ), LDX )
         END IF
*        Revert column permutation Π by permuting the columns of X
         CALL SLAPMT( .FALSE., L, N, WORK( 2 ), LDX, IWORK )
      END IF
*
*     Fix column order of U2
*     Because of the QR decomposition in the pre-processing, the first
*     rank(B) columns of U2 are a basis of range(B) but for matrix B,
*     the CS values are in ascending order. If B is singular, then the
*     first P - rank(B) columns should be a basis for the complement of
*     range(B). For this reason, the columns must be re-ordered.
*
      IF( PREPB.AND.WANTU2 ) THEN
         DO I = 1, ROWSB
            IWORK( I ) = P - ROWSB + I
         ENDDO
         DO I = ROWSB + 1, P - ROWSB
            IWORK( I ) = I
         ENDDO
         DO I = P - ROWSB + 1, P
            IWORK( I ) = I - (P - ROWSB)
         ENDDO
         CALL SLAPMT( .FALSE., P, P, U2, LDU2, IWORK )
      ENDIF
*
*     Adjust generalized singular values for matrix scaling
*     Compute sine, cosine values
*     Prepare row scaling of X
*
      K = MIN( M, P, L, M + P - L )
      K1 = MAX( L - P, 0 )
      K2 = MAX( L - M, 0 )
      KP = MIN( ROWSA, ROWSB, L, ROWSA + ROWSB - L )
      K1P = MAX( L - ROWSB, 0 )
      K2P = MAX( L - ROWSA, 0 )
*      PRINT*, "BETA", K, K1, K2
*      PRINT*, "BETA", KP, K1P, K2P
      IF( PREPA ) THEN
         DO I = KP + 1, K
            BETA( I ) = ACOS( 0.0E0 )
         ENDDO
      ENDIF
*         DO I = KP, 1, -1
*            BETA( K1P + I ) = BETA( I )
*         ENDDO
**
*         BETA( 1:K1P ) = 0.0E0
*         BETA( K1P+KP+1:K1P+KP+K2P ) = ACOS( 0.0E0 )
*      ENDIF
*
      DO I = 1, K
         THETA = BETA( I )
*        Do not adjust singular value if THETA is greater
*        than pi/2 (infinite singular values won't change)
         IF( COS( THETA ).LE.0.0E0 ) THEN
            ALPHA( I ) = 0.0E0
            BETA( I ) = 1.0E0
            IF( WANTX ) THEN
               WORK( IVT + I ) = 1.0E0
            END IF
         ELSE
*           iota comes in the greek alphabet after theta
            IOTA = ATAN( W * TAN( THETA ) )
*           ensure sine, cosine divisor is far away from zero
*           w is a power of two and will cause no trouble
            IF( SIN( IOTA ) .GE. COS( IOTA ) ) THEN
               ALPHA( I ) = ( SIN( IOTA ) / TAN( THETA ) ) / W
               BETA( I ) = SIN( IOTA )
               IF( WANTX ) THEN
                  WORK( IVT + I ) = SIN( THETA ) / SIN( IOTA )
               END IF
            ELSE
               ALPHA( I ) = COS( IOTA )
               BETA( I ) = SIN( IOTA )
               IF( WANTX ) THEN
                  WORK( IVT + I ) = COS( THETA ) / COS( IOTA ) / W
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
     $         WORK( LDX*J + I + K1 + 1 ) * WORK( IVT + I )
            END DO
         END DO
      END IF
*
      WORK( 1 ) = REAL( LWKOPT )
      RETURN
*
*     End of SGGQRCS
*
      END
