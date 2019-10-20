/*
 * Copyright (c) 2016, 2019, Christoph Conrads (https://christoph-conrads.name)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of the copyright holders nor the
 *    names of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef LAPACK_TESTS_LAPACK_HPP
#define LAPACK_TESTS_LAPACK_HPP

#include <config.h>

#if not defined HAS_GGSVD3
# include <algorithm>
#endif

#ifdef USE_MKL
# include <mkl.h>
#else
# include <lapacke.h>
#endif

#include <cctype>
#include <cassert>


extern "C"
{
	void sggqrcs_(
		char* jobu1, char* jobu2, char* jobqt,
		lapack_int* m, lapack_int* n, lapack_int* p,
		float* w, lapack_int* l,
		float* A, lapack_int* lda, float* B, lapack_int* ldb,
		float* theta,
		float* U1, lapack_int* ldu1, float* U2, lapack_int* ldu2,
		float* Qt, lapack_int* ldqt,
		float* work, lapack_int* lwork, lapack_int* iwork,
		lapack_int* info);

	void dggqrcs_(
		char* jobu1, char* jobu2, char* jobqt,
		lapack_int* m, lapack_int* n, lapack_int* p,
		double* w, lapack_int* l,
		double* A, lapack_int* lda, double* B, lapack_int* ldb,
		double* theta,
		double* U1, lapack_int* ldu1, double* U2, lapack_int* ldu2,
		double* Qt, lapack_int* ldqt,
		double* work, lapack_int* lwork, lapack_int* iwork,
		lapack_int* info);
}



namespace lapack
{
	namespace impl
	{
		template<typename T>
		T* return_not_null(T* p)
		{
#ifdef USE_MKL
			return p ? p : (T*)-1;
#else
			return p;
#endif
		}
	}

typedef lapack_int integer_t;



inline integer_t geqp3(
	integer_t m, integer_t n, float* A, integer_t lda,
	integer_t* pivot, float* tau, float* work, integer_t lwork)
{
	A = impl::return_not_null(A);
	pivot = impl::return_not_null(pivot);
	tau = impl::return_not_null(tau);

	integer_t info = -1;
	sgeqp3_(&m, &n, A, &lda, pivot, tau, work, &lwork, &info);
	return info;
}

inline integer_t geqp3(
	integer_t m, integer_t n, double* A, integer_t lda,
	integer_t* pivot, double* tau, double* work, integer_t lwork)
{
	A = impl::return_not_null(A);
	pivot = impl::return_not_null(pivot);
	tau = impl::return_not_null(tau);

	integer_t info = -1;
	dgeqp3_(&m, &n, A, &lda, pivot, tau, work, &lwork, &info);
	return info;
}



inline integer_t geqrf(
	integer_t m, integer_t n, float* A, integer_t lda, float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sgeqrf_(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t geqrf(
	integer_t m, integer_t n, double* A, integer_t lda, double* p_tau,
	double* p_work, integer_t lwork)
{
	integer_t info = -1;
	dgeqrf_(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t gerqf(
	integer_t m, integer_t n, float* A, integer_t lda, float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sgerqf_(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t gerqf(
	integer_t m, integer_t n, double* A, integer_t lda, double* p_tau,
	double* p_work, integer_t lwork)
{
	integer_t info = -1;
	dgerqf_(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t gesvd(
	char jobu, char jobvt, integer_t m, integer_t n, float* A, integer_t lda,
	float* sigma, float* U, integer_t ldu, float* Vt, integer_t ldvt,
	float* work, integer_t lwork)
{
	A = impl::return_not_null(A);
	sigma = impl::return_not_null(sigma);
	U = impl::return_not_null(U);

	integer_t info = -1;
	sgesvd_(
		&jobu, &jobvt, &m, &n, A, &lda, sigma, U, &ldu, Vt, &ldvt,
		work, &lwork, &info);
	return info;
}

inline integer_t gesvd(
	char jobu, char jobvt, integer_t m, integer_t n, double* A, integer_t lda,
	double* sigma, double* U, integer_t ldu, double* Vt, integer_t ldvt,
	double* work, integer_t lwork)
{
	A = impl::return_not_null(A);
	sigma = impl::return_not_null(sigma);
	U = impl::return_not_null(U);

	integer_t info = -1;
	dgesvd_(
		&jobu, &jobvt, &m, &n, A, &lda, sigma, U, &ldu, Vt, &ldvt,
		work, &lwork, &info);
	return info;
}



inline integer_t ggqrcs(
	char jobu1, char jobu2, char jobqt,
	integer_t m, integer_t n, integer_t p, float* w, integer_t* l,
	float* A, integer_t lda, float* B, integer_t ldb,
	float* theta,
	float* U1, integer_t ldu1, float* U2, integer_t ldu2,
	float* Qt, integer_t ldqt,
	float* work, integer_t lwork, integer_t* iwork)
{
	assert( w );
	assert( l );
	assert( work );

	integer_t info = -1;
	sggqrcs_(
		&jobu1, &jobu2, &jobqt,
		&m, &n, &p, w, l,
		A, &lda, B, &ldb,
		theta,
		U1, &ldu1, U2, &ldu2, Qt, &ldqt,
		work, &lwork, iwork, &info);
	return info;
}

inline integer_t ggqrcs(
	char jobu1, char jobu2, char jobqt,
	integer_t m, integer_t n, integer_t p, double* w, integer_t* l,
	double* A, integer_t lda, double* B, integer_t ldb,
	double* theta,
	double* U1, integer_t ldu1, double* U2, integer_t ldu2,
	double* Qt, integer_t ldqt,
	double* work, integer_t lwork, integer_t* iwork)
{
	assert( w );
	assert( l );
	assert( work );

	integer_t info = -1;
	dggqrcs_(
		&jobu1, &jobu2, &jobqt,
		&m, &n, &p, w, l,
		A, &lda, B, &ldb,
		theta,
		U1, &ldu1, U2, &ldu2, Qt, &ldqt,
		work, &lwork, iwork, &info);
	return info;
}



#ifdef HAS_GGSVD3
inline integer_t ggsvd3(
	char jobu, char jobv, char jobq, integer_t m, integer_t n, integer_t p,
	integer_t* p_k, integer_t* p_l,
	float* A, integer_t lda, float* B, integer_t ldb,
	float* alpha, float* beta,
	float* U, integer_t ldu, float* V, integer_t ldv, float* Q, integer_t ldq,
	float* work, integer_t lwork, integer_t* iwork)
{
	integer_t info = -1;
	sggsvd3_(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, &lwork, iwork, &info);
	return info;
}

inline integer_t ggsvd3(
	char jobu, char jobv, char jobq, integer_t m, integer_t n, integer_t p,
	integer_t* p_k, integer_t* p_l,
	double* A, integer_t lda, double* B, integer_t ldb,
	double* alpha, double* beta,
	double* U, integer_t ldu, double* V, integer_t ldv, double* Q, integer_t ldq,
	double* work, integer_t lwork, integer_t* iwork)
{
	integer_t info = -1;
	dggsvd3_(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, &lwork, iwork, &info);
	return info;
}
#endif



inline integer_t heevd(
	char jobz, char uplo, integer_t n, float* A, integer_t lda, float* lambda,
	float* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	A = impl::return_not_null(A);
	lambda = impl::return_not_null(lambda);

	integer_t info = -1;
	ssyevd_(
		&jobz, &uplo, &n, A, &lda, lambda, work, &lwork, iwork, &liwork, &info);
	return info;
}

inline integer_t heevd(
	char jobz, char uplo, integer_t n, double* A, integer_t lda, double* lambda,
	double* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	A = impl::return_not_null(A);
	lambda = impl::return_not_null(lambda);

	integer_t info = -1;
	dsyevd_(
		&jobz, &uplo, &n, A, &lda, lambda, work, &lwork, iwork, &liwork, &info);
	return info;
}



inline integer_t hegv(
	integer_t itype, char jobz, char uplo, integer_t n,
	float* A, lapack_int lda, float* B, lapack_int ldb, float* lambda,
	float* work, integer_t lwork)
{
	integer_t info = -1;
	ssygv_(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, &info);
	return info;
}

inline integer_t hegv(
	integer_t itype, char jobz, char uplo, integer_t n,
	double* A, lapack_int lda, double* B, lapack_int ldb, double* lambda,
	double* work, integer_t lwork)
{
	integer_t info = -1;
	dsygv_(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, &info);
	return info;
}



inline integer_t hegvd(
	integer_t itype, char jobz, char uplo, integer_t n,
	float* A, lapack_int lda, float* B, lapack_int ldb, float* lambda,
	float* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	ssygvd_(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, iwork, &liwork, &info);
	return info;
}

inline integer_t hegvd(
	integer_t itype, char jobz, char uplo, integer_t n,
	double* A, lapack_int lda, double* B, lapack_int ldb, double* lambda,
	double* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	dsygvd_(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, iwork, &liwork, &info);
	return info;
}



inline void lacpy(
	char uplo, integer_t m, integer_t n,
	const float* A, integer_t lda, float* B, integer_t ldb )
{
	slacpy_( &uplo, &m, &n, A, &lda, B, &ldb );
}

inline void lacpy(
	char uplo, integer_t m, integer_t n,
	const double* A, integer_t lda, double* B, integer_t ldb )
{
	dlacpy_( &uplo, &m, &n, A, &lda, B, &ldb );
}



inline float lange(
	char norm, integer_t m, integer_t n,
	const float* A, integer_t lda, float* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );
	return slange_(&norm, &m, &n, A, &lda, work);
}

inline double lange(
	char norm, integer_t m, integer_t n,
	const double* A, integer_t lda, double* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );
	return dlange_(&norm, &m, &n, A, &lda, work);
}



inline float lanhe(
	char norm, char uplo, integer_t n,
	const float* A, integer_t lda, float* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );

#ifdef USE_MKL
	// work around a bug with Intel MKL 11.3 Update 2
	const int num_threads = MKL_Set_Num_Threads_Local(1);
	const float ret = slansy(&norm, &uplo, &n, A, &lda, work);
	MKL_Set_Num_Threads_Local(num_threads);
	return ret;
#else
	return slansy_(&norm, &uplo, &n, A, &lda, work);
#endif
}

inline double lanhe(
	char norm, char uplo, integer_t n,
	const double* A, integer_t lda, double* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );

#ifdef USE_MKL
	// work around a bug with Intel MKL 11.3 Update 2
	const int num_threads = MKL_Set_Num_Threads_Local(1);
	const double ret = dlansy(&norm, &uplo, &n, A, &lda, work);
	MKL_Set_Num_Threads_Local(num_threads);
	return ret;
#else
	return dlansy_(&norm, &uplo, &n, A, &lda, work);
#endif
}



inline void lapmt(
	bool forward, integer_t m, integer_t n,
	float* A, integer_t lda, integer_t* piv)
{
	integer_t iforward = forward;
	slapmt_( &iforward, &m, &n, A, &lda, piv );
}

inline void lapmt(
	bool forward, integer_t m, integer_t n,
	double* A, integer_t lda, integer_t* piv)
{
	integer_t iforward = forward;
	dlapmt_( &iforward, &m, &n, A, &lda, piv );
}



inline integer_t lascl(
	char type, integer_t kl, integer_t ku, float cfrom, float cto,
	integer_t m, integer_t n, float* A, integer_t lda)
{
	integer_t info = -1;
	slascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, A, &lda, &info);
	return info;
}

inline integer_t lascl(
	char type, integer_t kl, integer_t ku, double cfrom, double cto,
	integer_t m, integer_t n, double* A, integer_t lda)
{
	integer_t info = -1;
	dlascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, A, &lda, &info);
	return info;
}



inline void laset(
	char uplo, integer_t m, integer_t n,
	float alpha, float beta, float* A, integer_t lda )
{
	slaset_( &uplo, &m, &n, &alpha, &beta, A, &lda );
}

inline void laset(
	char uplo, integer_t m, integer_t n,
	double alpha, double beta, double* A, integer_t lda )
{
	dlaset_( &uplo, &m, &n, &alpha, &beta, A, &lda );
}



inline integer_t pstrf(
	char uplo, integer_t n, float* A, integer_t lda,
	integer_t* p_piv, integer_t* p_rank, float tol,
	float* p_work)
{
	integer_t info = -1;
	spstrf_( &uplo, &n, A, &lda, p_piv, p_rank, &tol, p_work, &info );
	return info;
}

inline integer_t pstrf(
	char uplo, integer_t n, double* A, integer_t lda,
	integer_t* p_piv, integer_t* p_rank, double tol,
	double* p_work)
{
	integer_t info = -1;
	dpstrf_( &uplo, &n, A, &lda, p_piv, p_rank, &tol, p_work, &info );
	return info;
}



inline integer_t trcon(
	char norm, char uplo, char diag, integer_t n, const float* A,
	integer_t lda, float* p_rcond, float* p_work, integer_t* p_iwork)
{
	integer_t info = -1;
	strcon_(
		&norm, &uplo, &diag, &n, A, &lda, p_rcond, p_work, p_iwork, &info );
	return info;
}

inline integer_t trcon(
	char norm, char uplo, char diag, integer_t n, const double* A,
	integer_t lda, double* p_rcond, double* p_work, integer_t* p_iwork)
{
	integer_t info = -1;
	dtrcon_(
		&norm, &uplo, &diag, &n, A, &lda, p_rcond, p_work, p_iwork, &info );
	return info;
}



inline integer_t uncsd2by1(
	char jobu1, char jobu2, char jobv1t,
	integer_t m, integer_t p, integer_t q,
	float* X11, integer_t ldx11,
	float* X21, integer_t ldx21,
	float* theta,
	float* U1, integer_t ldu1,
	float* U2, integer_t ldu2,
	float* V1t, integer_t ldv1t,
	float* work, integer_t lwork, integer_t* iwork)
{
	X11 = impl::return_not_null(X11);
	X21 = impl::return_not_null(X21);
	theta = impl::return_not_null(theta);
	U1 = impl::return_not_null(U1);
	U2 = impl::return_not_null(U2);
	V1t = impl::return_not_null(V1t);
	iwork = impl::return_not_null(iwork);

	integer_t info = -1;
	sorcsd2by1_(
		&jobu1, &jobu2, &jobv1t,
		&m, &p, &q,
		X11, &ldx11, X21, &ldx21, theta,
		U1, &ldu1, U2, &ldu2, V1t, &ldv1t,
		work, &lwork, iwork, &info);
	return info;
}

inline integer_t uncsd2by1(
	char jobu1, char jobu2, char jobv1t,
	integer_t m, integer_t p, integer_t q,
	double* X11, integer_t ldx11,
	double* X21, integer_t ldx21,
	double* theta,
	double* U1, integer_t ldu1,
	double* U2, integer_t ldu2,
	double* V1t, integer_t ldv1t,
	double* work, integer_t lwork, integer_t* iwork)
{
	X11 = impl::return_not_null(X11);
	X21 = impl::return_not_null(X21);
	theta = impl::return_not_null(theta);
	U1 = impl::return_not_null(U1);
	U2 = impl::return_not_null(U2);
	V1t = impl::return_not_null(V1t);
	iwork = impl::return_not_null(iwork);

	integer_t info = -1;
	dorcsd2by1_(
		&jobu1, &jobu2, &jobv1t,
		&m, &p, &q,
		X11, &ldx11, X21, &ldx21, theta,
		U1, &ldu1, U2, &ldu2, V1t, &ldv1t,
		work, &lwork, iwork, &info);
	return info;
}



inline integer_t ungqr(
	integer_t m, integer_t n, integer_t k,
	float* A, integer_t lda, const float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sorgqr_(&m, &n, &k, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t ungqr(
	integer_t m, integer_t n, integer_t k,
	double* A, integer_t lda, const double* p_tau,
	double* p_work, integer_t lwork)
{
	A = impl::return_not_null(A);
	p_tau = impl::return_not_null(p_tau);

	integer_t info = -1;
	dorgqr_(&m, &n, &k, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t ungrq(
	integer_t m, integer_t n, integer_t k,
	float* A, integer_t lda, float* tau,
	float* work, integer_t lwork)
{
	integer_t info = -1;
	sorgrq_(&m, &n, &k, A, &lda, tau, work, &lwork, &info);
	return info;
}

inline integer_t ungrq(
	integer_t m, integer_t n, integer_t k,
	double* A, integer_t lda, double* tau,
	double* work, integer_t lwork)
{
	integer_t info = -1;
	dorgrq_(&m, &n, &k, A, &lda, tau, work, &lwork, &info);
	return info;
}

}

#endif
