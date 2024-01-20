/*
 * Copyright (c) 2016, 2019, 2020, 2023 Christoph Conrads
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
#ifndef LAPACK_TESTS_TOOLS_HPP
#define LAPACK_TESTS_TOOLS_HPP

#include "lapack.hpp"

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>


namespace lapack {
namespace tools
{

namespace ublas = boost::numeric::ublas;

using Integer = lapack::integer_t;



template<typename Number>
bool nan_p(Number x)
{
	return std::isnan(std::real(x)) or std::isnan(std::imag(x));
}


template<typename Number>
bool inf_p(Number x)
{
	return std::isinf(std::real(x)) or std::isinf(std::imag(x));
}


template<typename Number>
bool finite_p(Number x)
{
	return std::isfinite(std::real(x)) and std::isfinite(std::imag(x));
}



/**
 * Given a type `T`, `real_from<T>::type` is `T` if `T` is a built-in type,
 * otherwise is returns the numeric type used to implement `T`.
 *
 * Examples:
 * * `real_from<float>::type == float`
 * * `real_from<std::complex<float>>::type == float`
 */
template<typename T> struct real_from {};

template<> struct real_from<float> { using type = float; };
template<> struct real_from<double> { using type = double; };
template<typename Real>
struct real_from<std::complex<Real>> { using type = Real; };

static_assert(std::is_same<typename real_from<float>::type, float>::value, "");
static_assert(std::is_same<typename real_from<double>::type,double>::value, "");
static_assert(
	std::is_same<typename real_from<std::complex<float>>::type,float>::value, ""
);



template<typename Real>
struct not_a_number
{
	static constexpr Real value = std::numeric_limits<Real>::quiet_NaN();
};

template<typename Real>
constexpr Real not_a_number<Real>::value;


template<typename Real>
struct not_a_number<std::complex<Real>>
{
	using type = std::complex<Real>;

	static constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();
	static constexpr type value = type{nan, nan};
};

template<typename Real>
constexpr std::complex<Real> not_a_number<std::complex<Real>>::value;



/**
 * This function maps boolean values to LAPACK flags.
 */
char bool2lapackjob(bool p)
{
	return p ? 'Y' : 'N';
}



template<typename Real>
struct UniformDistribution
{
	UniformDistribution() : dist_(-1, +1) {}

	template<typename Engine>
	Real operator() (Engine& gen)
	{
		return dist_(gen);
	}

	std::uniform_real_distribution<Real> dist_;
};

template<typename Real>
struct UniformDistribution<std::complex<Real>>
{
	using result_type = std::complex<Real>;

	UniformDistribution() : dist_(-1, +1) {}

	template<typename Engine>
	result_type operator() (Engine& gen)
	{
		constexpr auto pi = Real{M_PI};
		auto radius = std::abs(dist_(gen));
		auto angle = 2 * pi * dist_(gen);

		return std::polar(radius, angle);
	}

	std::uniform_real_distribution<Real> dist_;
};


template<class Matrix>
void print_matrix(const char* name, const Matrix& a) {
	std::printf("%s\n", name);

	for(auto i = std::size_t{0}; i < a.size1(); ++i) {
		for(auto j = std::size_t{0}; j < a.size2(); ++j) {
			if(j + 1 < a.size2()) {
				std::printf("%+8.2e ", a(i, j));
			}
			else {
				std::printf("%+8.2e\n", a(i, j));
			}
		}
	}
}


template<class Matrix>
void print_matrix_to_file(const char* filename, const Matrix& a) {
	using Number = typename Matrix::value_type;
	constexpr const char* fmt = std::is_same<Number, float>::value
		? "%+16.9e%s"
		: "%+24.17e%s";

	auto f = std::fopen(filename, "w");
	if(!f) {
		std::perror("fopen");
		std::exit(EXIT_FAILURE);
	}

	for(auto i = std::size_t{0}; i < a.size1(); ++i) {
		for(auto j = std::size_t{0}; j < a.size2(); ++j) {
			const char* sep = (j + 1 < a.size2()) ? " " : "\n";
			std::fprintf(f, fmt, a(i, j), sep);
		}
	}

	if(fclose(f) < 0) {
		std::perror("fclose");
		std::exit(EXIT_FAILURE);
	}
}

template<class Real, class Storage>
void print_matrix_to_file(
	const char* filename, const ublas::matrix<std::complex<Real>, Storage>& a
) {
	constexpr const char* fmt = std::is_same<Real, float>::value
		? "(%+16.9e%+16.9ej)%s"
		: "(%+24.17e %+24.17ej)%s";

	auto f = std::fopen(filename, "w");
	if(!f) {
		std::perror("fopen");
		std::exit(EXIT_FAILURE);
	}

	for(auto i = std::size_t{0}; i < a.size1(); ++i) {
		for(auto j = std::size_t{0}; j < a.size2(); ++j) {
			const char* sep = (j + 1 < a.size2()) ? " " : "\n";
			std::fprintf(f, fmt, a(i, j).real(), a(i, j).imag(), sep);
		}
	}

	if(fclose(f) < 0) {
		std::perror("fclose");
		std::exit(EXIT_FAILURE);
	}
}


template<class Matrix>
void print_machine_readable_matrix(const char* identifier, const Matrix& a) {
	using Number = typename Matrix::value_type;
	constexpr const char* fmt = std::is_same<Number, float>::value
		? "%s(%zu, %zu) = %+16.9e;\n"
		: "%s(%zu, %zu) = %+24.17e;\n";

	for(auto i = std::size_t{0}; i < a.size1(); ++i) {
		for(auto j = std::size_t{0}; j < a.size2(); ++j) {
			std::printf(fmt, identifier, i, j, a(i, j));
		}
	}
}

template<class Real, class Storage>
void print_machine_readable_matrix(
		const char* identifier,
		const ublas::matrix<std::complex<Real>, Storage>& a
) {
	constexpr const char* fmt = std::is_same<Real, float>::value
		? "%s(%zu, %zu) = std::complex<float>{%+16.9e, %+16.9e};\n"
		: "%s(%zu, %zu) = std::complex<double>{%+24.17e, %+24.17e};\n";

	for(auto i = std::size_t{0}; i < a.size1(); ++i) {
		for(auto j = std::size_t{0}; j < a.size2(); ++j) {
			std::printf(fmt, identifier, i, j, a(i, j).real(), a(i, j).imag());
		}
	}
}

/**
 * @return The Frobenius norm of U* U - I
 */
template<
	typename Number,
	typename Real = typename real_from<Number>::type
>
Real measure_isometry(const ublas::matrix<Number, ublas::column_major>& U)
{
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	BOOST_VERIFY( U.size1() >= U.size2() );

	if(U.size2() == 0)
		return 0;

	auto m = U.size1();
	auto n = U.size2();
	auto I = ublas::identity_matrix<Number>(n);
	auto J = Matrix(n, n);
	auto alpha = Number{1};
	auto beta = Number{0};

	lapack::xGEMM(
		'C', 'N', n, n, m, alpha, &U(0,0), m, &U(0,0), m, beta, &J(0,0), n
	);

	return ublas::norm_frobenius(J - I);
}


/**
 * This function checks if a matrix A might be considered orthogonal or unitary,
 * respectively, by comparing the Frobenius norm of `A*A - I` to a cut-off
 * value.
 *
 * The cut-off value is based on Inequality (19.13), Equation (3.8) in Higham:
 * "Accuracy and Stability of Numerical Algorithms".
 */
template<
	typename Number,
	class Storage,
	typename Real = typename real_from<Number>::type
>
bool is_almost_isometric(
	const ublas::matrix<Number, Storage>& U, Real multiplier = 4)
{
	BOOST_VERIFY( multiplier >= 1 );

	constexpr auto eps = std::numeric_limits<Real>::epsilon();
	auto m = U.size1();
	auto n = U.size2();
	auto p = std::min(m, n);

	if(p == 0)
		return true;

	auto r = measure_isometry(U);
	auto tol = std::sqrt(p) * m * n * eps;

	return r <= multiplier * tol;
}


// std::conj re-implementations that do not forcefully return std::complex
float conj(float f) { return f; }
double conj(double f) { return f; }

template<typename T>
std::complex<T> conj(std::complex<T>& z) {
	return std::conj(z);
}


/**
 * @return Matrix A with m rows, n columns, and A^* A = I.
 */
template<
	typename Number,
	class Engine,
	class Storage = ublas::column_major
>
ublas::matrix<Number, Storage> make_isometric_matrix_like(
	Number, std::size_t m, std::size_t n, Engine* gen)
{
	BOOST_VERIFY( m >= n );

	using Matrix = ublas::matrix<Number, Storage>;

	auto p = std::min(m, n);

	if(p <= 1)
		return ublas::identity_matrix<Number, Storage>(m, n);

	// Approach:
	// 1. Generate a random matrix whose entries are identically and
	// independently distributed with mean 0 and variance 1.
	// 2. Compute the QR factorization.
	// 3. Ensure a unique QR decomposition: compute a unitary diagonal matrix D
	//    such that DR has only positive real-valued entries on its diagonal
	//    (the i-th diagonal entry is `|r(i,i)| / r(i,i)`).
	// 4. Return Q D^*.
	//
	// References:
	// Francesco Mezzadri: "How to generate random matrices from the classical
	// compact groups". 2007. URL: http://arXiv.org/abs/math-ph/0609050v2
	auto dist = UniformDistribution<Number>();
	auto rand = [gen, &dist] () { return dist(*gen); };
	auto A = Matrix(m, n);

	std::generate( A.data().begin(), A.data().end(), rand );

	constexpr auto nan = not_a_number<Number>::value;
	auto tau = ublas::vector<Number>(n, nan);
	auto k = std::max(m, n);
	auto lwork = static_cast<Integer>(2 * k * k);
	auto work = ublas::vector<Number>(lwork, nan);
	auto ret = lapack::xGEQRF(m, n, &A(0,0), m, &tau(0), &work(0), lwork);

	BOOST_VERIFY( ret == 0 );

	// save the diagonal entries of R
	auto r_diagonal = ublas::vector<Number>(n, nan);
	for(auto j = std::size_t{0}; j < n; ++j) {
		r_diagonal(j) = A(j,j);
	}

	// explicitly form Q
	ret = lapack::xUNGQR(m, n, n, &A(0,0), m, &tau(0), &work(0), lwork);

	BOOST_VERIFY( ret == 0 );
	BOOST_VERIFY( is_almost_isometric(A) );

	// compute Q D^*
	for(auto j = std::size_t{0}; j < n; ++j) {
		// exact zeros are known to occur in single-precision arithmetic
		if(r_diagonal(j) == 0) {
			continue;
		}

		auto column_multiplier = std::abs(r_diagonal(j)) / conj(r_diagonal(j));
		for(auto i = std::size_t{0}; i < m; ++i) {
			A(i, j) *= column_multiplier;
		}
	}

	return A;
}

/**
 * @return A random m x n matrix with spectral condition number cond2.
 */
template<
	typename Number,
	class Engine,
	typename Real = typename real_from<Number>::type
>
ublas::matrix<Number, ublas::column_major> make_matrix_like(
	Number dummy, std::size_t m, std::size_t n, Real cond2, Engine* gen)
{
	using Matrix = ublas::matrix<Number, ublas::column_major>;
	using BandedMatrix = ublas::banded_matrix<Number>;

	auto p = std::min(m, n);

	if(p <= 1)
		return ublas::identity_matrix<Number, ublas::column_major>(m, n);

	auto S = BandedMatrix(p, p);
	// do not sort singular values
	S(0,0) = 1;
	S(1,1) = cond2;
	auto sv_dist = std::uniform_real_distribution<Real>(1, cond2);

	for(auto i = std::size_t{2}; i < p; ++i)
		S(i,i) = sv_dist(*gen);

	auto U = make_isometric_matrix_like(dummy, m, p, gen);
	auto V = make_isometric_matrix_like(dummy, n, p, gen);
	auto US = Matrix(ublas::prod(U, S));
	auto alpha = Number{1};
	auto beta = Number{0};
	auto A = Matrix(m, n);

	lapack::xGEMM(
		'N', 'C', m, n, p, alpha, &US(0,0), m, &V(0,0), n, beta, &A(0,0), m
	);

	return A;
}

}
}

#endif
