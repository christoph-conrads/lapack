/*
 * Copyright (c) 2020-2021, 2023 Christoph Conrads
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

#include "config.hpp"
#include "xGGQRCS.hpp"
#include "xGGSVD3.hpp"

#include <cmath>
#include <ctime>

#include <boost/assert.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>


namespace ggqrcs = lapack::ggqrcs;
namespace ggsvd3 = lapack::ggsvd3;
namespace tools = lapack::tools;
namespace ublas = boost::numeric::ublas;

using types = lapack::supported_types;



/**
 * This LAPACK xerbla implementation prints an error message but does not
 * terminate the program thereby allowing the calling LAPACK function to return
 * to its caller.
 *
 * @param[in] caller A string WITHOUT ZERO TERMINATOR
 * @param[in] caller_len The length of the string referenced by caller
 */
extern "C" void xerbla_(
	const char* caller, int* p_info, std::size_t caller_len)
{
	BOOST_VERIFY( caller != nullptr );
	BOOST_VERIFY( p_info != nullptr );

	// "sz" prefix taken from hungarian notation (zero-terminated string)
	char szCaller[80];
	auto num_bytes_to_copy = std::min(sizeof(szCaller)-1, caller_len);

	std::memset(szCaller, 0, sizeof(szCaller));
	std::strncpy(szCaller, caller, num_bytes_to_copy);
	std::fprintf(
		stderr, "%s: parameter %d has illegal value\n", szCaller, *p_info
	);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_simple_1x2, Number, types)
{
	auto m = std::size_t{1};
	auto n = std::size_t{2};
	auto p = std::size_t{1};
	auto caller = ggqrcs::Caller<Number>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0,0) = 1; A(0,1) = +0;
	B(0,0) = 1; B(0,1) = -1;

	caller.A = A;
	caller.B = B;

	auto ret = caller();
	check_results(ret, A, B, caller);

	BOOST_CHECK_EQUAL( caller.rank, 2 );
}


BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_simple_2x2, Number, types)
{
	auto m = std::size_t{2};
	auto n = std::size_t{2};
	auto p = std::size_t{2};
	auto caller = ggqrcs::Caller<Number>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0,0) = 1;
	B(1,1) = 1;

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	check_results(ret, A, B, caller);

	BOOST_CHECK_EQUAL( caller.rank, 2 );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_simple_1_2_3, Number, types)
{
	auto m = std::size_t{1};
	auto n = std::size_t{2};
	auto p = std::size_t{3};
	auto caller = ggqrcs::Caller<Number>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0,0) = 1; A(0,1) = 1;
	            B(0,1) = 100;

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	check_results(ret, A, B, caller);
	BOOST_CHECK_EQUAL( caller.rank, 2 );
}



BOOST_AUTO_TEST_CASE_TEMPLATE(
	preprocessing, Number, types)
{
	for(auto m = std::size_t{1}; m < std::size_t{4}; ++m) {
		for(auto n = std::size_t{1}; n < std::size_t{6}; ++n) {
			for(auto p = std::size_t{1}; p < std::size_t{4}; ++p) {
				for(auto row_a = std::size_t{0}; row_a < m; ++row_a) {
					for(auto row_b = std::size_t{0}; row_b < p; ++row_b) {
						auto caller = ggqrcs::Caller<Number>(m, n, p);
						auto A = caller.A;
						auto B = caller.B;

						if(m > row_a && n > 0) {
							A(row_a,0) = 3;
						}
						if(p > row_b && n > 1) {
							B(row_b,1) = 4;
						}
						if(n > 2) {
							A(row_a, 2) = 1;
							B(row_b, 2) = 1;
						}

						caller.A = A;
						caller.B = B;
						caller.hint_preprocess_a = 'Y';
						caller.hint_preprocess_b = 'Y';

						auto ret = caller();

						check_results(ret, A, B, caller);
					}
				}
			}
		}
	}
}


BOOST_AUTO_TEST_CASE(preprocessing_g_regression_1)
{
	auto caller = ggqrcs::Caller<float>(1, 4, 1);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = +1.9089578629e+01;
	A(0, 1) = -8.3061685562e+00;
	A(0, 2) = +2.3080570221e+01;
	A(0, 3) = -2.6640016556e+01;
	B(0, 0) = -3.2287300110e+01;
	B(0, 1) = +1.4465919495e+01;
	B(0, 2) = -4.0601890564e+01;
	B(0, 3) = +4.8479816437e+01;

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	check_results(ret, A, B, caller);
}


BOOST_AUTO_TEST_CASE(preprocessing_g_regression_2)
{
	auto caller = ggqrcs::Caller<float>(3, 20, 3);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = -7.4534997940e+00;
	A(0, 1) = -1.2052089691e+01;
	A(0, 2) = -2.5186979294e+01;
	A(0, 3) = -9.8890800476e+00;
	A(0, 4) = +9.9974746704e+00;
	A(0, 5) = -5.1145874023e+01;
	A(0, 6) = -2.0494287491e+01;
	A(0, 7) = +7.5930290222e+00;
	A(0, 8) = +3.4897159576e+01;
	A(0, 9) = +1.8601310730e+01;
	A(0, 10) = +3.9464145660e+01;
	A(0, 11) = +3.5205165863e+01;
	A(0, 12) = +3.9551908970e+00;
	A(0, 13) = +1.1402584076e+01;
	A(0, 14) = -4.4898872375e+01;
	A(0, 15) = -2.8564651489e+01;
	A(0, 16) = -2.1241870880e+01;
	A(0, 17) = +3.0776770115e+00;
	A(0, 18) = +6.0094065666e+00;
	A(0, 19) = -2.0459209442e+01;
	A(1, 0) = -4.8526811600e+00;
	A(1, 1) = -7.8709039688e+00;
	A(1, 2) = -1.6456127167e+01;
	A(1, 3) = -6.4214162827e+00;
	A(1, 4) = +6.4817790985e+00;
	A(1, 5) = -3.3377914429e+01;
	A(1, 6) = -1.3378832817e+01;
	A(1, 7) = +4.9821729660e+00;
	A(1, 8) = +2.2786708832e+01;
	A(1, 9) = +1.2112653732e+01;
	A(1, 10) = +2.5754419327e+01;
	A(1, 11) = +2.2968740463e+01;
	A(1, 12) = +2.5615477562e+00;
	A(1, 13) = +7.4083642960e+00;
	A(1, 14) = -2.9314395905e+01;
	A(1, 15) = -1.8661394119e+01;
	A(1, 16) = -1.3844952583e+01;
	A(1, 17) = +1.9941833019e+00;
	A(1, 18) = +3.9057414532e+00;
	A(1, 19) = -1.3364761353e+01;
	A(2, 0) = +7.9642271996e+00;
	A(2, 1) = +1.2883576393e+01;
	A(2, 2) = +2.6927888870e+01;
	A(2, 3) = +1.0563210487e+01;
	A(2, 4) = -1.0676908493e+01;
	A(2, 5) = +5.4672367096e+01;
	A(2, 6) = +2.1909355164e+01;
	A(2, 7) = -8.1237182617e+00;
	A(2, 8) = -3.7306938171e+01;
	A(2, 9) = -1.9877315521e+01;
	A(2, 10) = -4.2182899475e+01;
	A(2, 11) = -3.7629554749e+01;
	A(2, 12) = -4.2238211632e+00;
	A(2, 13) = -1.2180659294e+01;
	A(2, 14) = +4.7995677948e+01;
	A(2, 15) = +3.0537448883e+01;
	A(2, 16) = +2.2700466156e+01;
	A(2, 17) = -3.2877850533e+00;
	A(2, 18) = -6.4189724922e+00;
	A(2, 19) = +2.1873659134e+01;
	B(0, 0) = -6.4645099640e-01;
	B(0, 1) = -4.3905949593e+00;
	B(0, 2) = +5.9523549080e+00;
	B(0, 3) = +6.4257392883e+00;
	B(0, 4) = -1.0519197464e+01;
	B(0, 5) = +2.1125761032e+01;
	B(0, 6) = +1.8307855606e+01;
	B(0, 7) = -8.3596887589e+00;
	B(0, 8) = -1.9427293777e+01;
	B(0, 9) = -7.8572788239e+00;
	B(0, 10) = +7.6021795273e+00;
	B(0, 11) = +5.8613586426e-01;
	B(0, 12) = -9.6025810242e+00;
	B(0, 13) = -4.7796545029e+00;
	B(0, 14) = -4.7791328430e+00;
	B(0, 15) = -5.6490306854e+00;
	B(0, 16) = -8.8328828812e+00;
	B(0, 17) = -1.6209197998e+01;
	B(0, 18) = +6.1756000519e+00;
	B(0, 19) = +1.4489887238e+01;
	B(1, 0) = -2.8246524811e+01;
	B(1, 1) = -6.0235071182e+00;
	B(1, 2) = +2.6751358032e+01;
	B(1, 3) = -5.7417312622e+01;
	B(1, 4) = +7.0231727600e+01;
	B(1, 5) = -4.4798183441e+00;
	B(1, 6) = +2.3379827499e+01;
	B(1, 7) = -5.9239257812e+01;
	B(1, 8) = -3.0095972061e+01;
	B(1, 9) = +5.1800170898e+01;
	B(1, 10) = +4.2670051575e+01;
	B(1, 11) = +4.0242782593e+01;
	B(1, 12) = +2.4640577316e+01;
	B(1, 13) = +6.3083602905e+01;
	B(1, 14) = -1.7027751923e+01;
	B(1, 15) = +7.0571880341e+00;
	B(1, 16) = -6.4803588867e+01;
	B(1, 17) = +2.6058349609e+00;
	B(1, 18) = +4.5089683533e+01;
	B(1, 19) = +3.2905242920e+01;
	B(2, 0) = -5.2207756042e+00;
	B(2, 1) = +1.5447164536e+01;
	B(2, 2) = +2.7230806351e+00;
	B(2, 3) = -3.3861686707e+01;
	B(2, 4) = +4.9599311829e+01;
	B(2, 5) = -3.8039974213e+01;
	B(2, 6) = -3.5415573120e+01;
	B(2, 7) = +2.5617790222e-01;
	B(2, 8) = +2.9875316620e+01;
	B(2, 9) = +3.1033021927e+01;
	B(2, 10) = -2.2411264420e+01;
	B(2, 11) = -3.3649044037e+00;
	B(2, 12) = +3.2331592560e+01;
	B(2, 13) = +3.0034729004e+01;
	B(2, 14) = +2.5472904205e+01;
	B(2, 15) = +2.8862970352e+01;
	B(2, 16) = +1.2360656738e+01;
	B(2, 17) = +4.4093997955e+01;
	B(2, 18) = -4.1140651703e+00;
	B(2, 19) = -2.0133462906e+01;

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	BOOST_CHECK_EQUAL( caller.rank, 4 );
	check_results(ret, A, B, caller);
}


BOOST_AUTO_TEST_CASE(regression_preprocessing_20210524)
{
	auto caller = ggqrcs::Caller<float>(3, 12, 3);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = +1.9984721439e-04;
	A(0, 1) = -8.9995574672e-04;
	A(0, 2) = -7.4270862387e-04;
	A(0, 3) = -3.3741045627e-04;
	A(0, 4) = +2.4893784939e-05;
	A(0, 5) = -6.2635983340e-04;
	A(0, 6) = -4.5278990001e-05;
	A(0, 7) = +1.1171223596e-03;
	A(0, 8) = +1.5522693866e-04;
	A(0, 9) = -6.6558702383e-04;
	A(0, 10) = -4.2838969966e-04;
	A(0, 11) = -7.6151467510e-05;
	B(0, 0) = -5.1384512335e-04;
	B(0, 1) = -4.1028897977e-04;
	B(0, 2) = +4.5113207307e-05;
	B(0, 3) = -5.8734833146e-05;
	B(0, 4) = -2.2601697128e-04;
	B(0, 5) = -2.8823193861e-04;
	B(0, 6) = -1.2712553144e-07;
	B(0, 7) = +1.1580594582e-04;
	B(0, 8) = +1.3096502516e-03;
	B(0, 9) = +6.7491852678e-04;
	B(0, 10) = +6.8548473064e-04;
	B(0, 11) = +2.3072116892e-04;
	B(1, 0) = -4.1295963456e-05;
	B(1, 1) = -6.8008419476e-05;
	B(1, 2) = +9.9207204767e-04;
	B(1, 3) = +6.3558039255e-04;
	B(1, 4) = +3.8785778452e-04;
	B(1, 5) = -6.9055025233e-04;
	B(1, 6) = -7.6664646622e-04;
	B(1, 7) = +7.1330112405e-04;
	B(1, 8) = -3.7726166192e-04;
	B(1, 9) = +5.2251038142e-05;
	B(1, 10) = +6.6571909701e-04;
	B(1, 11) = -4.0718507080e-05;
	B(2, 0) = +2.0352652064e-04;
	B(2, 1) = +1.5609170077e-04;
	B(2, 2) = +1.6319473798e-04;
	B(2, 3) = +1.4055416977e-04;
	B(2, 4) = +1.6389702796e-04;
	B(2, 5) = -8.0872559920e-06;
	B(2, 6) = -1.4038194786e-04;
	B(2, 7) = +8.3088554675e-05;
	B(2, 8) = -6.0711998958e-04;
	B(2, 9) = -2.6768984389e-04;
	B(2, 10) = -1.5965555212e-04;
	B(2, 11) = -1.0224065045e-04;

	caller.A = A;
	caller.B = B;
	caller.hint_preprocess_a = 'N';
	caller.hint_preprocess_b = 'Y';
	caller.hint_preprocess_cols = 'Y';

	auto ret = caller();

	BOOST_CHECK_EQUAL( caller.hint_preprocess_a, 'N' );
	BOOST_CHECK_EQUAL( caller.hint_preprocess_b, 'Y' );
	BOOST_CHECK_EQUAL( caller.hint_preprocess_cols, 'Y' );
	BOOST_CHECK_EQUAL( caller.rank, 3 );
	check_results(ret, A, B, caller);
}



BOOST_AUTO_TEST_CASE_TEMPLATE(
	xGGQRCS_test_workspace_size_check, Number, lapack::supported_real_types
)
{
	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	constexpr auto nan = tools::not_a_number<Number>::value;
	constexpr auto nan_real = tools::not_a_number<Real>::value;
	auto m = 17;
	auto n = 13;
	auto p = 8;
	auto hpa = '?';
	auto hpb = '?';
	auto hpc = '?';
	auto rank = -1;
	auto swapped_p = false;
	auto a = Matrix(m, n, 1);
	auto b = Matrix(p, n, 1);
	auto alpha = std::vector<Real>(n, nan_real);
	auto beta = std::vector<Real>(n, nan_real);
	auto u1 = Matrix(m, m, nan);
	auto u2 = Matrix(p, p, nan);
	auto x = Matrix(n, n, nan);
	auto tol = Real{0};
	auto work = Real{0};
	auto iwork = -1;
	auto info = lapack::xGGQRCS(
		'Y', 'Y', 'Y', &hpa, &hpb, &hpc,
		m, n, p, &rank, &swapped_p,
		&a(0, 0), m, &b(0, 0), p,
		&alpha[0], &beta[0],
		&u1(0, 0), m, &u2(0, 0), p, &x(0, 0), m+p,
		&tol,
		&work, 1, &iwork
	);

	BOOST_CHECK_EQUAL(info, -25);
}



BOOST_AUTO_TEST_CASE_TEMPLATE(
	xGGQRCS_test_workspace_size_check_complex, Number, lapack::supported_complex_types
)
{
	using Real = typename tools::real_from<Number>::type;

	auto m = 17;
	auto n = 13;
	auto p = 8;
	auto hpa = '?';
	auto hpb = '?';
	auto hpc = '?';
	auto rank = -1;
	auto swapped_p = false;
	auto a = Number{1};
	auto b = Number{2};
	auto alpha = Real{-1};
	auto beta = Real{-1};
	auto tol = Real{-1};
	auto u1 = Number{0};
	auto u2 = Number{0};
	auto x = Number{0};
	auto work = Number{0};
	auto rwork = Real{0};
	auto iwork = -1;
	auto info = lapack::xGGQRCS(
		'Y', 'Y', 'Y', &hpa, &hpb, &hpc, m, n, p, &rank, &swapped_p,
		&a, m, &b, p,
		&alpha, &beta,
		&u1, m, &u2, p, &x, m + p,
		&tol,
		&work, 1, &rwork, 1024, &iwork
	);

	BOOST_CHECK_EQUAL(info, -20);

	info = lapack::xGGQRCS(
		'Y', 'Y', 'Y', &hpa, &hpb, &hpc, m, n, p, &rank, &swapped_p,
		&a, m, &b, p,
		&alpha, &beta,
		&u1, m, &u2, p, &x, m + p,
		&tol,
		&work, 1024, &rwork, 1, &iwork
	);

	BOOST_CHECK_EQUAL(info, -22);
}




// this test does not pass with row sorting and no matrix scaling
BOOST_AUTO_TEST_CASE(xGGQRCS_test_matrix_scaling)
{
	using Number = float;

	auto m = std::size_t{1};
	auto n = std::size_t{2};
	auto p = std::size_t{10};
	auto caller = ggqrcs::Caller<Number>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0,0) = -8.519847412e+02; A(0,1) = +6.469862671e+02;
	B(0,0) = +5.485938125e+05; B(0,1) = -4.166526250e+05;
	B(1,0) = +1.846850781e+05; B(1,1) = -1.402660781e+05;
	B(2,0) = +5.322575625e+05; B(2,1) = -4.042448438e+05;
	B(3,0) = -1.630551465e+04; B(3,1) = +1.238360352e+04;
	B(4,0) = -1.286453438e+05; B(4,1) = +9.770555469e+04;
	B(5,0) = -1.323287812e+05; B(5,1) = +1.005026797e+05;
	B(6,0) = +5.681228750e+05; B(6,1) = -4.314841250e+05;
	B(7,0) = -3.107875312e+05; B(7,1) = +2.360408594e+05;
	B(8,0) = +1.456551719e+05; B(8,1) = -1.106233281e+05;
	B(9,0) = +1.365355156e+05; B(9,1) = -1.036972344e+05;

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	check_results(ret, A, B, caller);
}



BOOST_TEST_DECORATOR(* boost::unit_test::expected_failures(1))
BOOST_AUTO_TEST_CASE(xGGQRCS_test_conditional_backward_stability)
{
	using Number = float;
	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	constexpr auto eps = std::numeric_limits<Real>::epsilon();
	auto tol = [] (const Matrix& a) {
		return std::max(a.size1(), a.size2()) * ublas::norm_frobenius(a) * eps;
	};

	auto m = std::size_t{2};
	auto n = std::size_t{2};
	auto p = std::size_t{2};
	auto r = std::size_t{2};
	auto A = Matrix(m, n);
	auto B = Matrix(p, n);

	A(0,0) = +4.663013916e+02; A(0,1) = +4.046628418e+02;
	A(1,0) = +3.062543457e+03; A(1,1) = +2.648934082e+03;
	B(0,0) = -2.966550887e-01; B(0,1) = -2.563934922e-01;
	B(1,0) = +7.012547851e-01; B(1,1) = +6.062732935e-01;

	// compute GSVD of A, B and fail
	{
		auto caller = ggqrcs::Caller<Number>(m, n, p);

		caller.A = A;
		caller.B = B;

		auto ret = caller();
		BOOST_REQUIRE_EQUAL( ret, 0 );

		check_results(ret, A, B, caller);

		auto X = copy_X(caller);
		auto ds = ggqrcs::assemble_diagonals_like(
			Number{}, m, p, r, caller.swapped_p, caller.alpha, caller.beta
		);
		auto& D1 = ds.first;
		auto& D2 = ds.second;
		auto almost_A = ggqrcs::assemble_matrix(caller.U1, D1, X);
		auto almost_B = ggqrcs::assemble_matrix(caller.U2, D2, X);

		BOOST_CHECK_LE(ublas::norm_frobenius(A - almost_A), tol(A));
		// should fail
		BOOST_CHECK_LE(ublas::norm_frobenius(B - almost_B), tol(B));
	}

	// try again with norm(A) = norm(B)
	{
		auto w = std::ldexp(Real{1}, 12);
		auto caller = ggqrcs::Caller<Number>(m, n, p);

		caller.A = A;
		caller.B = w * B;

		auto ret = caller();
		BOOST_REQUIRE_EQUAL( ret, 0 );

		auto X = copy_X(caller);
		auto ds = ggqrcs::assemble_diagonals_like(
			Number{}, m, p, r, caller.swapped_p, caller.alpha, caller.beta
		);
		auto& D1 = ds.first;
		auto& D2 = ds.second;
		auto almost_A = ggqrcs::assemble_matrix(caller.U1, D1, X);
		auto almost_B = ggqrcs::assemble_matrix(caller.U2, D2, X);

		BOOST_CHECK_LE(ublas::norm_frobenius(A - almost_A), tol(A));
		BOOST_CHECK_LE(ublas::norm_frobenius(w*B - almost_B), tol(w*B));
	}
}



BOOST_TEST_DECORATOR(* boost::unit_test::expected_failures(1))
BOOST_AUTO_TEST_CASE(xGGQRCS_test_singular_accuracy_vs_radians_accuracy)
{
	using Number = float;
	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	constexpr auto eps = std::numeric_limits<Real>::epsilon();
	auto m = std::size_t{2};
	auto n = std::size_t{1};
	auto p = std::size_t{2};
	auto r = std::size_t{1};
	auto A = Matrix(m, n);
	auto B = Matrix(p, n);

	// you can probably turn this into a test with a pair of 1x1 matrices
	A(0,0) = +4.369503870e-07;
	A(1,0) = +1.496136406e-06;
	B(0,0) = -7.727422714e-01;
	B(1,0) = +6.347199082e-01;

	auto caller = ggqrcs::Caller<Number>(m, n, p);

	caller.A = A;
	caller.B = B;

	auto ret = caller();

	check_results(ret, A, B, caller);

	BOOST_REQUIRE_EQUAL( caller.rank, 1 );
	BOOST_REQUIRE( !caller.swapped_p );

	// computed with 2xSVD in double precision
	auto cos = Real{1.5586372e-06};

	// should fail
	BOOST_CHECK_LE(std::abs(cos - caller.alpha(0)), eps*cos);
	BOOST_CHECK_LE(std::abs(Real{1} - caller.beta(0)), eps);

	auto X = copy_X(caller);
	auto ds = ggqrcs::assemble_diagonals_like(
		Number{}, m, p, r, caller.swapped_p, caller.alpha, caller.beta);
	auto& D1 = ds.first;
	auto& D2 = ds.second;
	auto almost_A = ggqrcs::assemble_matrix(caller.U1, D1, X);
	auto almost_B = ggqrcs::assemble_matrix(caller.U2, D2, X);
	auto tol = [] (const Matrix& a) {
		return std::max(a.size1(), a.size2()) * ublas::norm_frobenius(a) * eps;
	};

	// should fail
	BOOST_CHECK_LE(ublas::norm_frobenius(A - almost_A), tol(A));
	BOOST_CHECK_LE(ublas::norm_frobenius(B - almost_B), tol(B));
}



template<
	typename Number,
	typename std::enable_if<
		std::is_fundamental<Number>::value, int
	>::type* = nullptr
>
void xGGQRCS_test_zero_dimensions_impl(
	std::size_t m, std::size_t n, std::size_t p) {
	using Storage = ublas::column_major;
	using Matrix = ublas::matrix<Number, Storage>;
	using Real = typename tools::real_from<Number>::type;
	using Integer = lapack::integer_t;

	constexpr auto nan = tools::not_a_number<Number>::value;
	constexpr auto real_nan = tools::not_a_number<Real>::value;
	constexpr auto one = std::size_t{1};

	auto hpa = '?';
	auto hpb = '?';
	auto hpc = '?';
	auto k = std::max(std::min(m + p, n), one);
	auto rank = Integer{-1};
	auto swapped_p = false;
	auto lda = std::max(m, one);
	auto A = Matrix(lda, std::max(n, one), 1);
	auto ldb = std::max(p, one);
	auto B = Matrix(ldb, std::max(n, one), 1);
	auto alpha = std::vector<Real>(k, real_nan);
	auto beta = std::vector<Real>(k, real_nan);
	auto ldu1 = std::max(m, one);
	auto U1 = Matrix(ldu1, std::max(m, one), nan);
	auto ldu2 = std::max(p, one);
	auto U2 = Matrix(ldu2, std::max(p, one), nan);
	auto ldx = std::max(std::min(m + p, n), one);
	auto X = Matrix(ldx, std::max(n, one), nan);
	auto tol = Real{0};
	// this must be large enough not to trigger the workspace size check
	auto lwork = std::max(4 * (m + p) * n, std::size_t{128});
	auto work = std::vector<Number>(lwork, nan);
	auto iwork = std::vector<Integer>(lwork, -1);
	auto ret = lapack::xGGQRCS(
		'Y', 'Y', 'N', &hpa, &hpb, &hpc,
		m, n, p, &rank, &swapped_p,
		&A(0, 0), lda, &B(0, 0), ldb,
		&alpha[0], &beta[0],
		&U1(0, 0), 1, &U2(0, 0), 1, &X(0, 0), 1,
		&tol,
		&work[0], lwork, &iwork[0]
	);

	BOOST_REQUIRE_EQUAL(ret, 0);

	constexpr auto eps = std::numeric_limits<Real>::epsilon();
	auto nan_p = [] (const Number& x) { return tools::nan_p(x); };

	if(m > 0) {
		BOOST_REQUIRE(std::none_of(U1.data().begin(), U1.data().end(), nan_p));
		BOOST_CHECK_LE(tools::measure_isometry(U1), m * eps);
	}
	if(p > 0) {
		BOOST_REQUIRE(std::none_of(U2.data().begin(), U2.data().end(), nan_p));
		BOOST_CHECK_LE(tools::measure_isometry(U2), p * eps);
	}
}

template<
	typename Number,
	typename std::enable_if<
		!std::is_fundamental<Number>::value, int
	>::type* = nullptr
>
void xGGQRCS_test_zero_dimensions_impl(
	std::size_t m, std::size_t n, std::size_t p) {
	using Storage = ublas::column_major;
	using Matrix = ublas::matrix<Number, Storage>;
	using Real = typename tools::real_from<Number>::type;
	using Integer = lapack::integer_t;

	constexpr auto nan = tools::not_a_number<Number>::value;
	constexpr auto real_nan = tools::not_a_number<Real>::value;
	constexpr auto one = std::size_t{1};

	auto hpa = '?';
	auto hpb = '?';
	auto hpc = '?';
	auto k = std::max(std::min(m + p, n), one);
	auto rank = Integer{-1};
	auto swapped_p = false;
	auto lda = std::max(m, one);
	auto A = Matrix(lda, std::max(n, one), 1);
	auto ldb = std::max(p, one);
	auto B = Matrix(ldb, std::max(n, one), 1);
	auto alpha = std::vector<Real>(k, real_nan);
	auto beta = std::vector<Real>(k, real_nan);
	auto ldu1 = std::max(m, one);
	auto U1 = Matrix(ldu1, std::max(m, one), nan);
	auto ldu2 = std::max(p, one);
	auto U2 = Matrix(ldu2, std::max(p, one), nan);
	auto ldx = std::max(m + p, one);
	auto X = Matrix(ldx, std::max(n, one), nan);
	auto tol = Real{0};
	// this must be large enough not to trigger the workspace size check
	auto lwork = std::max(4 * (m + p) * n, std::size_t{128});
	auto work = std::vector<Number>(lwork, nan);
	auto rwork = std::vector<Real>(lwork, real_nan);
	auto iwork = std::vector<Integer>(lwork, -1);
	auto ret = lapack::xGGQRCS(
		'Y', 'Y', 'N',
		&hpa, &hpb, &hpc,
		m, n, p, &rank, &swapped_p,
		&A(0, 0), lda, &B(0, 0), ldb,
		&alpha[0], &beta[0],
		&U1(0, 0), 1, &U2(0, 0), 1,
		&X(0, 0), 1,
		&tol,
		&work[0], work.size(), &rwork[0], rwork.size(), &iwork[0]
	);

	BOOST_REQUIRE_EQUAL(ret, 0);

	constexpr auto eps = std::numeric_limits<Real>::epsilon();
	auto nan_p = [] (const Number& x) { return tools::nan_p(x); };

	if(m > 0) {
		BOOST_REQUIRE(std::none_of(U1.data().begin(), U1.data().end(), nan_p));
		BOOST_CHECK_LE(tools::measure_isometry(U1), m * eps);
	}
	if(p > 0) {
		BOOST_REQUIRE(std::none_of(U2.data().begin(), U2.data().end(), nan_p));
		BOOST_CHECK_LE(tools::measure_isometry(U2), p * eps);
	}
}


BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_zero_dimensions, Number, types)
{
	for(auto m = std::size_t{0}; m < std::size_t{2}; ++m) {
		for(auto n = std::size_t{0}; n < std::size_t{2}; ++n) {
			for(auto p = std::size_t{0}; p < std::size_t{2}; ++p) {
				xGGQRCS_test_zero_dimensions_impl<Number>(m, n, p);
			}
		}
	}
}



BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_zero_input, Number, types)
{
	auto m = std::size_t{4};
	auto n = std::size_t{3};
	auto p = std::size_t{2};
	auto caller = ggqrcs::Caller<Number>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	auto ret = caller();
	check_results(ret, A, B, caller);

	BOOST_CHECK_EQUAL( caller.rank, 0 );
}


BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_rectangular_input, Number, types)
{
	for(std::size_t m : { 2, 13, 41 })
	{
		for(std::size_t n : {3, 7, 31})
		{
			for(std::size_t p : {5, 11, 17})
			{
				auto caller = ggqrcs::Caller<Number>(m, n, p);
				auto A = caller.A;
				auto B = caller.B;

				A(0,0) = 1;
				A(1,0) = 1;
				B(0,1) = 1;
				B(1,1) = 1;

				caller.A = A;
				caller.B = B;
				caller.hint_preprocess_a = 'N';
				caller.hint_preprocess_b = 'N';

				auto ret = caller();
				check_results(ret, A, B, caller);
			}
		}
	}
}



/**
 * This test checks the generalized singular values computed by xGGQRCS when
 * the matrices `A` and `B` differ significantly in norm.
 *
 * The GSVD allows us to decompose `A` and `B` into
 * * `A = U1 S R Q^*`,
 * * `B = U2 C R Q^*`,
 *
 * where
 * * `Q^*` is the complex-conjugate transpose of `Q`,
 * * `U1, `U2`, `Q` are unitary, and
 * * `S`, `C` are diagonal matrices which values `s_ii` and `c_ii` such that
 *   `s_ii/c_ii` is one of the generalized singular values of the matrix pencil
 *   `(A, B)`.
 *
 * To generate matrices, `A`, `B` with `A` much larger in norm than `B`, we
 * compute
 * * a random matrix `R Q^*`, and
 * * generalized singular values such that `s_ii >> c_ii`.
 */
BOOST_TEST_DECORATOR(* boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_singular_values, Number, types)
{
	// Numbers of the form 4n+1 are used here so that the the median as well as
	// the 25th and the 75th percentiles can be computed easily for the
	// five-number summary.
	// (Sort the values xs and use xs[0], xs[n/4], xs[n/2], xs[n/4*3], xs[n-1].)

	using Real = typename tools::real_from<Number>::type;

	auto gen = std::mt19937();

	gen.discard(1u << 17);

	std::printf(
		"%2s %8s  %44s  %44s\n",
		"d", "condition", "five-number summary norm(A)/norm(B)",
		"five-number summary relative forward error"
	);

	for(auto option = 1u; option <= 2u; ++option)
	{
		for(auto d = std::size_t{2}; d <= 65; d += 20)
		{
			auto real_nan = tools::not_a_number<Real>::value;
			auto stats_fe = ublas::vector<Real>(5, 0);
			auto cond_max = Real{0};
			auto num_iterations = std::size_t{101};
			auto rel_norms = ublas::vector<Real>(num_iterations, real_nan);

			for(auto it = std::size_t{0}; it < num_iterations; ++it)
			{
				auto m = d;
				auto n = d;
				auto p = d;
				auto r = std::min(m+p, n) - 1;
				auto k = std::min( {m, p, r, m + p - r} );

				BOOST_TEST_CONTEXT("m=" << m) {
				BOOST_TEST_CONTEXT("n=" << n) {
				BOOST_TEST_CONTEXT("p=" << p) {
				BOOST_TEST_CONTEXT("rank=" << r) {

				BOOST_VERIFY(k > 0);

				auto theta_dist = ggqrcs::ThetaDistribution<Real>(option);
				auto theta = ublas::vector<Real>(k, real_nan);

				std::generate(
					theta.begin(), theta.end(),
					[&gen, &theta_dist](){ return theta_dist(gen); }
				);
				std::sort(theta.begin(), theta.end());

				auto dummy = Number{};
				// Do not condition `X` too badly or we cannot directly compare
				// the computed generalized singular values with the generated
				// singular values.
				auto digits = std::numeric_limits<Real>::digits;
				auto cond_X = static_cast<Real>(1 << (digits/4));
				auto X = tools::make_matrix_like(dummy, r, n, cond_X, &gen);
				auto U1 = tools::make_isometric_matrix_like(dummy, m, m, &gen);
				auto U2 = tools::make_isometric_matrix_like(dummy, p, p, &gen);
				auto ds = ggqrcs::assemble_diagonals_like(dummy, m, p, r, theta);
				auto D1 = ds.first;
				auto D2 = ds.second;
				auto A = ggqrcs::assemble_matrix(U1, D1, X);
				auto B = ggqrcs::assemble_matrix(U2, D2, X);
				auto caller = ggqrcs::Caller<Number>(m, n, p);

				caller.A = A;
				caller.B = B;

				auto ret = caller();
				auto be_errors = check_results(ret, A, B, caller);
				auto norm_A = ublas::norm_frobenius(A);
				auto norm_B = ublas::norm_frobenius(B);
				auto min_be_error =
					std::min(be_errors.first/norm_A, be_errors.second/norm_B);

				BOOST_REQUIRE_LE(caller.rank, r);

				auto rank = static_cast<std::size_t>(caller.rank);
				auto l = std::min({m, p, rank, m+p-rank});
				auto alpha = ublas::subrange(caller.alpha, 0, l);
				auto beta = ublas::subrange(caller.beta, 0, l);
				auto eps = std::numeric_limits<Real>::epsilon();
				// relative forward error
				auto delta_fe = ublas::vector<Real>(l, real_nan);

				BOOST_REQUIRE_EQUAL(l, k);

				for(auto i = std::size_t{0}; i < l; ++i)
				{
					auto x = theta(i);
					auto abs = [] (Real x) { return std::abs(x); };
					auto cos = [] (Real x) { return std::cos(x); };
					auto sin = [] (Real x) { return std::sin(x); };
					auto rel_forward_error =
						(cos(x) >= sin(x)) ? abs(cos(x)-beta(i)) / cos(x) :
						                     abs(sin(x)-alpha(i)) / sin(x)
					;
					delta_fe(i) = rel_forward_error / eps;
					cond_max =
						std::max(cond_max, rel_forward_error/min_be_error);
				}

				std::sort(delta_fe.begin(), delta_fe.end());

				stats_fe(0) = std::min(stats_fe(0), delta_fe(0));
				stats_fe(1) += delta_fe(1*l/4);
				stats_fe(2) += delta_fe(2*l/4);
				stats_fe(3) += delta_fe(3*l/4);
				stats_fe(4) = std::max(stats_fe(4), delta_fe(l-1));

				rel_norms(it) = ublas::norm_frobenius(A) / ublas::norm_frobenius(B);
			}
			}
			}
			}
			}

			stats_fe(1) /= num_iterations;
			stats_fe(2) /= num_iterations;
			stats_fe(3) /= num_iterations;

			std::sort(rel_norms.begin(), rel_norms.end());

			auto rel_norm_0 =   rel_norms(num_iterations/4*0);
			auto rel_norm_25 =  rel_norms(num_iterations/4*1);
			auto rel_norm_50 =  rel_norms(num_iterations/4*2);
			auto rel_norm_75 =  rel_norms(num_iterations/4*3);
			auto rel_norm_100 = rel_norms(num_iterations/4*4);

			std::printf(
				"%2zu  %8.2e  %8.2e %8.2e %8.2e %8.2e %8.2e  %8.2e %8.2e %8.2e %8.2e %8.2e\n",
				d,
				cond_max,
				rel_norm_0, rel_norm_25, rel_norm_50, rel_norm_75, rel_norm_100,
				stats_fe(0), stats_fe(1), stats_fe(2), stats_fe(3), stats_fe(4)
			);
		}
	}
}



BOOST_TEST_DECORATOR(* boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_row_scaling, Number, types)
{
	using Real = typename tools::real_from<Number>::type;

	auto gen = std::mt19937();

	gen.discard(1u << 17);

	std::printf(
		"%2s %44s %44s\n",
		"d",
		"five-number summary relative backward error A",
		"five-number summary relative backward error B"
	);

	for(auto d = std::size_t{5}; d <= 45; d += 10)
	{
		auto num_iterations = std::size_t{101};
		auto real_nan = tools::not_a_number<Real>::value;
		// five-point summary backward error A
		auto stats_be_a = ublas::vector<Real>(num_iterations, real_nan);
		// five-point summary backward error B
		auto stats_be_b = ublas::vector<Real>(num_iterations, real_nan);

		for(auto it = std::size_t{0}; it < num_iterations; ++it)
		{
			auto m = d;
			auto n = d;
			auto p = 2*d;

			BOOST_TEST_CONTEXT("m=" << m) {
			BOOST_TEST_CONTEXT("n=" << n) {
			BOOST_TEST_CONTEXT("p=" << p) {
			BOOST_TEST_CONTEXT("iteration=" << it) {

			auto dummy = Number{};
			auto digits = std::numeric_limits<Real>::digits;
			auto cond = static_cast<Real>(1 << (digits/4));
			auto A = tools::make_matrix_like(dummy, m, n, cond, &gen);
			auto B = tools::make_matrix_like(dummy, p, n, cond, &gen);

			// row scaling
			auto row_norms_a = ublas::vector<Real>(m, Real{0});
			auto row_norms_b = ublas::vector<Real>(p, Real{0});

			for(auto j = std::size_t{0}; j < n; ++j)
			{
				for(auto i = std::size_t{0}; i < m; ++i)
					row_norms_a(i) = std::max(row_norms_a(i), std::abs(A(i,j)));
				for(auto i = std::size_t{0}; i < p; ++i)
					row_norms_b(i) = std::max(row_norms_b(i), std::abs(B(i,j)));
			}

			auto kappa = std::ldexp(Real{1}, digits/2);
			for(auto j = std::size_t{0}; j < n; ++j)
			{
				//for(auto i = std::size_t{0}; i < m; ++i)
				//	A(i,j) *= kappa * i / (m-1) / row_norms_a(i);
				for(auto i = std::size_t{0}; i < p; ++i)
					B(i,j) *= kappa * i / (p-1) / row_norms_b(i);
			}

			auto caller = ggqrcs::Caller<Number>(m, n, p);

			caller.A = A;
			caller.B = B;

			auto ret = caller();
			auto be_errors = check_results(ret, A, B, caller);

			stats_be_a(it) = be_errors.first;
			stats_be_b(it) = be_errors.second;
		}
		}
		}
		}
		}

		auto k = num_iterations - 1;

		std::sort(stats_be_a.begin(), stats_be_a.end());
		std::sort(stats_be_b.begin(), stats_be_b.end());
		std::printf(
			"%2zu  %8.2e %8.2e %8.2e %8.2e %8.2e  %8.2e %8.2e %8.2e %8.2e %8.2e\n",
			d,
			stats_be_a(k*0), stats_be_a(k/4), stats_be_a(k/2), stats_be_a(k/4*3), stats_be_a(k),
			stats_be_b(k*0), stats_be_b(k/4), stats_be_b(k/2), stats_be_b(k/4*3), stats_be_b(k)
		);
	}
}





template<typename Number>
void xGGQRCS_test_random_impl(
	Number dummy,
	std::size_t m, std::size_t n, std::size_t p, std::size_t r,
	std::uint64_t seed)
{
	using Real = typename tools::real_from<Number>::type;

	constexpr auto real_nan = tools::not_a_number<Real>::value;

	BOOST_TEST_CONTEXT("m=" << m) {
	BOOST_TEST_CONTEXT("n=" << n) {
	BOOST_TEST_CONTEXT("p=" << p) {
	BOOST_TEST_CONTEXT("rank=" << r) {
	BOOST_TEST_CONTEXT("seed=" << seed) {

	auto gen = std::mt19937(seed);
	auto option_dist = std::uniform_int_distribution<unsigned>(0, 2);

	gen.discard(1u << 17);

	auto option = option_dist(gen);
	auto theta_dist = ggqrcs::ThetaDistribution<Real>(option);
	auto k = std::min( {m, p, r, m + p - r} );
	auto theta = ublas::vector<Real>(k, real_nan);

	std::generate(
		theta.begin(), theta.end(),
		[&gen, &theta_dist](){ return theta_dist(gen); }
	);

	auto min_log_cond_X = Real{0};
	auto max_log_cond_X = static_cast<Real>(std::numeric_limits<Real>::digits/2);
	auto log_cond_dist =
		std::uniform_real_distribution<Real>(min_log_cond_X, max_log_cond_X);
	auto log_cond_X = log_cond_dist(gen);
	auto cond_X = std::pow(Real{2}, log_cond_X);
	auto X = tools::make_matrix_like(dummy, r, n, cond_X, &gen);
	auto U1 = tools::make_isometric_matrix_like(dummy, m, m, &gen);
	auto U2 = tools::make_isometric_matrix_like(dummy, p, p, &gen);
	auto ds = ggqrcs::assemble_diagonals_like(dummy, m, p, r, theta);
	auto D1 = ds.first;
	auto D2 = ds.second;
	auto A = ggqrcs::assemble_matrix(U1, D1, X);
	auto B = ggqrcs::assemble_matrix(U2, D2, X);

	// initialize caller
	auto lda = m + 11;
	auto ldb = p + 5;
	auto ldu1 = m + 13;
	auto ldu2 = p + 7;
	auto ldx = std::min(m + p, n) + 6;
	auto caller = ggqrcs::Caller<Number>(m, n, p, lda, ldb, ldu1, ldu2, ldx);

	ublas::subrange(caller.A, 0, m, 0, n) = A;
	ublas::subrange(caller.B, 0, p, 0, n) = B;

	auto ret = caller();

	check_results(ret, A, B, caller);

	BOOST_CHECK_LE( caller.rank, r );
}
}
}
}
}
}


BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_random, Number, types)
{
	constexpr std::size_t dimensions[] = { 1, 2, 3, 4, 10, 20 };

	auto gen = std::mt19937();
	auto seed_dist = std::uniform_int_distribution<std::uint64_t>(0);

	gen.discard(1u << 17);

	for(auto m : dimensions)
	{
		for(auto n : dimensions)
		{
			for(auto p : dimensions)
			{
				auto max_rank = std::min(m+p, n);
				for(auto rank = std::size_t{0}; rank <= max_rank; ++rank)
				{
					for(auto iteration = 0u; iteration < 10u; ++iteration)
					{
						auto seed = seed_dist(gen);

						xGGQRCS_test_random_impl(Number{0}, m, n, p, rank, seed);
					}
				}
			}
		}
	}
}

BOOST_TEST_DECORATOR(* boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(
	xGGQRCS_test_random_infinite, Number, types)
{
	constexpr auto min_dimension = std::size_t{1};
	constexpr auto max_dimension = std::size_t{500};

	auto master_seed = std::uintmax_t(std::time(nullptr));

	std::printf("xGGQRCS_test_random_infinite master_seed=%ju\n", master_seed);

	auto gen = std::mt19937(master_seed);
	auto dim_dist =
		std::uniform_int_distribution<std::size_t>(min_dimension,max_dimension);
	auto seed_dist = std::uniform_int_distribution<std::uint64_t>(0);

	gen.discard(1u << 17);

	auto second = std::time_t{1};
	auto delta_sec = 60 * second;
	auto start_time_sec = std::time(nullptr);
	auto last_time_sec = start_time_sec - delta_sec;
	auto iteration = std::uintmax_t{0};

	constexpr char FMT[] = "%7jd %13ju  %3zu %3zu %3zu %4zu  %20ju\n";
	std::printf(
		"%7s %13s  %3s %3s %3s %4s  %20s\n",
		"time(s)", "iteration", "m", "n", "p", "rank", "seed"
	);

	while(true)
	{
		auto m = dim_dist(gen);
		auto n = dim_dist(gen);
		auto p = dim_dist(gen);
		auto max_rank = (m + p <= n) ? m + p : n;

		for(auto rank = std::size_t{0}; rank <= max_rank; ++rank, ++iteration)
		{
			auto seed = seed_dist(gen);
			auto now_sec = std::time(nullptr);

			if(last_time_sec + delta_sec <= now_sec)
			{
				auto time_passed_sec = std::intmax_t{now_sec - start_time_sec};

				std::printf(
					FMT, time_passed_sec, iteration, m, n, p, rank, seed
				);
				last_time_sec += delta_sec;
			}

			xGGQRCS_test_random_impl(Number{0}, m, n, p, rank, seed);
		}
	}
}


template<typename Number>
void xGGQRCS_test_switches_impl(
	Number dummy,
	std::size_t m, std::size_t n, std::size_t p,
	std::size_t rank_A, std::size_t rank_B, std::size_t rank_G,
	char hintprepa, char hintprepb, char hintprepcols,
	typename tools::real_from<Number>::type w,
	std::uint32_t seed)
{
	BOOST_VERIFY(rank_A <= std::min(m, n));
	BOOST_VERIFY(rank_B <= std::min(p, n));
	BOOST_VERIFY(rank_G >= std::max(rank_A, rank_B));
	BOOST_VERIFY(rank_G <= rank_A + rank_B);
	BOOST_VERIFY(rank_G <= std::min(m + p, n));
	BOOST_VERIFY(hintprepa == 'Y' || hintprepa == 'N' || hintprepa == '?');
	BOOST_VERIFY(hintprepb == 'Y' || hintprepb == 'N' || hintprepb == '?');
	BOOST_VERIFY(
		hintprepcols == 'Y' || hintprepcols == 'N' || hintprepcols == '?');
	BOOST_VERIFY(std::isfinite(w));
	BOOST_VERIFY(w > 0);

	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	constexpr auto real_nan = tools::not_a_number<Real>::value;

	BOOST_TEST_CONTEXT("m=" << m) {
	BOOST_TEST_CONTEXT("n=" << n) {
	BOOST_TEST_CONTEXT("p=" << p) {
	BOOST_TEST_CONTEXT("rank(A)=" << rank_A) {
	BOOST_TEST_CONTEXT("rank(B)=" << rank_B) {
	BOOST_TEST_CONTEXT("rank(G)=" << rank_G) {
	BOOST_TEST_CONTEXT("hintprepa=" << hintprepa) {
	BOOST_TEST_CONTEXT("hintprepb=" << hintprepb) {
	BOOST_TEST_CONTEXT("hintprepcols=" << hintprepcols) {
	BOOST_TEST_CONTEXT("w=" << w) {
	BOOST_TEST_CONTEXT("seed=" << seed) {

	auto k = std::min({rank_A, rank_B, rank_A + rank_B - rank_G});
	auto k1 = rank_G - rank_B;
	auto k2 = rank_G - rank_A;

	BOOST_VERIFY(rank_G == k + k1 + k2);

	auto theta = ublas::vector<Real>(rank_G, real_nan);
	auto gen = std::mt19937(seed);
	auto theta_dist = std::uniform_real_distribution<Real>(0, M_PI/2);

	for(auto i = std::size_t{0}; i < k1; ++i) {
		theta[i] = M_PI_2;
	}
	for(auto i = k1; i < k1 + k; ++i) {
		theta[i] = std::atan(std::tan(theta_dist(gen)) / w);
	}
	for(auto i = k1 + k; i < k1 + k + k2; ++i) {
		theta[i] = 0;
	}
	// ensure singular values are in descending order
	std::sort(theta.data().begin(), theta.data().end(), std::greater<Real>());

	auto log2_cond_min = 0;
	auto log2_cond_max = std::numeric_limits<Real>::digits / 2;
	auto log2_cond_dist =
		std::uniform_int_distribution<int>(log2_cond_min,log2_cond_max);
	auto log2_cond_A = log2_cond_dist(gen);
	auto cond_A = std::ldexp(Real{1}, log2_cond_A);
	auto A = Matrix(w * tools::make_matrix_like(dummy, m, n, cond_A, &gen));
	auto log2_cond_B = log2_cond_dist(gen);
	auto cond_B = std::ldexp(Real{1}, log2_cond_B);
	auto B = tools::make_matrix_like(dummy, p, n, cond_B, &gen);

	// initialize caller
	auto lda = m + 11;
	auto ldb = p + 5;
	auto ldu1 = m + 13;
	auto ldu2 = p + 7;
	auto ldx = std::max(m + p, n) + 1;
	auto caller = ggqrcs::Caller<Number>(m, n, p, lda, ldb, ldu1, ldu2, ldx);
	ublas::subrange(caller.A, 0, m, 0, n) = A;
	ublas::subrange(caller.B, 0, p, 0, n) = B;
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;

	auto ret = caller();

	BOOST_REQUIRE_EQUAL(ret, 0);
	BOOST_REQUIRE_GE(caller.rank, rank_G);

	auto U1 = Matrix(ublas::subrange(caller.U1, 0, m, 0, m));
	auto U2 = Matrix(ublas::subrange(caller.U2, 0, p, 0, p));
	auto X = Matrix(ublas::subrange(caller.X, 0, rank_G, 0, n));
	auto ds = ggqrcs::assemble_diagonals_like(dummy, m, p, rank_G, theta);
	auto& D1 = ds.first;
	auto& D2 = ds.second;

	A = ggqrcs::assemble_matrix(U1, D1, X);
	B = ggqrcs::assemble_matrix(U2, D2, X);

	// re-run xGGQRCS
	caller = ggqrcs::Caller<Number>(m, n, p, lda, ldb, ldu1, ldu2, ldx);
	ublas::subrange(caller.A, 0, m, 0, n) = A;
	ublas::subrange(caller.B, 0, p, 0, n) = B;
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;
	//caller.compute_u1_p = jobu1;
	//caller.compute_u2_p = jobu2;
	//caller.compute_x_p = jobx;
	ret = caller();

	check_results(ret, A, B, caller);

	BOOST_CHECK_LE( caller.rank, rank_G );
	}}}}}}}}}}}
}


BOOST_AUTO_TEST_CASE_TEMPLATE(switches, Number, types)
{
	using Real = typename tools::real_from<Number>::type;

	constexpr auto num_digits_2 = std::numeric_limits<Real>::digits;

	auto gen = std::mt19937();
	auto seed_dist = std::uniform_int_distribution<std::uint64_t>(0);

	gen.discard(1u << 17);

	constexpr char PREPROCESSING_HINTS[] = { 'Y', 'N', '?' };

	auto log2_w_dist =
		std::uniform_int_distribution<int>(-num_digits_2, +num_digits_2);

	for(auto m = std::size_t{0}; m < 4; ++m) {
	for(auto p = std::size_t{0}; p < 4; ++p) {
	for(auto n = std::size_t{0}; n <= 2 * (m + p) + 1; ++n) {
	for(auto rank_A = std::size_t{0}; rank_A < std::min(m, n); ++rank_A) {
	for(auto rank_B = std::size_t{0}; rank_B < std::min(p, n); ++rank_B) {
	for(auto rank_G = std::max(rank_A, rank_B);
		rank_G <= std::min({m + p, n, rank_A + rank_B});
		++rank_G) {
	for(auto hintprepa : PREPROCESSING_HINTS) {
	for(auto hintprepb : PREPROCESSING_HINTS) {
	for(auto hintprepcols : PREPROCESSING_HINTS) {
		for(auto iteration = 0u; iteration < 1u; ++iteration)
		{
			auto seed = seed_dist(gen);
			auto log2_w = log2_w_dist(gen);
			auto w = std::ldexp(Real{1}, log2_w);

			xGGQRCS_test_switches_impl(
				Number{0},
				m, n, p, rank_A, rank_B, rank_G,
				hintprepa, hintprepb, hintprepcols,
				w,
				seed
			);
		}
	}}}}}}}}}
}


BOOST_AUTO_TEST_CASE(regression_preprocessing_20210606)
{
	using Number = float;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	auto m = std::size_t{2};
	auto n = std::size_t{7};
	auto p = std::size_t{3};
	auto A = Matrix(m, n, Number{});
	auto B = Matrix(p, n, Number{});

	A(0, 0) = -1.2146358490e+00;
	A(0, 1) = +3.7972930074e-01;
	A(0, 2) = +9.3337267637e-02;
	A(0, 3) = -4.9328170717e-02;
	A(0, 4) = +1.1591043323e-01;
	A(0, 5) = -3.5762786865e-07;
	A(0, 6) = +4.6371135861e-02;

	B(0, 0) = -3.3467432857e-01;
	B(0, 1) = -1.2347420454e+00;
	B(0, 2) = +9.5526170731e-01;
	B(0, 3) = +1.8088394403e+00;
	B(0, 4) = +1.7746002972e-01;
	B(0, 5) = -2.9758486748e+00;
	B(0, 6) = +1.5414557457e+00;
	B(1, 0) = +5.1581972837e-01;
	B(1, 1) = +2.5492978096e+00;
	B(1, 2) = -1.8518075943e+00;
	B(1, 3) = -3.2675864697e+00;
	B(1, 4) = -4.0914654732e-02;
	B(1, 5) = +4.5865530968e+00;
	B(1, 6) = -1.9628678560e+00;
	B(2, 0) = +5.6671768427e-02;
	B(2, 1) = +9.4010317326e-01;
	B(2, 2) = -5.9104621410e-01;
	B(2, 3) = -8.4892463684e-01;
	B(2, 4) = +2.3306103051e-01;
	B(2, 5) = +5.0391203165e-01;
	B(2, 6) = +2.0606298745e-01;

	auto caller = ggqrcs::Caller<Number>(m, n, p);
	ublas::subrange(caller.A, 0, m, 0, n) = A;
	ublas::subrange(caller.B, 0, p, 0, n) = B;
	caller.hint_preprocess_a = 'N';
	caller.hint_preprocess_b = 'Y';
	caller.hint_preprocess_cols = 'N';

	auto ret = caller();
	check_results(ret, A, B, caller);
}


BOOST_AUTO_TEST_CASE(regression_preprocessing_20231023)
{
	auto m = 3;
	auto n = 2;
	auto p = 2;
	//auto rank_A = 1;
	//auto rank_B = 1;
	//auto rank_G = 2;
	auto hintprepa = 'Y';
	auto hintprepb = 'Y';
	auto hintprepcols = 'N';
	auto caller = ggqrcs::Caller<float>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = -6.6977783203e+02;
	A(0, 1) = -1.7713051758e+03;
	A(1, 0) = +3.3091577148e+02;
	A(1, 1) = +8.7514514160e+02;
	A(2, 0) = +1.8713862610e+02;
	A(2, 1) = +4.9490979004e+02;
	B(0, 0) = +2.9615115625e+05;
	B(0, 1) = +1.4119034375e+05;
	B(1, 0) = -8.9898775000e+05;
	B(1, 1) = -4.2859328125e+05;

	caller.A = A;
	caller.B = B;
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;

	auto ret = caller();

	BOOST_REQUIRE_EQUAL(caller.hint_preprocess_a, 'Y');
	BOOST_REQUIRE_EQUAL(caller.hint_preprocess_b, 'Y');
	BOOST_REQUIRE_EQUAL(caller.hint_preprocess_cols, 'N');

	check_results(ret, A, B, caller);
}


BOOST_AUTO_TEST_CASE(regression_switches_20231107)
{
	auto dummy = float{0};
	auto m = 3;
	auto n = 3;
	auto p = 3;
	auto rank_A = 1;
	auto rank_B = 2;
	auto rank_G = 3;
	auto hintprepa = 'N';
	auto hintprepb = 'Y';
	auto hintprepcols = 'N';
	auto w = std::ldexp(float{2}, -20);
	auto seed = UINT32_C(2538164553);

	xGGQRCS_test_switches_impl(
			dummy, m, n, p, rank_A, rank_B, rank_G,
			hintprepa, hintprepb,
			hintprepcols, w, seed);
}


// U1 contains NaNs
BOOST_AUTO_TEST_CASE(regression_switches_20231108)
{
	auto m = 3;
	auto n = 6;
	auto p = 3;
	//auto rank_A = 1;
	//auto rank_B = 1;
	//auto rank_G = 2;
	auto hintprepa = 'N';
	auto hintprepb = 'Y';
	auto hintprepcols = 'N';
	//auto w = std::ldexp(float{1}, 9);
	//auto seed = UINT32_C(2951424079);

	auto caller = ggqrcs::Caller<float>(m, n, p);
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) =  0.00000000; A(0, 1) =  0.00000000; A(0, 2) = 0.00000000; A(0, 3) = 0.00000000; A(0, 4) =  0.00000000;     A(0, 5) =  0.00000000;
	A(1, 0) = -2084.21094; A(1, 1) = -470.370300; A(1, 2) = 782.985962; A(1, 3) = 977.149475; A(1, 4) = -2886.45068;     A(1, 5) =  581.008789;
	A(2, 0) =  0.00000000; A(2, 1) =  0.00000000; A(2, 2) = 0.00000000; A(2, 3) = 0.00000000; A(2, 4) =  0.00000000;     A(2, 5) =  0.00000000;
	B(0, 0) =  55.0295334; B(0, 1) =  98.2178268; B(0, 2) =-73.6211243; B(0, 3) = 68.5249176; B(0, 4) = -1.40125339E-04; B(0, 5) =  73.0102692;
	B(1, 0) = -353.966553; B(1, 1) = -631.765686; B(1, 2) = 473.552643; B(1, 3) =-440.771881; B(1, 4) =  8.09960547E-05; B(1, 5) = -469.623108;
	B(2, 0) =  1165.85156; B(2, 1) =  2080.83228; B(2, 2) =-1559.72949; B(2, 3) = 1451.76025; B(2, 4) = -3.09503201E-04; B(2, 5) =  1546.78687;

	caller.A = A;
	caller.B = B;

	auto ret = caller();
	check_results(ret, A, B, caller);

}


// U2 contains infinities
BOOST_AUTO_TEST_CASE(regression_switches_20231109)
{
	auto m = 2;
	auto n = 10;
	auto p = 3;
	//auto rank_A = 1;
	//auto rank_B = 2;
	//auto rank_G = 3;
	auto hintprepa = 'N';
	auto hintprepb = 'N';
	auto hintprepcols = '?';
	//auto w = std::ldexp(float{1}, 18);
	//auto seed = UINT32_C(4982116);

	auto caller = ggqrcs::Caller<float>(m, n, p);
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = -5.7609828949e+01;
	A(0, 1) = +7.2083221436e+01;
	A(0, 2) = +1.8676923752e+01;
	A(0, 3) = -4.4711860657e+01;
	A(0, 4) = +3.9915802002e+01;
	A(0, 5) = +8.5821571350e+00;
	A(0, 6) = -5.3112968445e+01;
	A(0, 7) = +1.1898607254e+01;
	A(0, 8) = -8.2098678589e+01;
	A(0, 9) = -1.1276281357e+01;
	A(1, 0) = +0.0000000000e+00;
	A(1, 1) = +0.0000000000e+00;
	A(1, 2) = +0.0000000000e+00;
	A(1, 3) = +0.0000000000e+00;
	A(1, 4) = +0.0000000000e+00;
	A(1, 5) = +0.0000000000e+00;
	A(1, 6) = +0.0000000000e+00;
	A(1, 7) = +0.0000000000e+00;
	A(1, 8) = +0.0000000000e+00;
	A(1, 9) = +0.0000000000e+00;
	B(0, 0) = -8.1019386292e+01;
	B(0, 1) = +4.7377765656e+01;
	B(0, 2) = +4.8631958008e+01;
	B(0, 3) = -4.1563831329e+01;
	B(0, 4) = +3.0281486511e+00;
	B(0, 5) = -2.2143983841e+01;
	B(0, 6) = -1.9816329956e+01;
	B(0, 7) = -1.5569763184e+00;
	B(0, 8) = -6.3010536194e+01;
	B(0, 9) = -1.2416633606e+01;
	B(1, 0) = +2.5182055197e-06;
	B(1, 1) = -3.1508577649e-06;
	B(1, 2) = -8.1639427663e-07;
	B(1, 3) = +1.9544174847e-06;
	B(1, 4) = -1.7447750906e-06;
	B(1, 5) = -3.7513800066e-07;
	B(1, 6) = +2.3216416594e-06;
	B(1, 7) = -5.2010466334e-07;
	B(1, 8) = +3.5886471323e-06;
	B(1, 9) = +4.9290190418e-07;
	B(2, 0) = +1.0960916138e+02;
	B(2, 1) = -5.6320083618e+01;
	B(2, 2) = -6.8430595398e+01;
	B(2, 3) = +5.3797996521e+01;
	B(2, 4) = +3.1055412292e+00;
	B(2, 5) = +3.4340988159e+01;
	B(2, 6) = +1.9633499146e+01;
	B(2, 7) = +3.6087913513e+00;
	B(2, 8) = +7.6907791138e+01;
	B(2, 9) = +1.7066524506e+01;

	caller.A = A;
	caller.B = B;

	auto ret = caller();
	check_results(ret, A, B, caller);
}


// X contains NaN
BOOST_AUTO_TEST_CASE(regression_switches_20231110)
{
	auto m = 2;
	auto n = 10;
	auto p = 3;
	//auto rank_A = 1;
	//auto rank_B = 2;
	//auto rank_G = 3;
	auto hintprepa = 'N';
	auto hintprepb = 'N';
	auto hintprepcols = '?';
	//auto w = std::ldexp(float{1}, -5);
	//auto seed = UINT32_C(3339546974);

	auto caller = ggqrcs::Caller<float>(m, n, p);
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = +1.4323523641e-01;
	A(0, 1) = +1.6907602549e-01;
	A(0, 2) = -3.2808166742e-01;
	A(0, 3) = +3.8594973087e-01;
	A(0, 4) = +9.0967498720e-02;
	A(0, 5) = +5.8918941021e-01;
	A(0, 6) = -1.5344560146e-01;
	A(0, 7) = -4.2795753479e-01;
	A(0, 8) = +2.6862102747e-01;
	A(0, 9) = -2.4501556158e-01;
	A(1, 0) = +0.0000000000e+00;
	A(1, 1) = +0.0000000000e+00;
	A(1, 2) = +0.0000000000e+00;
	A(1, 3) = +0.0000000000e+00;
	A(1, 4) = +0.0000000000e+00;
	A(1, 5) = +0.0000000000e+00;
	A(1, 6) = +0.0000000000e+00;
	A(1, 7) = +0.0000000000e+00;
	A(1, 8) = +0.0000000000e+00;
	A(1, 9) = +0.0000000000e+00;
	B(0, 0) = +6.3546645641e-01;
	B(0, 1) = -2.9088684916e-01;
	B(0, 2) = -1.2959568202e-01;
	B(0, 3) = -3.1598061323e-03;
	B(0, 4) = -2.4752910435e-01;
	B(0, 5) = -4.2362302542e-01;
	B(0, 6) = -3.8152527809e-01;
	B(0, 7) = -2.6240170002e-02;
	B(0, 8) = +2.5383472443e-01;
	B(0, 9) = -2.0821258426e-01;
	B(1, 0) = -6.2610108209e-09;
	B(1, 1) = -7.3905477294e-09;
	B(1, 2) = +1.4340905352e-08;
	B(1, 3) = -1.6870398412e-08;
	B(1, 4) = -3.9763157211e-09;
	B(1, 5) = -2.5754287591e-08;
	B(1, 6) = +6.7073204768e-09;
	B(1, 7) = +1.8706618476e-08;
	B(1, 8) = -1.1741797756e-08;
	B(1, 9) = +1.0709970688e-08;
	B(2, 0) = +3.5989657044e-02;
	B(2, 1) = +2.3119910061e-01;
	B(2, 2) = -4.5487210155e-01;
	B(2, 3) = -7.0289385319e-01;
	B(2, 4) = -2.3527285457e-01;
	B(2, 5) = +2.4428044260e-01;
	B(2, 6) = -2.0114529133e-01;
	B(2, 7) = +4.3311081827e-02;
	B(2, 8) = +5.2265182137e-02;
	B(2, 9) = +2.9015627503e-01;

	caller.A = A;
	caller.B = B;

	auto ret = caller();
	check_results(ret, A, B, caller);
}


// X contains NaN
BOOST_AUTO_TEST_CASE(regression_switches_20231111)
{
	auto m = 3;
	auto n = 14;
	auto p = 4;
	//auto rank_A = 1;
	//auto rank_B = 3;
	//auto rank_G = 4;
	auto hintprepa = 'N';
	auto hintprepb = '?';
	auto hintprepcols = '?';
	//auto w = std::ldexp(float{1}, -2);
	//auto seed = UINT32_C(1569762326);

	auto caller = ggqrcs::Caller<float>(m, n, p);
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = -2.0871433616e-01;
	A(0, 1) = +1.3997806609e-01;
	A(0, 2) = +3.3748316765e-01;
	A(0, 3) = -1.5584519506e-01;
	A(0, 4) = -3.7984651327e-01;
	A(0, 5) = -2.9502776265e-01;
	A(0, 6) = -1.9834339619e-01;
	A(0, 7) = -6.1293885112e-02;
	A(0, 8) = +4.0822848678e-01;
	A(0, 9) = +2.0716214180e-01;
	A(0, 10) = -2.8744748235e-01;
	A(0, 11) = +1.5408289433e-01;
	A(0, 12) = +4.1203722358e-01;
	A(0, 13) = +1.9629414380e-01;
	A(1, 0) = +0.0000000000e+00;
	A(1, 1) = +0.0000000000e+00;
	A(1, 2) = +0.0000000000e+00;
	A(1, 3) = +0.0000000000e+00;
	A(1, 4) = +0.0000000000e+00;
	A(1, 5) = +0.0000000000e+00;
	A(1, 6) = +0.0000000000e+00;
	A(1, 7) = +0.0000000000e+00;
	A(1, 8) = +0.0000000000e+00;
	A(1, 9) = +0.0000000000e+00;
	A(1, 10) = +0.0000000000e+00;
	A(1, 11) = +0.0000000000e+00;
	A(1, 12) = +0.0000000000e+00;
	A(1, 13) = +0.0000000000e+00;
	A(2, 0) = +0.0000000000e+00;
	A(2, 1) = +0.0000000000e+00;
	A(2, 2) = +0.0000000000e+00;
	A(2, 3) = +0.0000000000e+00;
	A(2, 4) = +0.0000000000e+00;
	A(2, 5) = +0.0000000000e+00;
	A(2, 6) = +0.0000000000e+00;
	A(2, 7) = +0.0000000000e+00;
	A(2, 8) = +0.0000000000e+00;
	A(2, 9) = +0.0000000000e+00;
	A(2, 10) = +0.0000000000e+00;
	A(2, 11) = +0.0000000000e+00;
	A(2, 12) = +0.0000000000e+00;
	A(2, 13) = +0.0000000000e+00;
	B(0, 0) = +3.4225958586e-01;
	B(0, 1) = -1.3295821846e-01;
	B(0, 2) = +6.5325424075e-02;
	B(0, 3) = +4.5359200239e-01;
	B(0, 4) = -7.8452304006e-02;
	B(0, 5) = +5.5642914027e-02;
	B(0, 6) = -4.6214762330e-01;
	B(0, 7) = -4.3300792575e-01;
	B(0, 8) = +1.0629543662e-01;
	B(0, 9) = -3.2566794753e-01;
	B(0, 10) = +2.1467489004e-01;
	B(0, 11) = +1.1327171326e-01;
	B(0, 12) = +2.2741633654e-01;
	B(0, 13) = -9.3098558486e-02;
	B(1, 0) = +3.6984127015e-02;
	B(1, 1) = +5.0585925579e-01;
	B(1, 2) = -6.7112646997e-02;
	B(1, 3) = -4.2712982744e-02;
	B(1, 4) = -1.3900932670e-01;
	B(1, 5) = -5.2921015024e-01;
	B(1, 6) = +2.0126114786e-01;
	B(1, 7) = -6.4844511449e-02;
	B(1, 8) = -5.4529726505e-02;
	B(1, 9) = -1.8819454312e-01;
	B(1, 10) = +4.9933913350e-01;
	B(1, 11) = +2.8953820467e-01;
	B(1, 12) = -1.4255237579e-01;
	B(1, 13) = -6.0195233673e-03;
	B(2, 0) = +6.8978257477e-02;
	B(2, 1) = -4.7068063170e-02;
	B(2, 2) = -2.5875438005e-02;
	B(2, 3) = +6.8423643708e-02;
	B(2, 4) = -9.5245420933e-02;
	B(2, 5) = +1.1003648862e-02;
	B(2, 6) = -6.8002492189e-03;
	B(2, 7) = -2.1869637072e-02;
	B(2, 8) = +2.9971696436e-02;
	B(2, 9) = -1.2855750509e-02;
	B(2, 10) = -3.7344492972e-02;
	B(2, 11) = +3.8631390780e-02;
	B(2, 12) = -1.1613555253e-02;
	B(2, 13) = -8.5148446262e-02;
	B(3, 0) = -2.3869907856e-01;
	B(3, 1) = +1.5380549431e-01;
	B(3, 2) = +2.0749506354e-01;
	B(3, 3) = -1.6126707196e-01;
	B(3, 4) = +5.6702405214e-01;
	B(3, 5) = +2.9868286103e-02;
	B(3, 6) = -2.4829450250e-01;
	B(3, 7) = -1.0399539769e-01;
	B(3, 8) = -1.1919376999e-01;
	B(3, 9) = -8.3095334470e-02;
	B(3, 10) = +2.9357478023e-01;
	B(3, 11) = -2.1254947782e-01;
	B(3, 12) = +2.2104801238e-01;
	B(3, 13) = +4.7884294391e-01;

	caller.A = A;
	caller.B = B;

	auto ret = caller();
	check_results(ret, A, B, caller);
}


// U2 is inaccurate
BOOST_AUTO_TEST_CASE(regression_switches_20231113)
{
	auto m = 2;
	auto n = 11;
	auto p = 3;
	auto rank_A = 1;
	auto rank_B = 2;
	auto rank_G = 3;
	auto hintprepa = 'N';
	auto hintprepb = 'N';
	auto hintprepcols = 'Y';
	auto w = std::ldexp(float{1}, 23);
	auto seed = UINT32_C(855203784);

	xGGQRCS_test_switches_impl(
			float{0}, m, n, p, rank_A, rank_B, rank_G, hintprepa, hintprepb, hintprepcols, w, seed);
}


// X contains NaN
BOOST_AUTO_TEST_CASE(regression_switches_20231117)
{
	auto m = 4;
	auto n = 5;
	auto p = 4;
	//auto rank_A = 3;
	//auto rank_B = 3;
	//auto rank_G = 5;
	auto hintprepa = 'Y';
	auto hintprepb = 'N';
	auto hintprepcols = 'N';
	//auto w = std::ldexp(float{1}, 17);
	//auto seed = UINT32_C(3464537296);

	auto caller = ggqrcs::Caller<float>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = -2.523180200e+07;
	A(0, 1) = +1.762831200e+07;
	A(0, 2) = -6.935665000e+06;
	A(0, 3) = +1.278419600e+07;
	A(0, 4) = +7.442814000e+06;
	A(1, 0) = -1.920080800e+07;
	A(1, 1) = +1.678079800e+07;
	A(1, 2) = -3.421343500e+06;
	A(1, 3) = +8.902876000e+06;
	A(1, 4) = +1.717498500e+06;
	A(2, 0) = -1.230522000e+06;
	A(2, 1) = +8.881116000e+06;
	A(2, 2) = +4.323080000e+06;
	A(2, 3) = -8.483245000e+05;
	A(2, 4) = -8.723171000e+06;
	A(3, 0) = +3.979254000e+06;
	A(3, 1) = -7.983786000e+06;
	A(3, 2) = -1.871042500e+06;
	A(3, 3) = -9.380397500e+05;
	A(3, 4) = +4.800332500e+06;
	B(0, 0) = +1.017645000e+07;
	B(0, 1) = +3.476455750e+06;
	B(0, 2) = +5.499623500e+06;
	B(0, 3) = -1.525317400e+07;
	B(0, 4) = -1.082843900e+07;
	B(1, 0) = +8.011083000e+06;
	B(1, 1) = +2.736736750e+06;
	B(1, 2) = +4.329398500e+06;
	B(1, 3) = -1.200757200e+07;
	B(1, 4) = -8.524348000e+06;
	B(2, 0) = +2.241878800e+07;
	B(2, 1) = +7.658647000e+06;
	B(2, 2) = +1.211571100e+07;
	B(2, 3) = -3.360284000e+07;
	B(2, 4) = -2.385511600e+07;
	B(3, 0) = +7.864986000e+06;
	B(3, 1) = +2.686812000e+06;
	B(3, 2) = +4.250449500e+06;
	B(3, 3) = -1.178858900e+07;
	B(3, 4) = -8.368877000e+06;

	caller.A = A;
	caller.B = B;
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;

	auto ret = caller();
	check_results(ret, A, B, caller);
}


// Problem: X contains NaN.
// This test uncovered two problems related to column pre-processing followed
// by pre-processing of A, B, or both matrices:
// * The numerical rank computed by the column pre-processing may be larger
//   than the number of rows in A and B combined after pre-processing them.
//   This was not handled correctly and caused SORGQR to abort because the
//   number of elementary reflectors was too large for the given matrix
//   dimensions.
// * The column pre-processing code was mixing up the matrix dimensions M and P
//   with the row ranks ROWSA and ROWSB, respectively.
BOOST_AUTO_TEST_CASE(regression_switches_20231124)
{
	auto m = 2;
	auto n = 14;
	auto p = 5;
	//auto rank_A = 1;
	//auto rank_B = 4;
	//auto rank_G = 5;
	auto hintprepa = 'Y';
	auto hintprepb = '?';
	auto hintprepcols = '?';
	//auto w = std::ldexp(float{1}, -17);
	//auto seed = UINT32_C(2493622919);
	auto caller = ggqrcs::Caller<float>(m, n, p);
	auto A = caller.A;
	auto B = caller.B;

	A(0, 0) = +1.410665321e+01;
	A(0, 1) = -1.911086202e+00;
	A(0, 2) = -1.272614479e+01;
	A(0, 3) = +6.018759727e+00;
	A(0, 4) = -1.010638714e+01;
	A(0, 5) = +7.009910583e+00;
	A(0, 6) = +8.174458504e+00;
	A(0, 7) = +3.823521137e+00;
	A(0, 8) = +9.947698593e+00;
	A(0, 9) = -7.558737278e+00;
	A(0, 10) = +5.945141792e+00;
	A(0, 11) = +1.320609283e+01;
	A(0, 12) = +5.964776039e+00;
	A(0, 13) = +8.835703850e+00;
	A(1, 0) = +9.461766243e+00;
	A(1, 1) = -1.281824350e+00;
	A(1, 2) = -8.535817146e+00;
	A(1, 3) = +4.036967754e+00;
	A(1, 4) = -6.778665066e+00;
	A(1, 5) = +4.701762676e+00;
	A(1, 6) = +5.482861042e+00;
	A(1, 7) = +2.564553499e+00;
	A(1, 8) = +6.672227859e+00;
	A(1, 9) = -5.069878101e+00;
	A(1, 10) = +3.987589598e+00;
	A(1, 11) = +8.857732773e+00;
	A(1, 12) = +4.000759125e+00;
	A(1, 13) = +5.926378727e+00;
	B(0, 0) = -2.354245663e+00;
	B(0, 1) = -1.274588227e+00;
	B(0, 2) = -1.022671223e+00;
	B(0, 3) = +3.332304239e+00;
	B(0, 4) = -1.049327612e+00;
	B(0, 5) = +6.667078733e-01;
	B(0, 6) = -2.975541115e+00;
	B(0, 7) = +5.985876083e+00;
	B(0, 8) = +2.940188408e+00;
	B(0, 9) = +1.365854263e+00;
	B(0, 10) = -1.094935179e+00;
	B(0, 11) = +2.838111401e+00;
	B(0, 12) = -4.488838673e+00;
	B(0, 13) = -1.741097569e+00;
	B(1, 0) = -1.748939514e+01;
	B(1, 1) = -1.145000935e+00;
	B(1, 2) = -1.172084427e+01;
	B(1, 3) = +1.768215561e+01;
	B(1, 4) = -5.721320629e+00;
	B(1, 5) = -3.447908878e+00;
	B(1, 6) = -1.173703575e+01;
	B(1, 7) = +2.050516891e+01;
	B(1, 8) = +1.636666298e+01;
	B(1, 9) = +4.553931236e+00;
	B(1, 10) = +2.183465481e+00;
	B(1, 11) = +1.269914436e+01;
	B(1, 12) = -1.865117836e+01;
	B(1, 13) = -9.262999535e+00;
	B(2, 0) = -5.253398895e+00;
	B(2, 1) = +7.169449806e+00;
	B(2, 2) = -7.242753983e+00;
	B(2, 3) = -1.073216319e+00;
	B(2, 4) = +1.535561681e-01;
	B(2, 5) = -8.618588448e+00;
	B(2, 6) = +5.833065033e+00;
	B(2, 7) = -1.547105980e+01;
	B(2, 8) = -2.641206980e-02;
	B(2, 9) = -3.680519581e+00;
	B(2, 10) = +9.968990326e+00;
	B(2, 11) = -3.754071236e+00;
	B(2, 12) = +7.662865162e+00;
	B(2, 13) = +5.316901803e-01;
	B(3, 0) = +2.875819921e+00;
	B(3, 1) = -8.103443384e-01;
	B(3, 2) = +2.421998501e+00;
	B(3, 3) = -2.058942080e+00;
	B(3, 4) = +6.919430494e-01;
	B(3, 5) = +1.574810505e+00;
	B(3, 6) = +6.860893965e-01;
	B(3, 7) = -4.968056679e-01;
	B(3, 8) = -2.034287691e+00;
	B(3, 9) = -7.783091068e-02;
	B(3, 10) = -1.596852064e+00;
	B(3, 11) = -1.082203865e+00;
	B(3, 12) = +1.303767562e+00;
	B(3, 13) = +1.082668185e+00;
	B(4, 0) = +2.451035690e+01;
	B(4, 1) = -8.808799744e+00;
	B(4, 2) = +2.158485031e+01;
	B(4, 3) = -1.593167973e+01;
	B(4, 4) = +5.423389435e+00;
	B(4, 5) = +1.534188271e+01;
	B(4, 6) = +3.478024006e+00;
	B(4, 7) = +1.242296219e+00;
	B(4, 8) = -1.608663750e+01;
	B(4, 9) = +6.148226261e-01;
	B(4, 10) = -1.596782017e+01;
	B(4, 11) = -7.307264328e+00;
	B(4, 12) = +7.753322601e+00;
	B(4, 13) = +8.388458252e+00;

	caller.A = A;
	caller.B = B;
	caller.hint_preprocess_a = hintprepa;
	caller.hint_preprocess_b = hintprepb;
	caller.hint_preprocess_cols = hintprepcols;

	auto ret = caller();
	check_results(ret, A, B, caller);
}

// expect failures because xLANGE overflows when it should not
BOOST_TEST_DECORATOR(* boost::unit_test::expected_failures(3))
BOOST_AUTO_TEST_CASE_TEMPLATE(
	overflow_checks, Number, lapack::supported_real_types)
{
	using Integer = lapack::integer_t;
	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;
	using ZeroMatrix = ublas::zero_matrix<Number>;

	// "big M" from linear programming: a large, positive value
	constexpr auto M = std::numeric_limits<Real>::max();
	constexpr auto nan = tools::not_a_number<Number>::value;
	constexpr auto real_nan = tools::not_a_number<Real>::value;

	auto jobu1 = 'N';
	auto jobu2 = 'Y';
	auto jobx = 'N';
	auto hint_preprocess_a = '?';
	auto hint_preprocess_b = '?';
	auto hint_preprocess_cols = '?';
	auto m = Integer{1};
	auto n = Integer{2};
	auto p = Integer{3};
	auto rank_max = std::min(m + p, n);
	auto rank = Integer{-1};
	auto swapped_p = false;
	auto lda = std::max(m, Integer{1});
	auto a = Matrix(lda, n);
	auto zero_a = ZeroMatrix(lda, n);
	auto ldb = std::max(p, Integer{1});
	auto b = Matrix(ldb, n);
	auto zero_b = ZeroMatrix(ldb, n);
	auto alpha = std::vector<Real>(rank_max, real_nan);
	auto beta = std::vector<Real>(rank_max, real_nan);
	auto ldu1 = std::max(m, Integer{1});
	auto u1 = Matrix(ldu1, m);
	auto ldu2 = std::max(p, Integer{1});
	auto u2 = Matrix(ldu2, p);
	auto ldx = std::max(rank_max, Integer{1});
	auto x = Matrix(ldx, n);
	auto tol = Real{-1};
	auto lwork = std::max(10 * rank_max, Integer{128});
	auto work = std::vector<Number>(lwork, nan);
	auto iwork = std::vector<Integer>(m + n + p, -1);
	auto call = [&] () -> Integer {
		return lapack::xGGQRCS(
			jobu1, jobu2, jobx,
			&hint_preprocess_a, &hint_preprocess_b, &hint_preprocess_cols,
			m, n, p, &rank, &swapped_p,
			&a(0, 0), lda, &b(0, 0), ldb,
			&alpha[0], &beta[0],
			&u1(0, 0), ldu1, &u2(0, 0), ldu2, &x(0, 0), ldx,
			&tol,
			&work[0], lwork, &iwork[0]);
	};

	a = zero_a;
	b = zero_b;
	BOOST_CHECK_EQUAL( call(), 0 );

	a = zero_a;
	b = zero_b;
	assert(n >= 2); // self-reminder
	a(0, 0) = M;
	a(0, 1) = M;
	b(0, 0) = 1;
	BOOST_CHECK_EQUAL( call(), 101 );

	a = zero_a;
	b = zero_b;
	a(0, 0) = 1;
	b(0, 0) = M;
	b(0, 1) = M;
	BOOST_CHECK_EQUAL( call(), 102 );

	a = zero_a;
	b = zero_b;
	a(0, 0) = M;
	b(0, 0) = M;
	// check fails if xLANGE overflows
	BOOST_CHECK_EQUAL( call(), 103 );

	a = zero_a;
	b = zero_b;
	a(0, 0) = std::numeric_limits<Real>::min();
	b(0, 0) = M;
	// check fails if xLANGE overflows
	BOOST_CHECK_EQUAL( call(), 104 );

	a = zero_a;
	b = zero_b;
	a(0, 0) = M;
	b(0, 0) = std::numeric_limits<Real>::min();
	// check fails if xLANGE overflows
	BOOST_CHECK_EQUAL( call(), 104 );
}


BOOST_TEST_DECORATOR(* boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(xGGQRCS_test_xGGSVD3_comparison, Number, types)
{
	using Real = typename tools::real_from<Number>::type;
	using Matrix = ublas::matrix<Number, ublas::column_major>;

	auto master_seed = std::uintmax_t(std::time(nullptr));

	std::printf(
		"xGGQRCS_test_xGGSVD3_comparison master_seed=%ju\n",
		master_seed
	);

	auto gen = std::mt19937(master_seed);
	auto option_dist = std::uniform_int_distribution<unsigned>(0, 1);

	gen.discard(1u << 17);

	std::printf(
		"%3s %3s %3s %4s  %17s  %17s\n",
		"m", "n", "p", "rank", "median-rel-error", "max-rel-error"
	);

	for(auto dim = std::size_t{10}; dim <= 50; dim += 10)
	{
		// make odd for easy median computation
		auto num_iterations = std::size_t{1001};
		auto m = dim;
		auto n = dim;
		auto p = dim;
		auto r = std::min( m+p, n );

		BOOST_TEST_CONTEXT("m=" << m) {
		BOOST_TEST_CONTEXT("n=" << n) {
		BOOST_TEST_CONTEXT("p=" << p) {
		BOOST_TEST_CONTEXT("rank=" << r) {

		auto nan = tools::not_a_number<Number>::value;
		auto real_nan = tools::not_a_number<Real>::value;
		auto eps = std::numeric_limits<Real>::epsilon();
		auto dummy = nan;
		auto A = Matrix(1, 1, nan);
		auto B = Matrix(1, 1, nan);
		auto norm_A = real_nan;
		auto norm_B = real_nan;
		auto delta_A_qrcs = ublas::vector<Real>(num_iterations, real_nan);
		auto delta_B_qrcs = ublas::vector<Real>(num_iterations, real_nan);
		auto delta_cs_qrcs = ublas::vector<Real>(num_iterations, Real{0});
		auto delta_A_svd3 = ublas::vector<Real>(num_iterations, real_nan);
		auto delta_B_svd3 = ublas::vector<Real>(num_iterations, real_nan);
		auto delta_cs_svd3 = ublas::vector<Real>(num_iterations, Real{0});

		for(auto it = std::size_t{0}; it < num_iterations; ++it)
		{
			// set up matrices
			{
				auto k = std::min( {m, p, r, m + p - r} );
				auto theta_dist =
					std::uniform_real_distribution<Real>(0, M_PI/2);
				auto theta = ublas::vector<Real>(k, real_nan);

				std::generate(
					theta.begin(), theta.end(),
					[&gen, &theta_dist](){ return theta_dist(gen); }
				);

				auto max_log_cond_R =
					static_cast<Real>(std::numeric_limits<Real>::digits/4);
				auto cond_R = std::pow(Real{2}, max_log_cond_R);
				auto R_Qt = tools::make_matrix_like(dummy, r, n, cond_R, &gen);
				auto U1 = tools::make_isometric_matrix_like(dummy, m, m, &gen);
				auto U2 = tools::make_isometric_matrix_like(dummy, p, p, &gen);
				auto ds = ggqrcs::assemble_diagonals_like(dummy, m, p, r, theta);
				auto D1 = ds.first;
				auto D2 = ds.second;
				auto option = option_dist(gen);
				auto d = std::numeric_limits<Real>::digits - 1;
				auto w =
					(option == 0) ? std::ldexp(Real{1}, +d/2) :
					(option == 1) ? std::ldexp(Real{1}, -d/2) : real_nan
				;

				A = ggqrcs::assemble_matrix(U1, D1, R_Qt);
				B = w * ggqrcs::assemble_matrix(U2, D2, R_Qt);

				norm_A = ublas::norm_frobenius(A);
				norm_B = ublas::norm_frobenius(B);
			}

			{
				auto qrcs = ggqrcs::Caller<Number>(m, n, p);

				qrcs.A = A; qrcs.B = B;

				auto ret = qrcs();

				BOOST_VERIFY(ret == 0);

				auto X = copy_X(qrcs);
				auto ds = ggqrcs::assemble_diagonals_like(
					dummy, m, p, qrcs.rank, qrcs.swapped_p, qrcs.alpha, qrcs.beta
				);
				auto& D1 = ds.first;
				auto& D2 = ds.second;
				auto almost_A = ggqrcs::assemble_matrix(qrcs.U1, D1, X);
				auto almost_B = ggqrcs::assemble_matrix(qrcs.U2, D2, X);

				delta_A_qrcs[it] =
					ublas::norm_frobenius(A-almost_A) / (eps * norm_A);
				delta_B_qrcs[it] =
					ublas::norm_frobenius(B-almost_B) / (eps * norm_B);
			}

			{
				auto svd3 = ggsvd3::Caller<Number>(p, n, m);

				svd3.A = B; svd3.B = A;

				auto ret = svd3();

				BOOST_VERIFY(  ret == 0 );

				auto R = ggsvd3::assemble_R(svd3.k, svd3.l, svd3.A, svd3.B);
				auto ds = ggsvd3::assemble_diagonals_like(
					Number{}, p, m, svd3.k, svd3.l, svd3.alpha, svd3.beta
				);
				auto& D1 = ds.first;
				auto& D2 = ds.second;
				auto Qt = Matrix(ublas::herm(svd3.Q));
				auto almost_A = ggqrcs::assemble_matrix(svd3.U2, D2, R, Qt);
				auto almost_B = ggqrcs::assemble_matrix(svd3.U1, D1, R, Qt);

				delta_A_svd3[it] =
					ublas::norm_frobenius(A-almost_A) / (eps * norm_A);
				delta_B_svd3[it] =
					ublas::norm_frobenius(B-almost_B) / (eps * norm_B);
			}
		}

		std::sort(delta_A_qrcs.begin(), delta_A_qrcs.end());
		std::sort(delta_B_qrcs.begin(), delta_B_qrcs.end());
		std::sort(delta_A_svd3.begin(), delta_A_svd3.end());
		std::sort(delta_B_svd3.begin(), delta_B_svd3.end());

		auto k = num_iterations - 1;

		std::printf(
			"%3zu %3zu %3zu %4zu  %8.2e %8.2e  %8.2e %8.2e\n",
			m, n, p, r,
			delta_A_qrcs[k/2] / delta_A_svd3[k/2],
			delta_B_qrcs[k/2] / delta_B_svd3[k/2],
			delta_A_qrcs[k] / delta_A_svd3[k],
			delta_B_qrcs[k] / delta_B_svd3[k]
		);
	}
	}
	}
	}
	}
}
