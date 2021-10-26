/*
 * Copyright (c) 2021 Christoph Conrads
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
#include "tools.hpp"

#include <cstddef>

#include <limits>
#include <vector>

#include <boost/assert.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/test/unit_test.hpp>

using Integer = lapack::integer_t;

namespace ublas = boost::numeric::ublas;
namespace tools = lapack::tools;


// test if xUNBDB6 respects vector increments
template<
    typename Number,
    typename std::enable_if<std::is_fundamental<Number>::value, int>::type* =
        nullptr>
void vector_increments_impl(Number) {
    using Matrix = ublas::matrix<Number, ublas::column_major>;

    auto m1 = std::size_t{2};
    auto n = std::size_t{4};
    auto m2 = std::size_t{3};
    auto ldq = m1 + m2;
    auto Q = Matrix(ublas::identity_matrix<Number>(m1 + m2, n));

    auto increments = std::vector<int>{+1, +2, +5};
    for(auto incx1 : increments) {
        for(auto incx2 : increments) {
            BOOST_REQUIRE(incx1 != 0);
            BOOST_REQUIRE(incx2 != 0);

            constexpr auto nan = tools::not_a_number<Number>::value;

            auto ldx1 = m1 * std::abs(incx1);
            auto x1 = ublas::vector<Number>(ldx1, nan);
            for(auto i = std::size_t{0}; i < m1; ++i) {
                x1(i * std::abs(incx1)) = 1;
            }

            auto ldx2 = m2 * std::abs(incx2);
            auto x2 = ublas::vector<Number>(ldx2, nan);
            for(auto i = std::size_t{0}; i < m2; ++i) {
                x2(i * std::abs(incx2)) = 1;
            }

            auto work = ublas::vector<Number>(n, nan);
            auto info = lapack::xUNBDB6(
                m1, m2, n, &x1(0), incx1, &x2(0), incx2, &Q(0, 0), ldq,
                &Q(m1, 0), ldq, &work(0), work.size());

            BOOST_REQUIRE_EQUAL(info, 0);
            for(auto i = std::size_t{0}; i < m1; ++i) {
                std::printf("%zu %16.10e\n", i, x1(i * std::abs(incx1)));
                BOOST_CHECK(tools::finite_p(x1(i * std::abs(incx1))));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    vector_increments, Number, lapack::supported_real_types) {
    vector_increments_impl(Number{});
}


// test if xUNBDB6 returns a zero vector if the vector lies in the
// column span of the matrix
template<
    typename Number,
    typename std::enable_if<std::is_fundamental<Number>::value, int>::type* =
        nullptr>
void require_zero_output_impl(Number) {
    using Real = typename tools::real_from<Number>::type;
    using Matrix = ublas::matrix<Number, ublas::column_major>;

    auto m = std::size_t{1};
    auto n = std::size_t{1};
    auto p = std::size_t{1};
    auto ldq = m + p;

    for(auto i = std::size_t{0}; i < m + p; ++i) {
        constexpr auto mu = std::numeric_limits<Real>::denorm_min();
        auto Q = Matrix(ldq, n, Number{});
        auto x = ublas::vector<Number>(m + p, Number{});
        Q(i, 0) = 1;
        Q(m + p - i - 1, 0) = mu;
        x(i) = 1;
        BOOST_VERIFY(tools::is_almost_isometric(Q));

        auto lwork = n;
        constexpr auto nan = tools::not_a_number<Number>::value;
        auto work = ublas::vector<Number>(lwork, nan);
        auto ret = lapack::xUNBDB6(
            m, p, n, &x(0), 1, &x(m), 1, &Q(0, 0), ldq, &Q(m, 0), ldq, &work(0),
            lwork);

        BOOST_TEST_CONTEXT("index=" << i) {
            BOOST_REQUIRE_EQUAL(ret, 0);
            BOOST_CHECK_EQUAL(Real{ublas::norm_inf(x)}, 0);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    require_zero_output, Number, lapack::supported_real_types) {
    require_zero_output_impl(Number{});
}
