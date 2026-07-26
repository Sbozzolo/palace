// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "linalg/chebyshev.hpp"
#include "linalg/jacobi.hpp"
#include "linalg/operator.hpp"
#include "linalg/smoother_utils.hpp"
#include "utils/communication.hpp"

namespace palace
{

using namespace Catch::Matchers;

namespace
{

class TestOperator : public Operator
{
private:
  double a00, a01, a10, a11;

public:
  TestOperator(double a00, double a01, double a10, double a11)
    : Operator(2), a00(a00), a01(a01), a10(a10), a11(a11)
  {
  }

  void AssembleDiagonal(Vector &diag) const override
  {
    diag.SetSize(2);
    diag[0] = a00;
    diag[1] = a11;
  }

  void Mult(const Vector &x, Vector &y) const override
  {
    y.SetSize(2);
    const double x0 = x[0], x1 = x[1];
    y[0] = a00 * x0 + a01 * x1;
    y[1] = a10 * x0 + a11 * x1;
  }

  void MultTranspose(const Vector &x, Vector &y) const override
  {
    y.SetSize(2);
    const double x0 = x[0], x1 = x[1];
    y[0] = a00 * x0 + a10 * x1;
    y[1] = a01 * x0 + a11 * x1;
  }
};

class RankDependentIsRealOperator : public ComplexOperator
{
private:
  MPI_Comm comm;

public:
  RankDependentIsRealOperator(MPI_Comm comm) : ComplexOperator(1), comm(comm) {}

  bool IsReal() const override { return Mpi::Rank(comm) == 0; }

  void Mult(const ComplexVector &x, ComplexVector &y) const override { y = x; }
};

const double lambda_max = 1.0 + 1.0 / std::sqrt(2.0);

void CheckFourthKind(ChebyshevSmoother<Operator> &smoother, const Operator &A)
{
  smoother.SetOperator(A);
  Vector x(2), y(2);
  x[0] = 100.0;
  x[1] = 2.0;
  smoother.Mult(x, y);
  const double expected = 4.0 / (3.0 * lambda_max);
  CHECK_THAT(y[0], WithinRel(expected, 2.0e-3));
  CHECK_THAT(y[1], WithinRel(expected, 2.0e-3));
}

void CheckFourthKind(ChebyshevSmoother<ComplexOperator> &smoother, const ComplexOperator &A)
{
  smoother.SetOperator(A);
  ComplexVector x(2), y(2);
  x = 0.0;
  x.Real()[0] = 100.0;
  x.Real()[1] = 2.0;
  smoother.Mult(x, y);
  const double expected = 4.0 / (3.0 * lambda_max);
  CHECK_THAT(y.Real()[0], WithinRel(expected, 2.0e-3));
  CHECK_THAT(y.Real()[1], WithinRel(expected, 2.0e-3));
  CHECK(y.Imag()[0] == 0.0);
  CHECK(y.Imag()[1] == 0.0);
}

void CheckFirstKind(ChebyshevSmoother1stKind<Operator> &smoother, const Operator &A)
{
  smoother.SetOperator(A);
  Vector x(2), y(2);
  x[0] = 100.0;
  x[1] = 2.0;
  smoother.Mult(x, y);
  constexpr double sf_min = 0.25;
  const double expected = 2.0 / ((1.0 + sf_min) * lambda_max);
  CHECK_THAT(y[0], WithinRel(expected, 2.0e-3));
  CHECK_THAT(y[1], WithinRel(expected, 2.0e-3));
}

void CheckFirstKind(ChebyshevSmoother1stKind<ComplexOperator> &smoother,
                    const ComplexOperator &A)
{
  smoother.SetOperator(A);
  ComplexVector x(2), y(2);
  x = 0.0;
  x.Real()[0] = 100.0;
  x.Real()[1] = 2.0;
  smoother.Mult(x, y);
  constexpr double sf_min = 0.25;
  const double expected = 2.0 / ((1.0 + sf_min) * lambda_max);
  CHECK_THAT(y.Real()[0], WithinRel(expected, 2.0e-3));
  CHECK_THAT(y.Real()[1], WithinRel(expected, 2.0e-3));
  CHECK(y.Imag()[0] == 0.0);
  CHECK(y.Imag()[1] == 0.0);
}

}  // namespace

TEST_CASE("Chebyshev uses the Hermitian Jacobi similarity spectrum",
          "[smoother][Serial][Parallel]")
{
  // A is SPD with a nonuniform diagonal. D⁻¹A is not symmetric, while
  // D⁻¹ᐟ²AD⁻¹ᐟ² has eigenvalues 1 ± 1/sqrt(2).
  TestOperator A(100.0, 10.0, 10.0, 2.0);
  ComplexWrapperOperator Ac(&A, nullptr);

  SECTION("fourth-kind real")
  {
    ChebyshevSmoother<Operator> smoother(Mpi::World(), 1, 1, 1.0);
    CheckFourthKind(smoother, A);
  }
  SECTION("fourth-kind exactly-real complex")
  {
    ChebyshevSmoother<ComplexOperator> smoother(Mpi::World(), 1, 1, 1.0);
    CheckFourthKind(smoother, Ac);
  }
  SECTION("first-kind real")
  {
    ChebyshevSmoother1stKind<Operator> smoother(Mpi::World(), 1, 1, 1.0, 0.25);
    CheckFirstKind(smoother, A);
  }
  SECTION("first-kind exactly-real complex")
  {
    ChebyshevSmoother1stKind<ComplexOperator> smoother(Mpi::World(), 1, 1, 1.0, 0.25);
    CheckFirstKind(smoother, Ac);
  }
}

TEST_CASE("General-complex smoothers use the analytic SVD norm",
          "[smoother][Serial][Parallel]")
{
  // The complex-symmetric A = [[1, 2i], [2i, 1]] has D = I and
  // AᴴA = 5I, so ||D⁻¹A||₂ = sqrt(5). It is not Hermitian.
  TestOperator Ar(1.0, 0.0, 0.0, 1.0);
  TestOperator Ai(0.0, 2.0, 2.0, 0.0);
  ComplexWrapperOperator A(&Ar, &Ai);
  ComplexVector x(2), y(2);
  x = 0.0;
  x.Real()[0] = 1.0;
  const double norm = std::sqrt(5.0);

  SECTION("fourth-kind Chebyshev")
  {
    ChebyshevSmoother<ComplexOperator> smoother(Mpi::World(), 1, 1, 1.0);
    smoother.SetOperator(A);
    smoother.Mult(x, y);
    CHECK_THAT(y.Real()[0], WithinRel(4.0 / (3.0 * norm), 2.0e-3));
    CHECK(y.Imag()[0] == 0.0);
    CHECK(y.Real()[1] == 0.0);
    CHECK(y.Imag()[1] == 0.0);
  }
  SECTION("automatically damped Jacobi")
  {
    JacobiSmoother<ComplexOperator> smoother(Mpi::World(), 0.0);
    smoother.SetOperator(A);
    smoother.Mult(x, y);
    CHECK_THAT(y.Real()[0], WithinRel(2.0 / norm, 2.0e-3));
    CHECK(y.Imag()[0] == 0.0);
    CHECK(y.Real()[1] == 0.0);
    CHECK(y.Imag()[1] == 0.0);
  }
}

TEST_CASE("Automatic Jacobi uses the Hermitian Jacobi similarity spectrum",
          "[smoother][Serial][Parallel]")
{
  TestOperator A(100.0, 10.0, 10.0, 2.0);
  ComplexWrapperOperator Ac(&A, nullptr);

  SECTION("real")
  {
    JacobiSmoother<Operator> smoother(Mpi::World(), 0.0);
    smoother.SetOperator(A);
    Vector x(2), y(2);
    x[0] = 100.0;
    x[1] = 2.0;
    smoother.Mult(x, y);
    CHECK_THAT(y[0], WithinRel(2.0 / lambda_max, 2.0e-3));
    CHECK_THAT(y[1], WithinRel(2.0 / lambda_max, 2.0e-3));
  }
  SECTION("exactly-real complex")
  {
    JacobiSmoother<ComplexOperator> smoother(Mpi::World(), 0.0);
    smoother.SetOperator(Ac);
    ComplexVector x(2), y(2);
    x = 0.0;
    x.Real()[0] = 100.0;
    x.Real()[1] = 2.0;
    smoother.Mult(x, y);
    CHECK_THAT(y.Real()[0], WithinRel(2.0 / lambda_max, 2.0e-3));
    CHECK_THAT(y.Real()[1], WithinRel(2.0 / lambda_max, 2.0e-3));
  }
}

TEST_CASE("Automatic Jacobi re-estimates after repeated SetOperator",
          "[smoother][Serial][Parallel]")
{
  TestOperator A1(1.0, 0.0, 0.0, 1.0);
  TestOperator A2(1.0, 0.9, 0.9, 1.0);
  JacobiSmoother<Operator> smoother(Mpi::World(), 0.0);
  Vector x(2), y(2);
  x[0] = 1.0;
  x[1] = 0.0;

  smoother.SetOperator(A1);
  smoother.Mult(x, y);
  CHECK_THAT(y[0], WithinRel(2.0, 2.0e-3));

  smoother.SetOperator(A2);
  smoother.Mult(x, y);
  CHECK_THAT(y[0], WithinRel(2.0 / 1.9, 2.0e-3));
}

TEST_CASE("Invalid smoother spectral scaling is rejected collectively",
          "[smoother][Serial][Parallel]")
{
  TestOperator A(1.0, 0.0, 0.0, 1.0);

  SECTION("automatic Jacobi scaling factor")
  {
    JacobiSmoother<Operator> smoother(Mpi::World(), 0.0, 0.0);
    CHECK_THROWS(smoother.SetOperator(A));
  }
  SECTION("Chebyshev scaled estimate")
  {
    ChebyshevSmoother<Operator> smoother(Mpi::World(), 1, 1,
                                         std::numeric_limits<double>::infinity());
    CHECK_THROWS(smoother.SetOperator(A));
  }
}

TEST_CASE("Fixed Jacobi accepts an invertible negative diagonal",
          "[smoother][Serial][Parallel]")
{
  TestOperator A(-2.0, 0.0, 0.0, 4.0);
  JacobiSmoother<Operator> smoother(Mpi::World(), 1.0);
  smoother.SetOperator(A);
  Vector x(2), y(2);
  x[0] = -2.0;
  x[1] = 4.0;
  smoother.Mult(x, y);
  CHECK_THAT(y[0], WithinRel(1.0));
  CHECK_THAT(y[1], WithinRel(1.0));
}

TEST_CASE("Smoother diagonal policies are collective", "[smoother][Serial][Parallel]")
{
  using internal::SmootherDiagonalPolicy;
  const auto comm = Mpi::World();

  SECTION("negative")
  {
    Vector d(1);
    d[0] = -1.0;
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK(status.IsValid(SmootherDiagonalPolicy::FINITE_NONZERO));
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::POSITIVE_REAL));
  }
  SECTION("zero")
  {
    Vector d(1);
    d[0] = 0.0;
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::FINITE_NONZERO));
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::POSITIVE_REAL));
  }
  SECTION("non-finite on one rank")
  {
    Vector d(1);
    d[0] = (Mpi::Rank(comm) == Mpi::Size(comm) - 1)
               ? std::numeric_limits<double>::quiet_NaN()
               : 1.0;
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK(status.nonfinite == 1);
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::FINITE_NONZERO));
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::POSITIVE_REAL));
  }
  SECTION("infinite")
  {
    Vector d(1);
    d[0] = std::numeric_limits<double>::infinity();
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::FINITE_NONZERO));
  }
  SECTION("non-real complex")
  {
    ComplexVector d(1);
    d.Real()[0] = 1.0;
    d.Imag()[0] = 0.5;
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK(status.IsValid(SmootherDiagonalPolicy::FINITE_NONZERO));
    CHECK_FALSE(status.IsValid(SmootherDiagonalPolicy::POSITIVE_REAL));
  }
  SECTION("non-finite reciprocal")
  {
    ComplexVector d(1);
    d.Real()[0] = std::numeric_limits<double>::denorm_min();
    d.Imag()[0] = 0.0;
    d.Reciprocal();
    const auto status = internal::CheckSmootherDiagonal(comm, d);
    CHECK(status.nonfinite == Mpi::Size(comm));
  }
  SECTION("positive finite scalar")
  {
    const auto status = internal::CheckSmootherPositiveFinite(comm, 1.0);
    CHECK(status.IsValid());
  }
  SECTION("zero scalar")
  {
    const auto status = internal::CheckSmootherPositiveFinite(comm, 0.0);
    CHECK(status.nonpositive == Mpi::Size(comm));
    CHECK_FALSE(status.IsValid());
  }
  SECTION("non-finite scalar on one rank")
  {
    const double value = (Mpi::Rank(comm) == Mpi::Size(comm) - 1)
                             ? std::numeric_limits<double>::infinity()
                             : 1.0;
    const auto status = internal::CheckSmootherPositiveFinite(comm, value);
    CHECK(status.nonfinite == 1);
    CHECK_FALSE(status.IsValid());
  }
  SECTION("rank-inconsistent IsReal representation")
  {
    RankDependentIsRealOperator op(comm);
    if (Mpi::Size(comm) == 1)
    {
      CHECK(internal::IsSmootherOperatorReal(comm, op, "Test smoother"));
    }
    else
    {
      CHECK_THROWS(internal::IsSmootherOperatorReal(comm, op, "Test smoother"));
    }
  }
}

}  // namespace palace
