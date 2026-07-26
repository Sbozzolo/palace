// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "jacobi.hpp"

#include <mfem/general/forall.hpp>
#include "linalg/smoother_utils.hpp"

namespace palace
{

namespace
{

double GetLambdaMax(MPI_Comm comm, const Operator &A, const Vector &dinv,
                    bool use_hermitian)
{
  MFEM_ASSERT(use_hermitian, "Real Jacobi operators use the Hermitian estimator!");
  // Automatic damping owns a Hermitian-PSD contract. Use the Euclidean-Hermitian
  // similarity D⁻¹ᐟ²AD⁻¹ᐟ², which has the spectrum targeted by Jacobi's D⁻¹A.
  Vector dinv_sqrt(dinv);
  linalg::Sqrt(dinv_sqrt);
  DiagonalOperator DinvSqrt(dinv_sqrt);
  ProductOperator ADinvSqrt(A, DinvSqrt);
  ProductOperator S(DinvSqrt, ADinvSqrt);
  return linalg::SpectralNorm(comm, S, true);
}

double GetLambdaMax(MPI_Comm comm, const ComplexOperator &A, const ComplexVector &dinv,
                    bool use_hermitian)
{
  if (use_hermitian)
  {
    // The collective IsReal() result selects an exactly-real representation; it does not
    // establish Hermiticity. The automatic-damping caller contract supplies that fact.
    ComplexVector dinv_sqrt(dinv);
    linalg::Sqrt(dinv_sqrt.Real());
    ComplexDiagonalOperator DinvSqrt(dinv_sqrt);
    ComplexProductOperator ADinvSqrt(A, DinvSqrt);
    ComplexProductOperator S(DinvSqrt, ADinvSqrt);
    return linalg::SpectralNorm(comm, S, true);
  }

  // Keep general-complex automatic damping non-Hermitian-safe.
  ComplexDiagonalOperator Dinv(dinv);
  ComplexProductOperator DinvA(Dinv, A);
  return linalg::SpectralNorm(comm, DinvA, false);
}

template <bool Transpose = false>
inline void Apply(const Vector &dinv, const Vector &x, Vector &y)
{
  const bool use_dev = dinv.UseDevice() || x.UseDevice() || y.UseDevice();
  const int N = dinv.Size();
  const auto *DI = dinv.Read(use_dev);
  const auto *X = x.Read(use_dev);
  auto *Y = y.Write(use_dev);
  mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE(int i) { Y[i] = DI[i] * X[i]; });
}

template <bool Transpose = false>
inline void Apply(const ComplexVector &dinv, const ComplexVector &x, ComplexVector &y)
{
  const bool use_dev = dinv.UseDevice() || x.UseDevice() || y.UseDevice();
  const int N = dinv.Size();
  const auto *DIR = dinv.Real().Read(use_dev);
  const auto *DII = dinv.Imag().Read(use_dev);
  const auto *XR = x.Real().Read(use_dev);
  const auto *XI = x.Imag().Read(use_dev);
  auto *YR = y.Real().Write(use_dev);
  auto *YI = y.Imag().Write(use_dev);
  if constexpr (!Transpose)
  {
    mfem::forall_switch(use_dev, N,
                        [=] MFEM_HOST_DEVICE(int i)
                        {
                          YR[i] = DIR[i] * XR[i] - DII[i] * XI[i];
                          YI[i] = DII[i] * XR[i] + DIR[i] * XI[i];
                        });
  }
  else
  {
    mfem::forall_switch(use_dev, N,
                        [=] MFEM_HOST_DEVICE(int i)
                        {
                          YR[i] = DIR[i] * XR[i] + DII[i] * XI[i];
                          YI[i] = -DII[i] * XR[i] + DIR[i] * XI[i];
                        });
  }
}

}  // namespace

template <typename OperType>
void JacobiSmoother<OperType>::SetOperator(const OperType &op)
{
  dinv.SetSize(op.Height());
  dinv.UseDevice(true);
  op.AssembleDiagonal(dinv);
  bool use_hermitian = false;
  if (automatic_damping)
  {
    internal::ValidateSmootherPositiveFinite(
        comm, sf_max,
        "Automatically damped Jacobi smoother maximum eigenvalue scaling factor");
    use_hermitian =
        internal::IsSmootherOperatorReal(comm, op, "Automatically damped Jacobi smoother");
  }
  internal::ValidateSmootherDiagonal(
      comm, dinv,
      (automatic_damping && use_hermitian)
          ? internal::SmootherDiagonalPolicy::POSITIVE_REAL
          : internal::SmootherDiagonalPolicy::FINITE_NONZERO,
      automatic_damping ? "Automatically damped Jacobi smoother" : "Jacobi smoother");
  dinv.Reciprocal();
  internal::ValidateSmootherReciprocal(
      comm, dinv,
      automatic_damping ? "Automatically damped Jacobi smoother" : "Jacobi smoother");

  // A constructor damping of zero selects automatic mode permanently. Re-estimate a local
  // effective damping on every setup so repeated SetOperator() calls track the new
  // operator.
  double omega_eff = omega;
  if (automatic_damping)
  {
    const double lambda_max = GetLambdaMax(comm, op, dinv, use_hermitian);
    internal::ValidateSmootherPositiveFinite(
        comm, lambda_max,
        "Automatically damped Jacobi smoother maximum eigenvalue estimate");
    const double damping_denominator = sf_max * lambda_max;
    internal::ValidateSmootherPositiveFinite(
        comm, damping_denominator,
        "Automatically damped Jacobi smoother damping denominator");
    omega_eff = 2.0 / damping_denominator;
    internal::ValidateSmootherPositiveFinite(
        comm, omega_eff, "Automatically damped Jacobi smoother effective damping");
  }
  if (omega_eff != 1.0)
  {
    dinv *= omega_eff;
  }

  this->height = op.Height();
  this->width = op.Width();
}

template <typename OperType>
void JacobiSmoother<OperType>::Mult(const VecType &x, VecType &y) const
{
  MFEM_ASSERT(!this->initial_guess, "JacobiSmoother does not use initial guess!");
  Apply(dinv, x, y);
}

template class JacobiSmoother<Operator>;
template class JacobiSmoother<ComplexOperator>;

}  // namespace palace
