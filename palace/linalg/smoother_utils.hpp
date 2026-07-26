// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_LINALG_SMOOTHER_UTILS_HPP
#define PALACE_LINALG_SMOOTHER_UTILS_HPP

#include "linalg/operator.hpp"

namespace palace::internal
{

// Diagonal requirements are smoother policy, not generic vector semantics.
enum class SmootherDiagonalPolicy
{
  FINITE_NONZERO,
  POSITIVE_REAL
};

struct SmootherDiagonalStatus
{
  long long size = 0;
  long long nonfinite = 0;
  long long zero = 0;
  long long nonreal = 0;
  long long nonpositive = 0;

  bool IsValid(SmootherDiagonalPolicy policy) const
  {
    return nonfinite == 0 && ((policy == SmootherDiagonalPolicy::POSITIVE_REAL)
                                  ? (nonreal == 0 && nonpositive == 0)
                                  : (zero == 0));
  }
};

struct SmootherPositiveFiniteStatus
{
  long long size = 0;
  long long nonfinite = 0;
  long long nonpositive = 0;

  bool IsValid() const { return nonfinite == 0 && nonpositive == 0; }
};

// These checks reduce their counts over comm, so every rank observes the same status.
SmootherDiagonalStatus CheckSmootherDiagonal(MPI_Comm comm, const Vector &diag);
SmootherDiagonalStatus CheckSmootherDiagonal(MPI_Comm comm, const ComplexVector &diag);
SmootherPositiveFiniteStatus CheckSmootherPositiveFinite(MPI_Comm comm, double value);

// Query ComplexOperator::IsReal() once and require a communicator-wide consistent
// representation before choosing a collective spectral-estimator algorithm.
bool IsSmootherOperatorReal(MPI_Comm comm, const Operator &op, const char *smoother);
bool IsSmootherOperatorReal(MPI_Comm comm, const ComplexOperator &op, const char *smoother);

void ValidateSmootherDiagonal(MPI_Comm comm, const Vector &diag,
                              SmootherDiagonalPolicy policy, const char *smoother);
void ValidateSmootherDiagonal(MPI_Comm comm, const ComplexVector &diag,
                              SmootherDiagonalPolicy policy, const char *smoother);

void ValidateSmootherReciprocal(MPI_Comm comm, const Vector &dinv, const char *smoother);
void ValidateSmootherReciprocal(MPI_Comm comm, const ComplexVector &dinv,
                                const char *smoother);
void ValidateSmootherPositiveFinite(MPI_Comm comm, double value, const char *quantity);

}  // namespace palace::internal

#endif  // PALACE_LINALG_SMOOTHER_UTILS_HPP
