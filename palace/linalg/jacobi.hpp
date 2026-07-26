// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_LINALG_JACOBI_SMOOTHER_HPP
#define PALACE_LINALG_JACOBI_SMOOTHER_HPP

#include "linalg/operator.hpp"
#include "linalg/solver.hpp"
#include "linalg/vector.hpp"

namespace palace
{

//
// Simple Jacobi smoother using the diagonal vector from OperType::AssembleDiagonal(),
// which allows for (approximate) diagonal construction for matrix-free operators. Fixed
// damping requires a finite, nonzero diagonal but permits negative entries. Automatic
// damping for real and exactly-real complex operators additionally requires a Hermitian
// positive semidefinite operator with a strictly positive diagonal and a finite, positive
// maximum-eigenvalue scaling factor. Every accepted diagonal entry must have a finite
// representable reciprocal.
//
template <typename OperType>
class JacobiSmoother : public Solver<OperType>
{
  using VecType = typename Solver<OperType>::VecType;

private:
  // MPI communicator associated with the solver operator and vectors.
  MPI_Comm comm;

  // Inverse diagonal scaling of the operator (real-valued for now).
  VecType dinv;

  // Constructor-selected damping mode, damping factor, and scaling factor for the maximum
  // eigenvalue. Automatic mode is immutable across repeated SetOperator() calls.
  const bool automatic_damping;
  const double omega, sf_max;

public:
  JacobiSmoother(MPI_Comm comm, double omega = 1.0, double sf_max = 1.0)
    : Solver<OperType>(), comm(comm), automatic_damping(omega == 0.0), omega(omega),
      sf_max(sf_max)
  {
  }

  void SetOperator(const OperType &op) override;

  void Mult(const VecType &x, VecType &y) const override;

  void MultTranspose(const VecType &x, VecType &y) const override { Mult(x, y); }
};

}  // namespace palace

#endif  // PALACE_LINALG_JACOBI_SMOOTHER_HPP
