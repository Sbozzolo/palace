// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "smoother_utils.hpp"

#include <array>
#include <cmath>
#include "utils/communication.hpp"

namespace palace::internal
{

namespace
{

void ReduceStatus(MPI_Comm comm, SmootherDiagonalStatus &status)
{
  std::array<long long, 5> counts = {status.size, status.nonfinite, status.zero,
                                     status.nonreal, status.nonpositive};
  Mpi::GlobalSum(static_cast<int>(counts.size()), counts.data(), comm);
  status = {counts[0], counts[1], counts[2], counts[3], counts[4]};
}

void ReduceStatus(MPI_Comm comm, SmootherPositiveFiniteStatus &status)
{
  std::array<long long, 3> counts = {status.size, status.nonfinite, status.nonpositive};
  Mpi::GlobalSum(static_cast<int>(counts.size()), counts.data(), comm);
  status = {counts[0], counts[1], counts[2]};
}

void VerifyStatus(const SmootherDiagonalStatus &status, SmootherDiagonalPolicy policy,
                  const char *smoother)
{
  MFEM_VERIFY(status.IsValid(policy),
              smoother << " requires "
                       << ((policy == SmootherDiagonalPolicy::POSITIVE_REAL)
                               ? "a finite, exactly-real, strictly positive"
                               : "a finite, nonzero")
                       << " operator diagonal (global entries: " << status.size
                       << ", non-finite: " << status.nonfinite << ", zero: " << status.zero
                       << ", non-real: " << status.nonreal
                       << ", non-positive: " << status.nonpositive << ")!");
}

}  // namespace

SmootherDiagonalStatus CheckSmootherDiagonal(MPI_Comm comm, const Vector &diag)
{
  SmootherDiagonalStatus status;
  status.size = diag.Size();
  const auto *d = diag.HostRead();
  for (int i = 0; i < diag.Size(); i++)
  {
    status.nonfinite += !std::isfinite(d[i]);
    status.zero += d[i] == 0.0;
    status.nonpositive += d[i] <= 0.0;
  }
  ReduceStatus(comm, status);
  return status;
}

SmootherDiagonalStatus CheckSmootherDiagonal(MPI_Comm comm, const ComplexVector &diag)
{
  SmootherDiagonalStatus status;
  status.size = diag.Size();
  const auto *dr = diag.Real().HostRead();
  const auto *di = diag.Imag().HostRead();
  for (int i = 0; i < diag.Size(); i++)
  {
    status.nonfinite += !std::isfinite(dr[i]) || !std::isfinite(di[i]);
    status.zero += dr[i] == 0.0 && di[i] == 0.0;
    status.nonreal += di[i] != 0.0;
    status.nonpositive += dr[i] <= 0.0;
  }
  ReduceStatus(comm, status);
  return status;
}

SmootherPositiveFiniteStatus CheckSmootherPositiveFinite(MPI_Comm comm, double value)
{
  SmootherPositiveFiniteStatus status;
  status.size = 1;
  status.nonfinite = !std::isfinite(value);
  status.nonpositive = value <= 0.0;
  ReduceStatus(comm, status);
  return status;
}

bool IsSmootherOperatorReal(MPI_Comm, const Operator &, const char *)
{
  return true;
}

bool IsSmootherOperatorReal(MPI_Comm comm, const ComplexOperator &op, const char *smoother)
{
  const bool is_real = op.IsReal();
  int real_ranks = is_real ? 1 : 0;
  Mpi::GlobalSum(1, &real_ranks, comm);
  MFEM_VERIFY(real_ranks == 0 || real_ranks == Mpi::Size(comm),
              smoother << " requires ComplexOperator::IsReal() to agree on every rank "
                       << "(real ranks: " << real_ranks
                       << ", communicator size: " << Mpi::Size(comm) << ")!");
  return real_ranks == Mpi::Size(comm);
}

void ValidateSmootherDiagonal(MPI_Comm comm, const Vector &diag,
                              SmootherDiagonalPolicy policy, const char *smoother)
{
  VerifyStatus(CheckSmootherDiagonal(comm, diag), policy, smoother);
}

void ValidateSmootherDiagonal(MPI_Comm comm, const ComplexVector &diag,
                              SmootherDiagonalPolicy policy, const char *smoother)
{
  VerifyStatus(CheckSmootherDiagonal(comm, diag), policy, smoother);
}

void ValidateSmootherReciprocal(MPI_Comm comm, const Vector &dinv, const char *smoother)
{
  const auto status = CheckSmootherDiagonal(comm, dinv);
  MFEM_VERIFY(status.nonfinite == 0,
              smoother << " produced a non-finite reciprocal operator diagonal (global "
                       << "entries: " << status.size << ", non-finite: " << status.nonfinite
                       << ")!");
}

void ValidateSmootherReciprocal(MPI_Comm comm, const ComplexVector &dinv,
                                const char *smoother)
{
  const auto status = CheckSmootherDiagonal(comm, dinv);
  MFEM_VERIFY(status.nonfinite == 0,
              smoother << " produced a non-finite reciprocal operator diagonal (global "
                       << "entries: " << status.size << ", non-finite: " << status.nonfinite
                       << ")!");
}

void ValidateSmootherPositiveFinite(MPI_Comm comm, double value, const char *quantity)
{
  const auto status = CheckSmootherPositiveFinite(comm, value);
  MFEM_VERIFY(status.IsValid(),
              quantity << " must be finite and strictly positive (ranks: " << status.size
                       << ", non-finite: " << status.nonfinite
                       << ", non-positive: " << status.nonpositive << ")!");
}

}  // namespace palace::internal
