// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "portrenormalization.hpp"

#include <cmath>
#include <Eigen/Dense>
#include <mfem.hpp>

namespace palace::port_renorm
{

using std::complex;

Coefficients ComputeCoefficients(double R_ref, complex<double> Z_ref)
{
  MFEM_VERIFY(R_ref > 0.0,
              "Port renormalization requires a positive real reference resistance!");
  Coefficients c;
  c.gamma = (Z_ref - R_ref) / (Z_ref + R_ref);
  const double one_minus_gamma = std::abs(1.0 - c.gamma);
  if (one_minus_gamma > 0.0)
  {
    // A_k = (1 − Γ*) sqrt(1 − |Γ|²) / |1 − Γ|. For a resistive port Γ = 0 ⇒ A = 1.
    const double disc = 1.0 - std::norm(c.gamma);
    c.a_wave = (1.0 - std::conj(c.gamma)) * std::sqrt(std::abs(disc)) / one_minus_gamma;
  }
  else
  {
    c.a_wave = 1.0;
  }
  return c;
}

std::complex<double> RenormalizeReflection(complex<double> S_raw, double R_ref,
                                           complex<double> Z_ref)
{
  // Closed form equivalent to the (i,i) entry of the single-reactive-port renormalization,
  // arranged to avoid the S_raw → 1 singularity of the intermediate input impedance
  // Z_in = R_ref (1 + S_raw)/(1 − S_raw). Reduces to S_raw exactly when Z_ref = R_ref.
  const complex<double> num =
      (R_ref - std::conj(Z_ref)) + (R_ref + std::conj(Z_ref)) * S_raw;
  const complex<double> den = (R_ref + Z_ref) + (R_ref - Z_ref) * S_raw;
  return num / den;
}

std::vector<complex<double>>
RenormalizeColumn(const std::vector<complex<double>> &S_full, int n, int i,
                  const std::vector<Coefficients> &coeffs)
{
  MFEM_VERIFY(static_cast<int>(S_full.size()) == n * n &&
                  static_cast<int>(coeffs.size()) == n && i >= 0 && i < n,
              "Inconsistent sizes in RenormalizeColumn!");

  // Assemble (I − Γ S) and (S − Γ*) using Eigen, solve (I − Γ S) x = e_i, then form
  // S'[:, i] = A^{-1} (S − Γ*) x · conj(A_i). Γ_k = 0 for resistive ports, so those rows
  // reduce to the identity and contribute nothing beyond passing S through.
  Eigen::MatrixXcd S(n, n);
  for (int r = 0; r < n; r++)
  {
    for (int c = 0; c < n; c++)
    {
      S(r, c) = S_full[r * n + c];
    }
  }
  Eigen::VectorXcd gamma(n), a_inv(n);
  for (int k = 0; k < n; k++)
  {
    gamma(k) = coeffs[k].gamma;
    a_inv(k) = 1.0 / coeffs[k].a_wave;
  }

  Eigen::MatrixXcd ImGS = Eigen::MatrixXcd::Identity(n, n) - gamma.asDiagonal() * S;
  Eigen::VectorXcd e_i = Eigen::VectorXcd::Zero(n);
  e_i(i) = 1.0;
  Eigen::VectorXcd x = ImGS.partialPivLu().solve(e_i);

  Eigen::MatrixXcd SmG = S;
  for (int k = 0; k < n; k++)
  {
    SmG(k, k) -= std::conj(gamma(k));
  }
  Eigen::VectorXcd col = a_inv.asDiagonal() * (SmG * x) * std::conj(coeffs[i].a_wave);

  std::vector<complex<double>> out(n);
  for (int k = 0; k < n; k++)
  {
    out[k] = col(k);
  }
  return out;
}

void RenormalizeMatrix(std::vector<complex<double>> &S, int n,
                       const std::vector<Coefficients> &coeffs)
{
  MFEM_VERIFY(static_cast<int>(S.size()) == n * n &&
                  static_cast<int>(coeffs.size()) == n,
              "Inconsistent sizes in RenormalizeMatrix!");
  Eigen::MatrixXcd Sm(n, n);
  for (int r = 0; r < n; r++)
  {
    for (int c = 0; c < n; c++)
    {
      Sm(r, c) = S[r * n + c];
    }
  }
  Eigen::VectorXcd gamma(n), a(n), a_inv(n);
  for (int k = 0; k < n; k++)
  {
    gamma(k) = coeffs[k].gamma;
    a(k) = coeffs[k].a_wave;
    a_inv(k) = 1.0 / coeffs[k].a_wave;
  }
  Eigen::MatrixXcd SmG = Sm;
  for (int k = 0; k < n; k++)
  {
    SmG(k, k) -= std::conj(gamma(k));
  }
  Eigen::MatrixXcd ImGS = Eigen::MatrixXcd::Identity(n, n) - gamma.asDiagonal() * Sm;
  Eigen::MatrixXcd Sp = a_inv.asDiagonal() * SmG * ImGS.partialPivLu().solve(
                                                 Eigen::MatrixXcd(a.conjugate().asDiagonal()));
  for (int r = 0; r < n; r++)
  {
    for (int c = 0; c < n; c++)
    {
      S[r * n + c] = Sp(r, c);
    }
  }
}

}  // namespace palace::port_renorm
