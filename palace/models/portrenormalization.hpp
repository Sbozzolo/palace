// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_PORT_RENORMALIZATION_HPP
#define PALACE_MODELS_PORT_RENORMALIZATION_HPP

#include <complex>
#include <vector>

namespace palace
{

// Generalized (Kurokawa power-wave) scattering-parameter renormalization for ports whose
// reference impedance is complex (reactive lumped ports). Palace measures scattering
// parameters referenced to a real per-port reference resistance R_ref,k (the resistance of
// a resistive port, or the unit internal reference for a purely reactive port). A port with
// a reactive circuit termination has a physical reference impedance
//
//     Z_ref,k(ω) = R_k ‖ (iωL_k) ‖ (1/(iωC_k)),
//
// which is complex and frequency dependent. Reporting scattering parameters against this
// complex reference requires renormalizing the measured matrix.
//
// The conjugate-match (power-wave) renormalization from a real diagonal reference R_ref to a
// complex diagonal reference Z_ref is, in matrix form [Kurokawa 1965]:
//
//     Γ_k = (Z_ref,k − R_ref,k) / (Z_ref,k + R_ref,k),
//     A_k = (1 − Γ_k*) · sqrt(1 − |Γ_k|²) / |1 − Γ_k|,
//     S' = diag(A)^{-1} (S − diag(Γ)*) (I − diag(Γ) S)^{-1} diag(A)*.
//
// For a resistive port Γ_k = 0 and A_k = 1, so it drops out and S'_kk = S_kk is unchanged —
// resistive-only systems are therefore untouched (bit-identical). For a lossless drive into
// a lossless termination the transformation yields |S'| = 1, as required physically.
namespace port_renorm
{

// Per-port renormalization coefficients for the transform above. Resistive ports (Z_ref
// real and equal to R_ref) yield gamma = 0, a_wave = 1.
struct Coefficients
{
  std::complex<double> gamma = 0.0;   // Γ_k
  std::complex<double> a_wave = 1.0;  // A_k
};

// Compute Γ_k and A_k for a single port from its real reference R_ref and complex reference
// Z_ref. R_ref must be > 0. If Z_ref == R_ref (resistive) the result is {0, 1} to machine
// precision, so callers may pass every port through this without special-casing.
Coefficients ComputeCoefficients(double R_ref, std::complex<double> Z_ref);

// Renormalize a full scattering matrix S (row-major, size n×n, referenced to the real
// per-port references implied by coeffs) to the complex references encoded in coeffs, in
// place. Applies S' = A^{-1} (S − Γ*) (I − Γ S)^{-1} A*. n is the matrix dimension.
//
// This requires the FULL matrix: entry S'[j][i] depends on the rows of S at every port with
// Γ_k ≠ 0 (the reactive ports), through (I − Γ S)^{-1}. Callers that only have a subset of
// columns (e.g. only excited ports) must instead use RenormalizeColumn for each available
// drive column and leave un-renormalizable ports referenced to R_ref.
void RenormalizeMatrix(std::vector<std::complex<double>> &S, int n,
                       const std::vector<Coefficients> &coeffs);

// Renormalize a single measured column i of the scattering matrix (S_raw[:, i], the response
// at every port to driving port i), returning the renormalized column S'[:, i]. Requires the
// rows of S at every reactive port (Γ_k ≠ 0), supplied as the full raw matrix S_full
// (row-major n×n); only its reactive rows and column i are read. Exact for any reactive
// subset. When only port i is reactive this reduces to the per-column scalar
//     S'[j][i] = conj(A_i) / (1 − Γ_i S_ii) · S_raw[j][i].
std::vector<std::complex<double>>
RenormalizeColumn(const std::vector<std::complex<double>> &S_full, int n, int i,
                  const std::vector<Coefficients> &coeffs);

// Closed-form, singularity-free renormalization of a single reflection (diagonal) entry
// S_ii from real reference R_ref to complex reference Z_ref, when only that port is being
// renormalized (all other ports resistive):
//     S' = ((R_ref − Z_ref*) + (R_ref + Z_ref*) S_raw)
//          / ((R_ref + Z_ref) + (R_ref − Z_ref) S_raw).
// Equal to the (i,i) entry of RenormalizeColumn in that single-reactive-port case; provided
// separately for the inline uniform-solver path where only the drive reflection is needed.
std::complex<double> RenormalizeReflection(std::complex<double> S_raw, double R_ref,
                                           std::complex<double> Z_ref);

}  // namespace port_renorm

}  // namespace palace

#endif  // PALACE_MODELS_PORT_RENORMALIZATION_HPP
