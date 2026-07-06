// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "models/portrenormalization.hpp"

using namespace palace;
using std::complex;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{

// Reference generalized-S renormalization computed directly from the input impedance, for
// cross-checking the singularity-free closed form. Z_in = R_ref (1 + S)/(1 − S); the
// conjugate-match reflection is (Z_in − Z_ref*)/(Z_in + Z_ref).
complex<double> ReflectionDirect(complex<double> S_raw, double R_ref, complex<double> Z_ref)
{
  complex<double> Z_in = R_ref * (1.0 + S_raw) / (1.0 - S_raw);
  return (Z_in - std::conj(Z_ref)) / (Z_in + Z_ref);
}

}  // namespace

TEST_CASE("PortRenorm_Coefficients_Resistive", "[portrenorm][Serial]")
{
  // A resistive port (Z_ref == R_ref) must give Γ = 0, A = 1 so it drops out of the
  // transform and its scattering entries are unchanged.
  auto c = port_renorm::ComputeCoefficients(50.0, complex<double>{50.0, 0.0});
  CHECK_THAT(std::abs(c.gamma), WithinAbs(0.0, 1e-15));
  CHECK_THAT(c.a_wave.real(), WithinRel(1.0));
  CHECK_THAT(c.a_wave.imag(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("PortRenorm_Reflection_ReducesToRaw_WhenResistive", "[portrenorm][Serial]")
{
  // With Z_ref == R_ref the reflection renormalization is the identity.
  const double R_ref = 50.0;
  for (auto S_raw : {complex<double>{0.3, -0.4}, complex<double>{-0.7, 0.2},
                     complex<double>{0.0, 0.0}, complex<double>{0.9, 0.0}})
  {
    auto S = port_renorm::RenormalizeReflection(S_raw, R_ref, {R_ref, 0.0});
    CHECK_THAT(S.real(), WithinRel(S_raw.real(), 1e-12) || WithinAbs(S_raw.real(), 1e-14));
    CHECK_THAT(S.imag(), WithinRel(S_raw.imag(), 1e-12) || WithinAbs(S_raw.imag(), 1e-14));
  }
}

TEST_CASE("PortRenorm_Reflection_MatchesDirectFormula", "[portrenorm][Serial]")
{
  // Singularity-free closed form must equal the direct Z_in-based formula.
  const double R_ref = 50.0;
  const complex<double> Z_ref{30.0, 45.0};
  for (auto S_raw : {complex<double>{0.3, -0.4}, complex<double>{-0.6, 0.25},
                     complex<double>{0.1, 0.8}})
  {
    auto S_closed = port_renorm::RenormalizeReflection(S_raw, R_ref, Z_ref);
    auto S_direct = ReflectionDirect(S_raw, R_ref, Z_ref);
    CHECK_THAT(std::abs(S_closed - S_direct), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("PortRenorm_Reflection_LosslessDriveGivesUnitMagnitude", "[portrenorm][Serial]")
{
  // A purely reactive reference (R = 0 ⇒ R_ref = unit) into a lossless network must reflect
  // all power: |S'| = 1 regardless of the raw phase.
  const double R_ref = 1.0;  // internal unit reference used for a purely reactive port
  const complex<double> Z_ref{0.0, 50.0};  // pure inductive reactance
  for (auto S_raw : {complex<double>{0.0, 0.0}, complex<double>{0.5, 0.0},
                     complex<double>{0.0, 0.7}, complex<double>{0.6, -0.3}})
  {
    auto S = port_renorm::RenormalizeReflection(S_raw, R_ref, Z_ref);
    CHECK_THAT(std::abs(S), WithinRel(1.0, 1e-10));
  }
}

TEST_CASE("PortRenorm_Column_MatchesReflectionClosedForm", "[portrenorm][Serial]")
{
  // For a single reactive drive port among resistive ports, the column solve must reproduce
  // the reflection closed form on the diagonal and the per-column scalar off-diagonal.
  const int n = 3;
  const double R_ref = 50.0;
  const int i = 0;  // reactive drive
  const complex<double> Z_ref_drive{20.0, -60.0};

  // Symmetric (reciprocal) raw scattering matrix, all ports referenced to R_ref.
  std::vector<complex<double>> S = {
      {0.20, 0.10}, {0.30, -0.05}, {-0.10, 0.02},  // row 0
      {0.30, -0.05}, {-0.15, 0.08}, {0.22, 0.03},  // row 1
      {-0.10, 0.02}, {0.22, 0.03}, {0.05, -0.12}   // row 2
  };
  std::vector<port_renorm::Coefficients> coeffs(n);
  coeffs[i] = port_renorm::ComputeCoefficients(R_ref, Z_ref_drive);
  coeffs[1] = port_renorm::ComputeCoefficients(R_ref, {R_ref, 0.0});
  coeffs[2] = port_renorm::ComputeCoefficients(R_ref, {R_ref, 0.0});

  auto col = port_renorm::RenormalizeColumn(S, n, i, coeffs);

  // Diagonal equals the closed-form reflection renormalization.
  auto refl = port_renorm::RenormalizeReflection(S[i * n + i], R_ref, Z_ref_drive);
  CHECK_THAT(std::abs(col[i] - refl), WithinAbs(0.0, 1e-12));

  // Off-diagonal equals conj(A_i)/(1 − Γ_i S_ii) · S_raw[j,i].
  const complex<double> factor =
      std::conj(coeffs[i].a_wave) / (1.0 - coeffs[i].gamma * S[i * n + i]);
  for (int j = 1; j < n; j++)
  {
    complex<double> expect = factor * S[j * n + i];
    CHECK_THAT(std::abs(col[j] - expect), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("PortRenorm_Matrix_MultipleReactivePorts", "[portrenorm][Serial]")
{
  // Full-matrix renorm with two simultaneously reactive ports must match the per-column
  // solve for every column (self-consistency of the two entry points).
  const int n = 4;
  const double R_ref = 50.0;
  std::vector<complex<double>> S = {
      {0.10, 0.05},  {0.25, -0.10}, {-0.08, 0.02}, {0.12, 0.06},
      {0.25, -0.10}, {-0.20, 0.07}, {0.18, 0.04},  {-0.05, 0.09},
      {-0.08, 0.02}, {0.18, 0.04},  {0.06, -0.11}, {0.21, -0.03},
      {0.12, 0.06},  {-0.05, 0.09}, {0.21, -0.03}, {-0.14, 0.08}};

  std::vector<port_renorm::Coefficients> coeffs(n);
  coeffs[0] = port_renorm::ComputeCoefficients(R_ref, {30.0, 40.0});
  coeffs[1] = port_renorm::ComputeCoefficients(R_ref, {R_ref, 0.0});
  coeffs[2] = port_renorm::ComputeCoefficients(R_ref, {70.0, -25.0});
  coeffs[3] = port_renorm::ComputeCoefficients(R_ref, {R_ref, 0.0});

  std::vector<complex<double>> S_full = S;
  port_renorm::RenormalizeMatrix(S_full, n, coeffs);

  for (int i = 0; i < n; i++)
  {
    auto col = port_renorm::RenormalizeColumn(S, n, i, coeffs);
    for (int j = 0; j < n; j++)
    {
      CHECK_THAT(std::abs(col[j] - S_full[j * n + i]), WithinAbs(0.0, 1e-11));
    }
  }
}
