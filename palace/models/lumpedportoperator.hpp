// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_MODELS_LUMPED_PORT_OPERATOR_HPP
#define PALACE_MODELS_LUMPED_PORT_OPERATOR_HPP

#include <complex>
#include <map>
#include <memory>
#include <vector>
#include <mfem.hpp>
#include "fem/lumpedelement.hpp"

namespace palace
{

class GridFunction;
class IoData;
class MaterialOperator;
class Units;
class MaterialPropertyCoefficient;
class SumVectorCoefficient;

namespace config
{

struct LumpedPortData;

}  // namespace config

//
// Helper class for lumped ports in a model.
//
class LumpedPortData
{
public:
  // Reference to material property data (not owned).
  const MaterialOperator &mat_op;

  // To accommodate multielement lumped ports, a port may be made up of elements with
  // different attributes and directions which add in parallel.
  std::vector<std::unique_ptr<LumpedElementData>> elems;

  // Lumped port properties.
  double R, L, C;
  int excitation;
  bool active;
  bool include_in_synthesis;

protected:
  // Linear forms for postprocessing integrated quantities on the port.
  mutable std::unique_ptr<mfem::LinearForm> s, v;

  void InitializeLinearForms(mfem::ParFiniteElementSpace &nd_fespace) const;

public:
  LumpedPortData(const config::LumpedPortData &data, const MaterialOperator &mat_op,
                 const mfem::ParMesh &mesh);

  double GetToSquare(const LumpedElementData &elem) const
  {
    return elem.GetGeometryWidth() / elem.GetGeometryLength() * elems.size();
  }

  // Normalization of tangential electric field of port corresponding ∫|E|²ds = |Z₀| *
  // ∑(Wₑ/Lₑ). In this function we set the impedance magnitude|Z₀| = 1 in internal units (=
  // Z_freespace Ohm). The actual lumped impedance is frequency dependant for ports with L,C
  // so it is more convenient to add this factor later as needed.
  double GetExcitationFieldEtNormSqWithUnityZR() const
  {
    double norm_et = 0.0;
    for (const auto &el : elems)
    {
      norm_et += el->GetGeometryWidth() / el->GetGeometryLength();
    }
    return norm_et;
  }

  // Normalization of tangential electric field of port corresponding ∫|H|²ds = 1 / (|Z₀|
  // nₑₗₑₘₛ²) * ∑(Lₑ/Wₑ). Same details as in GetExcitationFieldEtNormSqWithUnityZR above.
  double GetExcitationFieldHtNormSqWithUnityZR() const
  {
    double norm_ht = 0.0;
    for (const auto &el : elems)
    {
      norm_ht += el->GetGeometryLength() / el->GetGeometryWidth();
    }
    norm_ht /= elems.size() * elems.size();
    return norm_ht;
  }

  constexpr bool HasExcitation() const { return excitation != 0; }

  // True if the port carries reactance (L and/or C) in addition to (or instead of) a
  // resistance. Used to dispatch the reactive-port excitation path: a purely resistive
  // port (L == C == 0) always takes the legacy code path so its results are unchanged.
  constexpr bool HasReactance() const { return L != 0.0 || C != 0.0; }

  // Reference resistance used to normalize the incident-field amplitude of an excited
  // port. For a resistive port this is R (so the reference impedance is the port's own
  // resistance and legacy behavior is preserved exactly). For a purely reactive port
  // (R == 0) there is no real resistance to reference the incident power to, so we fall
  // back to the unit reference impedance |Z_R| = 1 in internal units (= Z_freespace).
  // The generalized (conjugate-match) scattering parameter is subsequently renormalized
  // from this real reference to the true complex reference impedance Z_ref(ω).
  double GetExcitationRefResistance() const { return (std::abs(R) > 0.0) ? R : 1.0; }

  enum class Branch
  {
    TOTAL,
    R,
    L,
    C
  };
  std::complex<double> GetCharacteristicImpedance(double omega = 0.0,
                                                  Branch branch = Branch::TOTAL) const;

  double GetExcitationPower() const;
  double GetExcitationVoltage() const;

  std::complex<double> GetPower(GridFunction &E, GridFunction &B) const;
  std::complex<double> GetSParameter(GridFunction &E) const;
  std::complex<double> GetVoltage(GridFunction &E) const;
};

//
// A class handling lumped port boundaries and their postprocessing.
//
class LumpedPortOperator
{
private:
  // Mapping from port index to data structure containing port information and methods to
  // calculate circuit properties like voltage and current on lumped or multielement lumped
  // ports.
  std::map<int, LumpedPortData> ports;

  void SetUpBoundaryProperties(const std::map<int, config::LumpedPortData> &lumpedport,
                               const MaterialOperator &mat_op, const mfem::ParMesh &mesh);
  void PrintBoundaryInfo(const Units &units, const mfem::ParMesh &mesh);

public:
  LumpedPortOperator(const std::map<int, config::LumpedPortData> &lumpedport,
                     const Units &units, const MaterialOperator &mat_op,
                     const mfem::ParMesh &mesh);
  LumpedPortOperator(const IoData &iodata, const MaterialOperator &mat_op,
                     const mfem::ParMesh &mesh);

  // Access data structures for the lumped port with the given index.
  const LumpedPortData &GetPort(int idx) const;
  auto begin() const { return ports.begin(); }
  auto end() const { return ports.end(); }
  auto rbegin() const { return ports.rbegin(); }
  auto rend() const { return ports.rend(); }
  auto Size() const { return ports.size(); }

  // Returns array of lumped port attributes.
  mfem::Array<int> GetAttrList() const;
  mfem::Array<int> GetRsAttrList() const;
  mfem::Array<int> GetLsAttrList() const;
  mfem::Array<int> GetCsAttrList() const;

  // Add contributions to system matrices from lumped elements with nonzero inductance,
  // resistance, and/or capacitance.
  void AddStiffnessBdrCoefficients(double coeff, MaterialPropertyCoefficient &fb);
  void AddDampingBdrCoefficients(double coeff, MaterialPropertyCoefficient &fb);
  void AddMassBdrCoefficients(double coeff, MaterialPropertyCoefficient &fb);

  // Add contributions to the right-hand side source term vector for an incident field at
  // excited port boundaries, -U_inc/(iω) for the real version (versus the full -U_inc for
  // the complex one).
  void AddExcitationBdrCoefficients(int excitation_idx, SumVectorCoefficient &fb);
};

}  // namespace palace

#endif  // PALACE_MODELS_LUMPED_PORT_OPERATOR_HPP
