### Reference state
using DocStringExtensions
using Unitful; using CLIMA.UnitAnnotations #FIXME
import CLIMA.DGmethods: space_unit, time_unit, mass_unit #FIXME
export NoReferenceState, HydrostaticState, IsothermalProfile, LinearTemperatureProfile

"""
    ReferenceState

Reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

vars_state(m::ReferenceState    , FT) = @vars()
vars_gradient(m::ReferenceState , FT) = @vars()
vars_diffusive(m::ReferenceState, FT) = @vars()
vars_aux(m::ReferenceState, FT) = @vars()
atmos_init_aux!(::ReferenceState, ::AtmosModel, aux::Vars, geom::LocalGeometry) = nothing

space_unit(::ReferenceState) = NoUnits
time_unit(::ReferenceState) = NoUnits

"""
    NoReferenceState <: ReferenceState

No reference state used
"""
struct NoReferenceState <: ReferenceState end



"""
    HydrostaticState{P,T} <: ReferenceState

A hydrostatic state specified by a temperature profile and relative humidity.
"""
struct HydrostaticState{P,F} <: ReferenceState
  temperatureprofile::P
  relativehumidity::F
end

vars_aux(m::HydrostaticState, FT) = @vars begin
  ρ::U(FT, u"kg/m^3")
  p::U(FT, u"Pa")
  T::U(FT, u"K")
  ρe::U(FT, u"J/m^3")
  ρq_tot::U(FT, u"kg/m^3")
end


function atmos_init_aux!(m::HydrostaticState{P,F}, atmos::AtmosModel, aux::Vars, geom::LocalGeometry) where {P,F}
  FT = eltype(aux)
  T,p = m.temperatureprofile(atmos.orientation, aux)
  aux.ref_state.T = T
  aux.ref_state.p = p
  aux.ref_state.ρ = ρ = p/(FT(R_d)*T)
  q_vap_sat = q_vap_saturation(T, ρ)
  aux.ref_state.ρq_tot = ρq_tot = ρ * m.relativehumidity * q_vap_sat

  q_pt = PhasePartition(ρq_tot / (mass_unit(atmos) / space_unit(atmos)^3))
  aux.ref_state.ρe = ρ * internal_energy(T, q_pt)

  e_kin = F(0) * (space_unit(atmos)^2 / time_unit(atmos)^2)
  e_pot = gravitational_potential(atmos.orientation, aux)
  aux.ref_state.ρe = ρ*total_energy(e_kin, e_pot, T, q_pt)
end



"""
    TemperatureProfile

Specifies the temperature profile for a reference state.

Instances of this type are required to be callable objects with the following signature

    T,p = (::TemperatureProfile)(orientation::Orientation, aux::Vars)

where `T` is the temperature (in K), and `p` is the pressure (in hPa).
"""
abstract type TemperatureProfile
end

"""
    IsothermalProfile{F} <: TemperatureProfile

A uniform temperature profile.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct IsothermalProfile{F} <: TemperatureProfile
  "temperature (K)"
  T::U(F,:temp)
end

function (profile::IsothermalProfile)(orientation::Orientation, aux::Vars)
  FT = eltype(aux)
  p = FT(MSLP) * exp(-gravitational_potential(orientation, aux)/(FT(R_d)*profile.T))
  return (profile.T, p)
end

"""
    LinearTemperatureProfile{F} <: TemperatureProfile

A temperature profile which decays linearly with height `z`, until it reaches a minimum specified temperature.

```math
T(z) = \\max(T_{\\text{surface}} − Γ z, T_{\\text{min}})
```

# Fields

$(DocStringExtensions.FIELDS)
"""
struct LinearTemperatureProfile{F} <: TemperatureProfile
  "minimum temperature (K)"
  T_min::U(F,:temp)
  "surface temperature (K)"
  T_surface::U(F,:temp)
  "lapse rate (K/m)"
  Γ::U(F,:lincond)
end

function (profile::LinearTemperatureProfile)(orientation::Orientation, aux::Vars)
  z = altitude(orientation, aux)
  T = max(profile.T_surface - profile.Γ*z, profile.T_min)

  p = MSLP * (T/profile.T_surface)^(grav/(R_d*profile.Γ))
  if T == profile.T_min
    z_top = (profile.T_surface - profile.T_min) / profile.Γ
    H_min = R_d * profile.T_min / grav
    p *= exp(-(z-z_top)/H_min)
  end
  return (T, p)
end
