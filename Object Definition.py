# pip install numpy
# pip install numpy --upgrade --ignore-installed
# pip install wheel
# pip install --upgrade pip
# pip install --upgrade setuptools
# pip install thermo
# pip install gas_dynamics
# pip install ambiance

import numpy as np
import matplotlib.pyplot as plt
import thermo.mixture, matplotlib.widgets, json, scipy.integrate, scipy.optimize, scipy.interpolate
import gas_dynamics as gd
from ambiance import Atmosphere

class AeroHeatingAnalysis:
    """
    Object used to run aerodynamic heating analyses
    Inputs
    -------
    tangent_ogive : TangentOgive
        TangentOgive object specifying the nosecone geometry
    trajectory_data : dict or pandas DataFrame
        Data on the rocket's trajectory, needs to have "pos_i", "vel_i" and "time".
    rocket : Rocket
        Rocket object. It's only needed to get LaunchSite data for coordinate transforms.
    fixed_wall_temperature : bool
        If True, the wall temperature is fixed to its starting value. Otherwise a simple model is used to model its temperature change.
    starting_temperature : float, optional
        Temperature that the nose cone starts with (K). Defaults to None, in which case the rocket starts with the local atmospheric temperature.
    nosecone_mass : float, optional
        Mass of the nosecone (kg) - used to find its heat capacity. Only needed if you're modelling variable temperatures
    specific_heat_capacity : float, optional
        Specific heat capacity of the nosecone (J/kg/K). Defaults to an approximate value for aluminium.
    turbulent_transition_Rex : float, optional
        Local Reynolds number at which the boundary layer transition from laminar to turbulent. Defaults to 7.5e6.
    Attributes
    ----------
    i : int
        Current index in the timestep array.
    tangent_ogive : TangentOgive
        A TangentOgive object containing nosecone geometry data
    trajectory_data : dict or pandas DataFrame
        Contains trajectory data.
    trajectory_dict : dict
        Same data trajectory_data, but converted to a dictionary if it wasn't already one.
    rocket : Rocket
        Rocket object used to run the simulation. Its only purpose is to get LaunchSite data for coordinate transforms.
    fixed_wall_temperature : bool
        If True, the wall temperature is fixed to its starting value. Otherwise a simple model is used to model its temperature change.
    turbulent_transition_Rex : float, optional
        Local Reynolds number at which the boundary layer transition from laminar to turbulent. Defaults to 7.5e6.
    heat_capacity : float
        Heat capacity the nosecone (J/K). Is caclulated by doing specific_heat_capacity*nosecone_mass.
    M : numpy ndarray
        Local Mach number at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    P : numpy ndarray
        Local static pressure at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Te : numpy ndarray
        Temperature at the edge of the boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Tw : numpy ndarray
        Wall temperature at each timestep. Local static pressure at each station and timestep. Has dimensions (N) where N is the length of the "time" array in trajectory_dict.
    Trec_lam : numpy ndarray
        Adiabatic wall tempearture (also known as the 'recovery temperature') for a laminar boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Trec_turb : numpy ndarray
        Adiabatic wall tempearture (also known as the 'recovery temperature') for a turbulent boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Hstar_function : numpy ndarray
        Function that needs to be integrated to get H*(x), as defined in the NASA documents. This is not equal to H*(x) itself. This is stored because it's needed for integration. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    Rex : numpy ndarray
        Local Reynolds number at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    q_lam : numpy ndarray
        Heat transfer rate (W/m^2) with a laminar boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    q_turb : numpy ndarray
        Heat transfer rate (W/m^2) with a turbulent boundary layer, at each station and timestep. Has dimensions (15, N) where N is the length of the "time" array in trajectory_dict.
    q0_hemispherical_nose : numpy ndarray
        Heat transfer rate (W/m^2) at the stagnation point at each timestep, if we were using a hemispherical nosecone. Has dimensions (N) where N is the length of the "time" array in trajectory_dict.
    """

    def __init__(
        self,
        tangent_ogive,
        trajectory_data,
        fixed_wall_temperature=True,
        starting_temperature=None,
        nosecone_mass=None,
        specific_heat_capacity=900,
        turbulent_transition_Rex=7.5e6,
    ):

        self.tangent_ogive = tangent_ogive
        self.trajectory_data = trajectory_data
        self.turbulent_transition_Rex = turbulent_transition_Rex
        self.fixed_wall_temperature = fixed_wall_temperature

        # Convert the data into a dictionary if it isn't already one:
        if type(self.trajectory_data) is dict:
            self.trajectory_dict = self.trajectory_data
        else:
            self.trajectory_dict = self.trajectory_data.to_dict(orient="list")

        # If we want a variable wall temperature:
        if self.fixed_wall_temperature == False:
            assert (
                nosecone_mass != None
            ), "You need to input a value for the nosecone mass if you want to model a variable wall temperature"
            self.heat_capacity = specific_heat_capacity * nosecone_mass

        # Timestep index:
        self.i = 0

        # Arrays to store the fluid properties at each discretised point on the nose cone (1 to 15), and at each timestep
        self.M = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Local Mach number
        self.P = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Local pressure

        # Initialise the wall temperature:
        if starting_temperature == None:
            starting_temperature = Atmosphere(
                self.trajectory_dict["altitude"][0]
            ).temperature[
                0
            ]  # Assume the nose cone starts with ambient temperature

        if self.fixed_wall_temperature == True:
            self.Tw = np.full(len(self.trajectory_dict["time"]), starting_temperature)
        else:
            self.Tw = np.full(len(self.trajectory_dict["time"]), float("NaN"))
            self.Tw[0] = starting_temperature

        self.Te = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Temperature at the edge of the boundary layer
        self.Tstar = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # T* as defined in the paper
        self.Trec_lam = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Temperature corresponding to hrec_lam
        self.Trec_turb = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Temperature corresponding to hrec_turb
        self.Hstar_function = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # This array is used to minimise number of calculations for the integration needed in H*(x)
        self.Rex = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Local Reynolds number

        # Arrays to store the heat transfer rates
        self.q_lam = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Laminar boundary layer
        self.q_turb = np.full(
            [15, len(self.trajectory_dict["time"])], float("NaN")
        )  # Turbunulent boundary layer
        self.q0_hemispherical_nose = np.full(
            len(self.trajectory_dict["time"]), float("NaN")
        )  # At the stagnation point for a rocket with a hemispherical nose cone - used as a reference point

    def step(self, print_style=None):
        """
        Performs one step of the aerodynamic analysis, starting from the current value of self.i.
        Inputs:
        -------
        print_style : str
            Options for print style:
            None - nothing is printed
            "FORTRAN" - same output as the examples in https://ntrs.nasa.gov/citations/19730063810, printing in the Imperial units listed
            "metric" - outputs useful data in metric units
        """

        # Get altitude:
        alt = self.trajectory_dict["altitude"][self.i]

        if alt < -5004:  # hacky way to fix out of bound altitudes for ambience
            alt = 5004
        elif alt > 81020:
            alt = 81020

        # Get ambient conditions:
        Pinf = Atmosphere(alt).pressure[0]
        Tinf = Atmosphere(alt).temperature[0]
        rhoinf = Atmosphere(alt).density[0]

        # Get the freestream velocity and Mach number
        Vinf = self.trajectory_dict["speed"][self.i]
        Minf = Vinf / Atmosphere(alt).speed_of_sound[0]

        if print_style == "FORTRAN":
            print("")
            print("FREE STREAM CONDITIONS")
            print(
                "XMINF={:<10}     VINFY={:.4e}         GAMINF={:.4e}       RHOINF={:.4e}".format(
                    0, 3.28084 * Vinf, gamma_air(), 0.00194032 * rhoinf
                )
            )
            print(
                "HINFY={:.4e}     PINF ={:.4e} (ATMOS) PINFY ={:.4e} (PSF)".format(
                    0.000429923 * cp_air() * Tinf, Pinf / 101325, 0.0208854 * Pinf
                )
            )
            print("TINFY={:.4e}".format(Tinf))
            print("")

        if print_style == "metric":
            print("")
            print("SUBCRIPTS:")
            print("0 or STAG  : At the stagnation point for a hemispherical nose")
            print(
                "REF        : At 'reference' enthalpy and local pressure - I think this is like an average-ish boundary layer enthalpy"
            )
            print(
                "REC        : At 'recovery' enthalpy and local pressure - I believe this is the wall enthalpy at which no heat transfer takes place"
            )
            print("W          : At the wall temperature and local pressure")
            print("INF        : Freestream (i.e. atmospheric) property")
            print("LAM        : With a laminar boundary layer")
            print("TURB       : With a turbulent boundary layer")
            print("")
            print("FREE STREAM CONDITIONS")
            print(
                "ALT ={:06.2f} km    TINF={:06.2f} K    PINF={:06.2f} kPa    RHOINF={:06.2f} kg/m^3".format(
                    alt / 1000, Tinf, Pinf / 1000, rhoinf
                )
            )
            print("VINF={:06.2f} m/s   MINF={:06.2f}".format(Vinf, Minf))
            print("")

        # Check if we're supersonic - if so we'll have a shock wave
        if Minf > 1:
            # For an oblique shock (tangent ogive nose cone)
            oblique_shock_data = oblique_shock(
                self.tangent_ogive.theta, Minf, Tinf, Pinf, rhoinf
            )  # MS, TS, PS, rhoS, beta
            oblique_MS = oblique_shock_data[0]
            oblique_TS = oblique_shock_data[1]
            oblique_PS = oblique_shock_data[2]
            oblique_rhoS = oblique_shock_data[3]

            oblique_P0S = p2p0(oblique_PS, oblique_MS)
            oblique_T0S = T2T0(oblique_TS, oblique_MS)
            oblique_rho0S = rho2rho0(oblique_rhoS, oblique_MS)

            # For a normal shock (hemispherical nosecone)
            normal_shock_data = normal_shock(Minf)
            normal_MS = normal_shock_data[0]
            normal_PS = normal_shock_data[1] * Pinf
            normal_TS = normal_shock_data[2] * Tinf
            normal_rhoS = normal_shock_data[3] * rhoinf

            normal_P0S = p2p0(normal_PS, normal_MS)
            normal_T0S = T2T0(normal_TS, normal_MS)
            normal_rho0S = rho2rho0(normal_rhoS, normal_MS)

            # Stagnation point heat transfer rate for a hemispherical nosecone
            Pr0 = Pr_air(normal_T0S, normal_P0S)
            h0 = cp_air() * normal_T0S
            hw = cp_air() * self.Tw[self.i]
            mu0 = mu_air(normal_T0S, normal_P0S)
            rhow0 = normal_P0S / (
                R_air() * self.Tw[self.i]
            )  # p = rho * R * T (ideal gas)
            muw0 = mu_air(self.Tw[self.i], normal_P0S)

            RN = 0.3048  # Let RN = 1 ft = 0.3048m, as it recommends using that as a reference value (although apparently it shouldn't matter?)
            dVdx0 = (2 ** 0.5) / RN * ((normal_P0S - Pinf) / normal_rho0S) ** 0.5

            # Equation (29) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
            # Note that the equation only works in Imperial units, and requires you to specify density in slugs/ft^3, which is NOT lbm/ft^3
            # Metric density (kg/m^3) --> Imperial density (slugs/ft^3): Multiply by 0.00194032
            # Metric viscosity  (Pa s) --> Imperial viscosity (lbf sec/ft^2): Divide by 47.880259
            # Metric enthalpy (J/kg/s) --> Imperial enthalpy (Btu/lbm): Multiply by 0.000429923
            # Note that 'g', the acceleration of gravity, is equal to 32.1740 ft/s^2
            self.q0_hemispherical_nose[self.i] = (
                0.76
                * 32.1740
                * Pr0 ** (-0.6)
                * (0.00194032 * rhow0 * muw0 / 47.880259) ** 0.1
                * (0.00194032 * normal_rho0S * mu0 / 47.880259) ** 0.4
                * (0.000429923 * h0 - 0.000429923 * hw)
                * dVdx0 ** 0.5
            )

            # Now convert from Imperial heat transfer rate (Btu/ft^2/s) --> Metric heat transfer rate (W/m^2): Divide by 0.000088055
            self.q0_hemispherical_nose[self.i] = (
                self.q0_hemispherical_nose[self.i] / 0.000088055
            )

            if print_style == "FORTRAN":
                print("")
                print("STAGNATION POINT DATA FOR SPHERICAL NOSE")
                print(
                    "HREF0 ={:<10}     TREF0 ={:<10}   VISCR0={:<10}   TKREF0={:<10}".format(
                        0, 0, 0, 0
                    )
                )
                print(
                    "ZREF0 ={:<10}     PRREF0={:<10}   CPREF0={:<10}   RHOR0 ={:<10}".format(
                        0, 0, 0, 0
                    )
                )
                print(
                    "CPCVR0={:.4e}     RN    ={:.4e}   T0    ={:.4e}".format(
                        gamma_air(), RN / 0.3048, normal_T0S
                    )
                )
                print(
                    "P0    ={:.4e}     RHO0  ={:.4e}   SR0   ={:<10}   TK0   ={:<10}".format(
                        normal_P0S / 101325, 0.00194032 * normal_rho0S, 0, 0
                    )
                )
                print(
                    "VISC0 ={:.4e}     DVDX0 ={:.4e}   Z0    ={:<10}   CP0   ={:.4e}".format(
                        mu0 / 47.880259, dVdx0, 0, 0.000429923 * cp_air()
                    )
                )
                print(
                    "A0    ={:<10}     TW0   ={:.4e}   VISCW0={:.4e}   HW0   ={:.4e}".format(
                        0, self.Tw[self.i], muw0 / 47.880259, 0.000429923 * hw
                    )
                )
                print("")
                print(
                    "CPW0  ={:.4e}     PR0   ={:.4e}".format(
                        0.000429923 * cp_air(), Pr0
                    )
                )
                print(
                    "QSTPT ={:.4e}     = NOSE STAGNATION POINT HEAT RATE".format(
                        0.000088055 * self.q0_hemispherical_nose[self.i]
                    )
                )
                print(
                    "H0    ={:.4e}     HT    ={:<10}   RHOW0={:.4e}".format(
                        0.000429923 * h0, 0, 0.00194032 * rhow0
                    )
                )
                print("")

            if print_style == "metric":
                print("")
                print("STAGNATION POINT DATA FOR SPHERICAL NOSE")
                print(
                    "P0   ={:06.2f} kPa    T0   ={:06.2f} K       RHO0={:06.2f} kg/m^3".format(
                        normal_P0S / 1000, normal_T0S, normal_rho0S
                    )
                )
                print(
                    "TW   ={:06.2f} K      RHOW0={:06.2f} kg/m^3".format(
                        self.Tw[self.i], rhow0
                    )
                )
                print(
                    "QSTAG={:06.2f} kW/m^2".format(
                        self.q0_hemispherical_nose[self.i] / 1000
                    )
                )
                print("")

            # Prandtl-Meyer expansion (only possible for supersonic flow):
            if oblique_MS > 1:
                # Get values at the nose cone tip:
                nu1 = prandtl_meyer(oblique_MS)
                theta1 = self.tangent_ogive.theta

                # Prandtl-Meyer expansion from post-shockwave to each discretised point
                for j in range(10):
                    # Angle between the flow and the horizontal:
                    theta = self.tangent_ogive.theta - self.tangent_ogive.dtheta * j

                    # Across a +mu characteristic: nu1 + theta1 = nu2 + theta2
                    nu = nu1 + theta1 - theta

                    # Check if we've exceeded nu_max, in which case we can't turn the flow any further
                    if nu > (np.pi / 2) * (
                        np.sqrt((gamma_air() + 1) / (gamma_air() - 1)) - 1
                    ):
                        raise ValueError(
                            "Cannot turn flow any further at nosecone position {}, exceeded nu_max. Flow will have seperated (which is not yet implemented). Stopping simulation.".format(
                                j + 1
                            )
                        )

                    # Record the local Mach number and pressure
                    self.M[j, self.i] = nu2mach(nu)
                    self.P[j, self.i] = p02p(oblique_P0S, self.M[j, self.i])

                # Expand for the last few points using Equations (1) - (6) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                for j in [10, 11, 12, 13, 14]:
                    if j >= 10 and j <= 13:
                        self.P[j, self.i] = (Pinf + self.P[j - 1, self.i]) / 2
                    elif j == 14:
                        self.P[j, self.i] = Pinf
                    self.M[j, self.i] = pressure_ratio_to_mach(
                        self.P[j, self.i] / oblique_P0S
                    )

                # Now deal with the heat transfer itself
                for j in range(15):
                    # Edge of boundary layer temperature - i.e. flow temperature post-shock and after Prandtl-Meyer expansion
                    self.Te[j, self.i] = T02T(oblique_T0S, self.M[j, self.i])

                    # Enthalpies
                    he = cp_air() * self.Te[j, self.i]

                    # Prandtl numbers and specific heat capacities
                    Pre = Pr_air(self.Te[j, self.i], self.P[j, self.i])

                    #'Reference' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3
                    hstar = (he + hw) / 2 + 0.22 * (Pre ** 0.5) * (h0 - hw)
                    self.Tstar[j, self.i] = hstar / cp_air()
                    Prstar = Pr_air(self.Tstar[j, self.i], self.P[j, self.i])

                    #'Recovery' values, as defined in https://arc.aiaa.org/doi/pdf/10.2514/3.62081 page 3 - I think these are the wall enthalpies for zero heat transfer
                    hrec_lam = he * (1 - Prstar ** (1 / 2)) + h0 * (Prstar ** (1 / 2))
                    hrec_turb = he * (1 - Prstar ** (1 / 3)) + h0 * (Prstar ** (1 / 3))
                    self.Trec_lam[j, self.i] = hrec_lam / cp_air()
                    self.Trec_turb[j, self.i] = hrec_turb / cp_air()

                    # Get H*(x) - I'm not sure about if I did the integral bit right
                    rhostar0 = normal_P0S / (R_air() * self.Tstar[j, self.i])
                    mustar0 = mu_air(T=self.Tstar[j, self.i], P=normal_P0S)

                    rhostar = self.P[j, self.i] / (R_air() * self.Tstar[j, self.i])
                    mustar = mu_air(T=self.Tstar[j, self.i], P=self.P[j, self.i])

                    r = self.tangent_ogive.r(j + 1)
                    V = (
                        gamma_air() * R_air() * T02T(oblique_T0S, self.M[j, self.i])
                    ) ** 0.5 * self.M[j, self.i]

                    self.Hstar_function[j, self.i] = (rhostar * mustar * V * r ** 2) / (
                        rhostar0 * mustar0 * Vinf
                    )

                    # Get the integral bit of H*(x) using trapezium rule
                    integral = np.trapz(
                        self.Hstar_function[0 : j + 1, self.i],
                        self.tangent_ogive.S_array[0 : j + 1],
                    )

                    # Equation (17) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    if j == 0:
                        Hstar = np.inf
                    else:
                        Hstar = (
                            (rhostar * V * r) / (rhostar0 * Vinf) / (integral ** 0.5)
                        )

                    # Get H*(0) - Equation (18) - it seems like the (x) values in Equation (18) are actually (0) values
                    # It seems weird that they still included them though, since they end up cancelling out
                    # Hstar0 = ( ((2*rhostar/rhostar0)*dVdx0 )/(Vinf * mustar/mustar0) )**0.5 * (2)**0.5
                    Hstar0 = (2 * dVdx0 / Vinf) ** 0.5 * (2 ** 0.5)

                    # Laminar heat transfer rate, normalised by that for a hemispherical nosecone
                    kstar = k_air(T=self.Tstar[j, self.i], P=self.P[j, self.i])
                    kstar0 = k_air(T=self.Tstar[j, self.i], P=normal_P0S)
                    Cpw = cp_air()
                    Cpw0 = cp_air()

                    # Equation (13) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081 - wasn't sure which 'hrec' to use here but I think it's the laminar one
                    qxq0_lam = (kstar * Hstar * (hrec_lam - hw) * Cpw0) / (
                        kstar0 * Hstar0 * (h0 - hw) * Cpw
                    )

                    # Now we can find the absolute laminar heat transfer rates, in W/m^2
                    self.q_lam[j, self.i] = (
                        qxq0_lam * self.q0_hemispherical_nose[self.i]
                    )

                    # Turbulent heat transfer rate - using Equation (20) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    # THERE LOOKS LIKE THERE'S A TYPO IN THE EQUATION! It should be {1 - (Pr*)^(1/3)}*he, not 1 - {(Pr*)^(1/3)}*he
                    # For the correct version see page Eq (20) on page 14 of https://ntrs.nasa.gov/citations/19730063810

                    # Note that the equation only works in Imperial units, and requires you to specify density in slugs/ft^3, which is NOT lbm/ft^3
                    # Density (kg/m^3) --> Density (slugs/ft^3): Multiply by 0.00194032
                    # Viscosity  (Pa s) --> Viscosity (lbf sec/ft^2): Divide by 47.880259
                    # Enthalpy (J/kg/s) --> Enthalpy (Btu/lbm): Multiply by 0.000429923
                    # Thermal conductivity (W/m/K) --> Thermal conductivity (Btu/ft/s/K): Multiply by 0.000288894658
                    # Velocity (m/s) ---> Velocity (ft/s): Multiply by 3.28084
                    # Note that 'g', the acceleration of gravity, is equal to 32.1740 ft/s^2

                    Cpstar0 = cp_air()
                    if j == 0:
                        self.q_turb[j, self.i] = np.inf
                    else:
                        self.q_turb[j, self.i] = (
                            0.03
                            * 32.1740 ** (1 / 3)
                            * (2 ** 0.2)
                            * (0.000288894658 * kstar) ** (2 / 3)
                            * (0.00194032 * rhostar * 3.28084 * V) ** 0.8
                            * (
                                (1 - Prstar ** (1 / 3)) * 0.000429923 * he
                                + Prstar ** (1 / 3) * 0.000429923 * h0
                                - 0.000429923 * hw
                            )
                        ) / (
                            (mustar / 47.880259) ** (7 / 15)
                            * (0.000429923 * Cpstar0) ** (2 / 3)
                            * (3.28084 * self.tangent_ogive.S(j + 1)) ** 0.2
                        )

                    # Now convert from Imperial heat transfer rate (Btu/ft^2/s) --> Metric heat transfer rate (W/m^2): Divide by 0.000088055
                    self.q_turb[j, self.i] = self.q_turb[j, self.i] / 0.000088055

                    # Local Reynolds number, Re(x), using Equation (25) from https://arc.aiaa.org/doi/pdf/10.2514/3.62081
                    rho = self.P[j, self.i] / (
                        R_air() * self.Te[j, self.i]
                    )  # Ideal gas law: p = rho*R*T, rho = p/(RT)
                    mu = mu_air(self.Te[j, self.i], self.P[j, self.i])
                    self.Rex[j, self.i] = rho * V * self.tangent_ogive.S_array[j] / mu

                    # FORTRAN style output:
                    if print_style == "FORTRAN":
                        print("")
                        print(
                            "WALL, REFERENCE AND EXTERNAL-TO-BOUNDARY-LAYER FLOW PROPERTIES AT STATION = {}".format(
                                j + 1
                            )
                        )
                        print(
                            "HW    ={:.4e}    CPW   ={:.4e}   HREFX={:<10}    PRREFX={:<10}".format(
                                0.000429923 * hw, 0.000429923 * Cpw, 0, 0
                            )
                        )
                        print(
                            "TKREFX={:<10}    VISCRX={:<10}   RHORX={:<10}    TREFX ={:<10}".format(
                                0, 0, 0, 0
                            )
                        )
                        print(
                            "ZREFX ={:<10}    CPCVRX={:<10}   PX   ={:.4e}    TX    ={:.4e}".format(
                                0, 0, self.P[j, self.i] / 101325, self.Te[j, self.i]
                            )
                        )
                        print(
                            "TKX   ={:<10}    VISCX ={:.4e}   PRX  ={:.4e}    ZX    ={:<10}".format(
                                0,
                                mu_air(self.Te[j, self.i], self.P[j, self.i])
                                / 47.880259,
                                Pre,
                                0,
                            )
                        )
                        print(
                            "SRX   ={:<10}    HX    ={:.4e}   VX   ={:.4e}    CPCVX ={:.4e}".format(
                                0, 0.000429923 * he, 3.28084 * V, gamma_air()
                            )
                        )
                        print(
                            "AAX   ={:<10}    RHOX  ={:.4e}   XM   ={:<10}    CPX   ={:.4e}".format(
                                0,
                                0.00194032 * rho02rho(oblique_rho0S, self.M[j, self.i]),
                                0,
                                0.000429923 * cp_air(),
                            )
                        )
                        print("")
                        print(
                            "X = {:.3f}".format(3.28084 * self.tangent_ogive.S_array[j])
                        )
                        print(
                            "QLAM={:.3f}     QTURB={:.3f}     QLAM/QSTAG={:.3f}     QTURB/QSTAG={:.3f}".format(
                                0.000088055 * self.q_lam[j, self.i],
                                0.000088055 * self.q_turb[j, self.i],
                                self.q_lam[j, self.i]
                                / self.q0_hemispherical_nose[self.i],
                                self.q_turb[j, self.i]
                                / self.q0_hemispherical_nose[self.i],
                            )
                        )
                        print("")

                    if print_style == "metric":
                        print("")
                        print(
                            "WALL, REFERENCE AND EXTERNAL-TO-BOUNDARY-LAYER FLOW PROPERTIES AT STATION = {}".format(
                                j + 1
                            )
                        )
                        print("X   ={:.6} m".format(self.tangent_ogive.S_array[j]))
                        print(
                            "PX  ={:.6} kPa        TX   ={:06.2f} K        RHOX      ={:06.2f} kg/m^3".format(
                                self.P[j, self.i] / 1000,
                                self.Te[j, self.i],
                                rho02rho(oblique_rho0S, self.M[j, self.i]),
                            )
                        )
                        print(
                            "TW  ={:.6} K          TREF ={:06.2f} K        TREC_LAM  ={:06.2f} K     TREC_TURB  ={:06.2f} K".format(
                                self.Tw[self.i],
                                self.Tstar[j, self.i],
                                hrec_lam / cp_air(),
                                hrec_lam / cp_air(),
                            )
                        )
                        print(
                            "QLAM={:.6} kW/m^2     QTURB={:06.2f} kW/m^2   QLAM/QSTAG={:06.2f}       QTURB/QSTAG={:06.2f}".format(
                                self.q_lam[j, self.i] / 1000,
                                self.q_turb[j, self.i] / 1000,
                                self.q_lam[j, self.i]
                                / self.q0_hemispherical_nose[self.i],
                                self.q_turb[j, self.i]
                                / self.q0_hemispherical_nose[self.i],
                            )
                        )
                        print("")

                # Simple lumped mass model for increase in wall temperature:
                if self.fixed_wall_temperature == False:

                    # Points 12 - 15 are below the bottom of the nosecone, so we'll ignore them.
                    qdot_array = np.zeros(11)
                    qdotr_array = np.zeros(11)  # Local q * local nosecone radius

                    for j in range(len(qdot_array)):
                        # The nose tip (Station 1) has q = infinity, so we'll ignore it for now
                        if j == 0:
                            pass

                        # Check if we have a laminar or turbulent boundary layer at each point:
                        else:
                            if self.Rex[j, self.i] < self.turbulent_transition_Rex:
                                # Laminar boundary layer
                                qdot_array[j] = self.q_lam[j, self.i]
                            else:
                                # Turbulent boundary layer
                                qdot_array[j] = self.q_turb[j, self.i]

                        # qdot(x) * r(x)
                        qdotr_array[j] = qdot_array[j] * self.tangent_ogive.r(j + 1)

                    # Set the heat transfer rates at Station 1 (the nose tip) to be the same as that at Station 2
                    qdot_array[0] = qdot_array[1]
                    qdotr_array[0] = qdotr_array[1]

                    # Integrate to get the total heat transferred
                    Qdot_tot = (
                        2
                        * np.pi
                        * np.trapz(qdotr_array, self.tangent_ogive.S_array[:11])
                    )  # Qdot = ∫qdot dA = ∫qdot (2πrdx) = 2π∫qdot r dx
                    Q_tot = Qdot_tot * (
                        self.trajectory_dict["time"][self.i + 1]
                        - self.trajectory_dict["time"][self.i]
                    )  # Q = ∫Qdot dt, using left Riemann sum

                    # Get the change in temperature, and add it to the current temperature
                    dT = Q_tot / self.heat_capacity
                    self.Tw[self.i + 1] = self.Tw[self.i] + dT

                else:
                    # If using a fixed wall temperature:
                    self.Tw[self.i + 1] = self.Tw[self.i]

            else:
                if print_style != None:
                    print(
                        "Subsonic flow post-shock (Minf = {:.2f}, MS = {:.2f}), skipping step number {}".format(
                            Minf, oblique_MS, self.i
                        )
                    )

                # Wall temperature doesn't change:
                self.Tw[self.i + 1] = self.Tw[self.i]

        else:
            if print_style != None:
                print(
                    "Subsonic freestream flow, skipping step number {}".format(self.i)
                )

            # Wall temperature doesn't change:
            self.Tw[self.i + 1] = self.Tw[self.i]

        self.i = self.i + 1

    def run(self, number_of_steps=None, starting_index=0, print_style="minimal"):
        """
        Runs the simulation for a set number of steps, starting from starting_index. Updates all of its attributes as it does so.
        Inputs:
        -------
        number_of_steps : int
            Number of steps you would like to perform. Defaults to None, in which case the programme goes through all available data in trajectory_data.
        starting_index : int
            The index in the "time" array that you want to start from. Note that you should always start from 0 if you're using a variable wall temperature (previous wall temperatures will affect the heat transfer rate, and hence affect future wall temperatures).
        print_style : str
            Options for print_style:
            None - Nothing is printed
            "minimal" - Minimalistic printing, printing progress every 10%, and the max. and min. wall temperature if a variable wall temperature is used.
        """
        if number_of_steps == None:
            number_of_steps = len(self.trajectory_dict["time"]) - 1 - starting_index

        if self.fixed_wall_temperature == False and starting_index != 0:
            print(
                "WARNING: You should normally start the simulation from starting_index = 0 if you're using a variable wall temperature. Doing otherwise may give inaccurate results or errors."
            )

        self.i = starting_index
        counter = 0

        while self.i - starting_index <= number_of_steps:
            if print_style == "minimal":
                if (self.i - starting_index) % (int(number_of_steps / 100)) == 0:
                    print("{:.1f}% complete, i = {}".format(counter / 100 * 100, self.i))
                    counter = counter + 1

            self.step()

        if self.fixed_wall_temperature == False and print_style == "minimal":
            print(
                "Maximum wall temparature = {:.4f} °C".format(
                    np.nanmax(self.Tw) - 273.15
                )
            )
            print(
                "Minimum wall temperature = {:.4f} °C".format(
                    np.nanmin(self.Tw) - 273.15
                )
            )

