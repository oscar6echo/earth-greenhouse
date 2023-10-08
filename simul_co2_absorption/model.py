import numpy as np


class Model:
    """"""

    def __init__(self, temperature_model=None):
        """"""
        if temperature_model:
            self.temperature_model = temperature_model
            self.temperature = self.select_temperature_model(temperature_model)

        self.T0 = 288.2

    # ===================
    # BLACKBODY RADIATION
    # ===================

    @staticmethod
    def planck_function_1d(lambda_wavelength, T):
        """
        input unit      : m^-1
        output unit     : W m^-2 m^-1

        ref:
            https://fr.wikipedia.org/wiki/Loi_de_Planck#Exitance_%C3%A9nerg%C3%A9tique_spectrale
            https://www.tec-science.com/thermodynamics/temperature/different-forms-of-plancks-law/
        """
        # Planck's constant, J*s
        h = 6.62607015e-34

        # Speed of light, m/s
        c = 2.998e8

        # Boltzmann's constant, J/K
        kB = 1.380649e-23

        term1 = (2 * h * c**2) / lambda_wavelength**5
        term2 = np.exp((h * c) / (lambda_wavelength * kB * T)) - 1

        return np.pi * term1 / term2

    # ================
    # ATMOSPHERE MODEL
    # ================

    @staticmethod
    def pressure(z):
        """
        input unit      : m
        output unit     : N m^-2
        """
        # Pressure at sea level in Pa
        P0 = 101325

        # Scale height in m
        H = 8500

        return P0 * np.exp(-z / H)

    @staticmethod
    def temperature_uniform(z):
        """
        input unit      : m
        output unit     : K
        """
        T0 = 288.2
        return T0 * np.ones_like(z)

    @staticmethod
    def temperature_simple(z):
        """
        input unit      : m
        output unit     : K
        """
        # Temperature at sea level in K
        T0 = 288.2

        # Tropopause height in m
        z_trop = 11000

        # Temperature gradient in K/m
        Gamma = -0.0065

        T_trop = T0 + Gamma * z_trop

        affine = np.piecewise(
            z,
            [z < z_trop, z >= z_trop],
            [lambda z: T0 + Gamma * z, lambda z: T_trop],
        )
        return affine

    @staticmethod
    def temperature_US1976(z):
        """
        input unit      : m
        output unit     : K
        """
        # Convert altitude to km for easier comparisons
        z_km = z / 1000

        # Troposphere (0 to 11 km)
        T0 = 288.15
        z_trop = 11

        # Tropopause (11 to 20 km)
        T_tropopause = 216.65
        z_tropopause = 20

        # Stratosphere 1 (20 to 32 km)
        T_strat1 = T_tropopause
        z_strat1 = 32

        # Stratosphere 2 (32 to 47 km)
        T_strat2 = 228.65
        z_strat2 = 47

        # Stratopause (47 to 51 km)
        T_stratopause = 270.65
        z_stratopause = 51

        # Mesosphere 1 (51 to 71 km)
        T_meso1 = T_stratopause
        z_meso1 = 71

        # Mesosphere 2 (71 to ...)
        T_meso2 = 214.65

        affine = np.piecewise(
            z_km,
            [
                z_km < z_trop,
                (z_km >= z_trop) & (z_km < z_tropopause),
                (z_km >= z_tropopause) & (z_km < z_strat1),
                (z_km >= z_strat1) & (z_km < z_strat2),
                (z_km >= z_strat2) & (z_km < z_stratopause),
                (z_km >= z_stratopause) & (z_km < z_meso1),
                z_km >= z_meso1,
            ],
            [
                lambda z: T0 - 6.5 * z,
                lambda z: T_tropopause,
                lambda z: T_strat1 + 1 * (z - z_tropopause),
                lambda z: T_strat2 + 2.8 * (z - z_strat1),
                lambda z: T_stratopause,
                lambda z: T_meso1 - 2.8 * (z - z_stratopause),
                lambda z: T_meso2 - 2 * (z - z_meso1),
            ],
        )
        return affine

    def select_temperature_model(self, model):
        """"""
        if model == "simple":
            return Model.temperature_simple
        if model == "uniform":
            return Model.temperature_uniform
        if model == "US1976":
            return Model.temperature_US1976
        raise Exception("UNEXPECTED")

    def air_number_density(self, z):
        """
        input unit      : m
        output unit     : N m^-2 / N m = m^-3
        """
        # Boltzmann's constant, J/K
        kB = 1.380649e-23

        return Model.pressure(z) / (kB * self.temperature(z))

    # ==============
    # CO2 ABSORPTION
    # ==============

    @staticmethod
    def cross_section_co2(wavelength):
        """
        unit input      : m^-1
        unit output     : zero
        """
        # Band center in m
        LAMBDA_0 = 15e-6

        exponent = -22.5 - 24 * np.abs((wavelength - LAMBDA_0) / LAMBDA_0)
        sigma = 10**exponent
        return sigma
