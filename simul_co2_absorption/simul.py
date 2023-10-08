from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from .model import Model


class Simul(Model):
    """"""

    def __init__(
        self,
        z_max=80_000,
        z_delta=10,
        lambda_min=0.1e-6,
        lambda_max=100e-6,
        lambda_delta=0.01e-6,
    ):
        """"""

        # self.co2_fraction = co2_fraction
        self.z_max = z_max
        self.z_delta = z_delta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_delta = lambda_delta

        s2 = f"lammbda min              = {lambda_min/1e-6:<4.2f} x 10⁻⁶"
        s3 = f"lammbda max              = {lambda_max/1e-6:<4.0f} x 10⁻⁶"
        s4 = f"lammbda delta            = {lambda_delta/1e-6:<4.0f} x 10⁻⁶"
        s5 = f"alt min                  = {0/1e-6:.0f}"
        s6 = f"alt max                  = {z_max:,}"
        s7 = f"alt delta                = {z_delta:,}"

        print(s2)
        print(s3)
        print(s4)
        print()
        print(s5)
        print(s6)
        print(s7)

    # =============================
    # RADIATIVE TRANSFER SIMULATION
    # =============================

    def calc_earth_flux(self):
        """"""
        super().__init__()

        lambda_range = np.arange(
            self.lambda_min,
            self.lambda_max,
            self.lambda_delta,
        )

        self.earth_flux = (
            self.planck_function_1d(lambda_range, self.T0) * self.lambda_delta
        )
        self.total_earth = self.earth_flux.sum()
        msg = f"Total earth surface flux in wavelength range: {self.total_earth:.2f} W m⁻²"
        print(msg)

    def calc_radiative_transfer(
        self,
        temperature_model=None,
        co2_fraction=None,
    ):
        """"""
        super().__init__(temperature_model)
        self.co2_fraction = co2_fraction

        s0 = f"temperature model        = {temperature_model}"
        s1 = f"co2 fraction             = {co2_fraction/1e-6:<3.0f} x 10⁻⁶"
        print(s0)
        print(s1)

        t0 = timer()

        # altitude
        self.z_range = np.arange(
            0,
            self.z_max,
            self.z_delta,
        )
        # wavelength
        self.lambda_range = np.arange(
            self.lambda_min,
            self.lambda_max,
            self.lambda_delta,
        )

        # init
        self.upward_flux = np.zeros((len(self.z_range), len(self.lambda_range)))
        self.optical_thickness = np.zeros((len(self.z_range), len(self.lambda_range)))

        print(f"nb layers = {len(self.z_range):,}")

        # boundary condition : vertical earth radiation per m⁻² at z=0 for all wavelengths
        flux_in = self.earth_flux

        for i, z in enumerate(self.z_range):
            if i % 100 == 0:
                print(i, end=" ")

            # co2 density
            n_co2 = self.air_number_density(z) * self.co2_fraction

            #  absorption coefficient
            kappa = self.cross_section_co2(self.lambda_range) * n_co2

            # compute absorded and emitted flux within layer
            self.optical_thickness[i, :] = kappa * self.z_delta
            absorbed_flux = np.minimum(kappa * self.z_delta * flux_in, flux_in)
            emitted_flux = (
                self.optical_thickness[i, :]
                * self.planck_function_1d(self.lambda_range, self.temperature(z))
                * self.lambda_delta
            )
            self.upward_flux[i, :] = flux_in - absorbed_flux + emitted_flux

            # flux leaving layer is flux entering layer above
            flux_in = self.upward_flux[i, :]

        print()

        self.total_atmostphere = self.upward_flux[-1, :].sum()
        msg = f"Total atmosphere flux in wavelength range: {self.total_atmostphere:.2f} W m⁻²"
        print(msg)

        t1 = timer()
        print(f"runtime = {t1-t0:.2f} s")

    def calc_scenarii(
        self,
        temperature_model=None,
    ):
        """"""
        print("\n--- earth flux")
        self.calc_earth_flux()

        print("\n--- scenario 1")
        co2_fraction = 280e-6
        self.calc_radiative_transfer(
            temperature_model,
            co2_fraction,
        )

        self.co2_fraction_1 = self.co2_fraction
        self.upward_flux_1 = self.upward_flux.copy()
        self.optical_thickness_1 = self.optical_thickness.copy()
        self.total_atmostphere_1 = self.total_atmostphere.copy()

        print("\n--- scenario 2")
        co2_fraction *= 2
        self.calc_radiative_transfer(
            temperature_model,
            co2_fraction,
        )

        self.co2_fraction_2 = self.co2_fraction
        self.upward_flux_2 = self.upward_flux.copy()
        self.optical_thickness_2 = self.optical_thickness.copy()
        self.total_atmostphere_2 = self.total_atmostphere.copy()

    # =============================
    # RADIATIVE TRANSFER SIMULATION
    # =============================

    def plot(self):
        """"""
        lambda_delta = self.lambda_range[1] - self.lambda_range[0]

        plt.figure(figsize=(12, 6))

        # top of atmosphere spectrum
        # blackbody spectrum at T0
        plt.plot(
            1e6 * self.lambda_range,
            self.planck_function_1d(self.lambda_range, self.T0) / 1e6,
            # self.earth_flux / lambda_delta / 1e6,
            "--k",
            label=f"black body spectrum T={self.T0:.1f} K",
        )

        # blackbody spectrum at T so that T matches zero co2
        if self.temperature_model == "simple":
            T2 = 216
            plt.plot(
                1e6 * self.lambda_range,
                self.planck_function_1d(self.lambda_range, T2) / 1e6,
                "--b",
                label=f"black body spectrum T={T2:.1f} K",
            )

        # atmosphere flux scenario 1
        f = self.co2_fraction_1 * 1e6
        plt.plot(
            1e6 * self.lambda_range,
            self.upward_flux_1[-1, :] / lambda_delta / 1e6,
            "-g",
            label=f"(1) CO2 fraction = {f:.1f} 10⁻⁶",
        )

        # atmosphere flux scenario 2
        f = self.co2_fraction_2 * 1e6
        plt.plot(
            1e6 * self.lambda_range,
            self.upward_flux_2[-1, :] / lambda_delta / 1e6,
            "-r",
            label=f"(2) CO2 fraction = {f:.1f} 10⁻⁶",
        )

        # diff between scenarii
        plt.fill_between(
            1e6 * self.lambda_range,
            self.upward_flux_1[-1, :] / lambda_delta / 1e6,
            self.upward_flux_2[-1, :] / lambda_delta / 1e6,
            color="yellow",
            alpha=0.9,
        )

        plt.xlabel("wavelength (µm)")
        plt.ylabel("vertical spectral luminance (W m⁻² μm⁻¹)")
        plt.xlim(0, 50)
        plt.ylim(0, 30)
        plt.grid(True)
        plt.legend()

        tm = self.temperature_model
        ef = self.total_earth
        af1 = self.total_atmostphere_1
        af2 = self.total_atmostphere_2
        diff = af1 - af2

        title = "\n".join(
            [
                f"temperature model={tm}",
                f"total earth flux (W m⁻²): surface={ef:.1f}, atmosphere(1) = {af1:.1f}, atmosphere(2) = {af2:.1f}, diff = {diff:.1f}",
            ]
        )
        plt.title(title)

        plt.show()
