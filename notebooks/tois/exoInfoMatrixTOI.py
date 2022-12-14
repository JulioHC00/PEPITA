import numpy as np
import theano
import theano.tensor as tt
import exoplanet as xo
import matplotlib.pyplot as plt


class exoInfoMatrix:
    def __init__(self, exptime, oversample=10000):
        self.exptime = exptime
        self.oversample = oversample
        self._t = None,
        self._period = None
        self._t0 = None
        self._ror = None
        self._b = None
        self._u1 = None
        self._u2 = None
        self._rho_star_val = None
        self._r_star_val = None

        self.params = None

        self.legend = {
                "t"      : 0,
                "period" : 1,
                "t0"     : 2,
                "ror"    : 3,
                "b"      : 4,
                "u1"     : 5,
                "u2"     : 6,
                "rho_star" : 7,
                "r_star" : 8
                }

        self._has_deriv = False
        self._has_params = False
        self._has_priors = False

        self.deriv_matrix = None
        self.priors = None
        self.cov_matrix = None

    def set_data(self, time_array, period_val, t0_val, ror_val, b_val, u1_val, u2_val, rho_star_val, r_star_val):

        # Because changing data,
        # reset derivative matrix and require it to be calculated again

        self._has_deriv = False
        self.deriv_matrix = None
        self._has_fisher = False
        self.fisher_matrix = None

        self._has_params = True

        # Set values

        self._t = time_array
        self._period = period_val
        self._t0 = t0_val
        self._ror = ror_val
        self._b = b_val
        self._u1 = u1_val
        self._u2 = u2_val
        self._rho_star = rho_star_val
        self._r_star = r_star_val

        # Initialize scalar variables

        t = tt.dscalar()
        period = tt.dscalar()
        t0 = tt.dscalar()
        ror = tt.dscalar()
        b = tt.dscalar()
        u1 = tt.dscalar()
        u2 = tt.dscalar()
        rho_star = tt.dscalar()
        r_star = tt.dscalar()
        r = ror * r_star

        self.params = [t, period, t0, ror, b, u1, u2, rho_star, r_star]

        # Initialize star

        self.star = xo.LimbDarkLightCurve(u1, u2)

        # Initialize orbit

        self.orbit = xo.orbits.KeplerianOrbit(
                r_star=r_star,
                rho_star=rho_star,
                period=period,
                t0=t0,
                b=b
                )

        # Get light_curve function

        self.lc = tt.sum(
                self.star.get_light_curve(
                    orbit=self.orbit,
                    r=r,
                    t=[t],
                    texp=self.exptime/(3600.0*24.0),
                    oversample=self.oversample
                    )
                )

        self.val_and_grad_func = theano.function(
                self.params,
                [self.lc] + list(theano.grad(self.lc, self.params))
                )

    def _updateParams(self):
        # Call set data to update params

        self.set_data(
                time_array=self._t,
                period_val=self._period,
                t0_val=self._t0,
                ror_val=self._ror,
                b_val=self._b,
                u1_val=self._u1,
                u2_val=self._u2,
                rho_star_val=self._rho_star,
                r_star_val=self._r_star
                )

    def eval_point(self, tval):
        if not self._has_params:
            raise ValueError("Must define parameters first")

        return np.stack(
                self.val_and_grad_func(
                    tval,
                    self._period,
                    self._t0,
                    self._ror,
                    self._b,
                    self._u1,
                    self._u2,
                    self._rho_star,
                    self._r_star
                    )
                )

    def eval_deriv_matrix(self):
        if not self._has_params:
            raise ValueError("Must define parameters first")

        self.deriv_matrix = []
        for tval in self._t:
            result = self.eval_point(tval)
            self.deriv_matrix.append(result)

        self.deriv_matrix = np.array(self.deriv_matrix)

        self._has_deriv = True
        return self.deriv_matrix

    def plot_derivs(self, fig_ax=None):
        # If derivative matrix hasn't yet been calculated, do it
        if not self._has_deriv:
            self.eval_deriv_matrix()

        # Use figure if given
        if fig_ax is None:
            fig, ax = plt.subplots(len(self.deriv_matrix[1]), figsize=(10, 20))
        else:
            fig, ax = fig_ax

        for n, label in enumerate([
            "Flux",
            "dF/dt",
            "dF/dperiod",
            "dF/dt0",
            "dF/dror",
            "dF/b",
            "dF/du1",
            "dF/du2",
            "dF/drho_star",
            "dF/dr_star"
        ]):

            if n == 0:
                ax[n].plot(
                        self._t,
                        self.deriv_matrix[:, n],
                        label=str(self.exptime) + "s"
                        )
            else:
                ax[n].plot(
                        self._t,
                        self.deriv_matrix[:, n]
                        )

            ax[n].set_ylabel(label)
        ax[-1].set_xlabel("time [days]")
        ax[0].legend()

        return fig, ax

    def setExptime(self, exptime):
        self.exptime = exptime

        self._has_fisher = False
        self._has_deriv = False
        self._has_params = False

        self._updateParams()

    def set_priors(self, period_prior=np.nan, t0_prior=np.nan, ror_prior=np.nan, b_prior=np.nan, u1_prior=np.nan, u2_prior=np.nan, rho_star_prior=np.nan, r_star_prior=np.nan):
        diag = [
                period_prior,
                t0_prior,
                ror_prior,
                b_prior,
                u1_prior,
                u2_prior,
                rho_star_prior,
                r_star_prior
                ]

        self.priors = np.nan_to_num(np.diag(np.power(diag, -2)), 0)
        self._has_priors = True

        return self.priors

    def erase_priors(self):
        self.priors = None
        self._has_priors = False

    def eval_fisher(self, sigma):
        if not self._has_params:
            raise ValueError("Must define params first")

        if not self._has_deriv:
            self.eval_deriv_matrix()

        self.fisher_matrix = np.full(
                (len(self.params)-1, len(self.params)-1),
                np.nan
                )

        # -1 because not interested in time
        for i in range(0, len(self.params)-1):
            for j in range(0, len(self.params)-1):
                # +2 because we don't want the value of flux [0] 
                # or the value of the time derivative [1]
                self.fisher_matrix[i, j] = np.sum(
                        self.deriv_matrix[:, i+2] * self.deriv_matrix[:, j+2] * np.power(sigma, -2)
                        )

        if np.nan in self.fisher_matrix:
            raise ValueError("Something didn't work in the calculation")

        if self._has_priors:
            self.fisher_matrix = self.fisher_matrix + self.priors

        return self.fisher_matrix

    def eval_cov(self, sigma=None):
        # If Fisher matrix not calculated, calculate it
        if self._has_fisher is False:
            # Need sigma to calculate it
            if sigma is None:
                raise ValueError("Need a sigma")
            else:
                self.eval_fisher(sigma)
        # If it is calculated but a sigma is given, recalculate it
        elif sigma is not None:
            self.eval_fisher(sigma)

        self.cov_matrix = np.linalg.inv(self.fisher_matrix)

        if np.nan in self.cov_matrix:
            raise ValueError("Inverting the Fisher Matrix didn't work")

        return self.cov_matrix

    def get_in_transit(self):
        orbit = xo.orbits.KeplerianOrbit(
                t0=self._t0,
                period=self._period,
                b=self._b,
                rho_star=self._rho_star,
                r_star=self._r_star,
                ror=self._ror
                )
        in_transit = orbit.in_transit(
                self._t,
                r=self._ror * self._r_star,
                texp=self.exptime/(3600*24)
                ).eval()
        return in_transit

    def get_approx_transit_duration(self, n_points=10000):

        dur_t = np.linspace(-self._period, self._period, n_points)

        orbit = xo.orbits.KeplerianOrbit(
                t0=0,
                period=self._period,
                b=self._b,
                rho_star=self._rho_star,
                r_star=self._r_star,
                ror=self._ror
                )
        in_transit = orbit.in_transit(
                dur_t,
                r=self._ror * self._r_star,
                texp=self.exptime/(3600*24)
                ).eval()

        start = None
        end = None

        prev_val = None

        for i, val in enumerate(in_transit):
            if i == 0:
                prev_val = val
                continue

            next_val = prev_val + 1

            if next_val != val:
                if start is None:
                    start = val
                    prev_val = val
                    continue
                elif end is None:
                    end = prev_val
                    prev_val = val
                    break

            prev_val = val

        duration = dur_t[end] - dur_t[start]

        return duration

    def set_t(self, time_array):
        self._t = time_array

        self._has_deriv = False
        self._has_fisher = False

        self._updateParams()
