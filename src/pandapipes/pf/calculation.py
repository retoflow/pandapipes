# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.pf.pipeflow_setup import (
    get_net_option, get_net_options, set_net_option, create_internal_results, write_internal_results
)

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Calculation:
    """
    Base class for one Newton-Raphson nonlinear solve (e.g. hydraulics, heat transfer,
    bidirectional). A subclass declares, as class attributes, what used to be passed
    around as parallel lists (``solver_vars``/``tols``/``pit_names``/``iter_name``) and
    implements :meth:`solve_step` to perform one linearized assemble-and-solve pass.

    Newton-Raphson iteration recap:
      1. Build the Jacobian df/dx at the current guess x
      2. Solve J @ dx = -f(x) (here: spsolve)
      3. Update x -= dx * alpha, repeat until the residual/variable changes fall below tol

    Class attributes to set on subclasses
    --------------------------------------
    MODE : str        identifies this calculation in logs/internal results (e.g. "hydraulics")
    ITER : str        net option name holding the max-iteration count (e.g. "max_iter_hyd")
    VARS : list[str]  names of the variables tracked for convergence (e.g. ["mdot", "p"])
    TOLS : list[str]  net option names holding the tolerance for each VARS entry
    PITS : list[str]  which pit ("branch"/"node") each VARS entry lives in
    COLS : list[int]  PIT column index for each VARS entry (used for damping-fallback writes)
    """

    MODE = None
    ITER = None
    VARS = []
    TOLS = []
    PITS = []
    COLS = []

    def solve_step(self, net):
        """
        Perform one linearized solve (assemble Jacobian, spsolve, update pit values).

        :param net: the pandapipesNet to solve on
        :return: (results, residual, filtered) where results is a flat list of
                 [var1_new, var1_old, var2_new, var2_old, ...] (one pair per solver_var,
                 in the same order as ``solver_vars``), residual is the raw load-vector
                 residual, and filtered contains a row-index array (or None) per solver_var
                 selecting which pit rows that var's damping-fallback should write back to.
        """
        raise NotImplementedError

    def tols(self, net):
        return list(get_net_options(net, *self.TOLS))

    def run(self, net):
        """Run the Newton-Raphson loop until convergence or ITER's max iterations."""
        max_iter, nonlinear_method, tol_res = get_net_options(
            net, self.ITER, "nonlinear_method", "tol_res"
        )
        tols = self.tols(net)
        niter = 0
        errors = {var: [] for var in self.VARS}
        create_internal_results(net)
        residual_norm = None

        while not net.converged and niter < max_iter:
            logger.debug("niter %d" % niter)
            results, residual, filtered = self.solve_step(net)
            residual_norm = np.max(np.abs(residual))
            logger.debug("residual: %s" % residual_norm.round(4))

            results = np.array(results, object)
            pos = np.arange(len(self.VARS) * 2)
            vals_new = results[pos[::2]]
            vals_old = results[pos[1::2]]
            for var, val_new, val_old in zip(self.VARS, vals_new, vals_old):
                dval = val_new - val_old
                errors[var].append(np.max(np.abs(dval)) if len(dval) else 0)

            self._finalize_iteration(net, niter, residual_norm, nonlinear_method, errors, tols,
                                     tol_res, vals_old, filtered)
            niter += 1

        write_internal_results(net, **errors)
        kwargs = {
            'residual_norm_%s' % self.MODE: residual_norm,
            'iterations_%s' % self.MODE: niter,
        }
        write_internal_results(net, **kwargs)
        self._log_final_results(net, niter, residual_norm, tols)

    def _finalize_iteration(self, net, niter, residual_norm, nonlinear_method, errors, tols, tol_res,
                            vals_old, filtered):
        if nonlinear_method == "automatic":
            errors_increased = set_damping_factor(net, niter, errors)
            logger.debug("alpha: %s" % get_net_option(net, "alpha"))
            for error_increased, val, pit, col, f in zip(
                errors_increased, vals_old, self.PITS, self.COLS, filtered
            ):
                if error_increased:
                    if f is None:
                        # todo: not working in bidirectional mode as bidirectional is not
                        #  distinguishing between hydraulics and heat transfer active pit
                        net["_active_pit"][pit][:, col] = val
                    else:
                        net["_active_pit"][pit][f, col] = val
            if get_net_option(net, "alpha") != 1:
                net.converged = False
                return
        elif nonlinear_method != "constant":
            logger.warning("No proper nonlinear method chosen. Using constant settings.")
        converged = True
        for var, error, tol in zip(self.VARS, errors.values(), tols):
            converged = error[niter] <= tol
            if not converged:
                break
            logger.debug("error_%s: %s" % (var, error[niter]))
        net.converged = converged and residual_norm <= tol_res

    def _log_final_results(self, net, niter, residual_norm, tols):
        logger.debug("--------------------------------------------------------------------------------")
        if not net.converged:
            logger.debug(
                "Maximum number of iterations reached but %s solver did not converge." % self.MODE)
            logger.debug("Norm of residual: %s" % residual_norm)
        else:
            logger.debug("Calculation completed. Preparing results...")
            logger.debug("Converged after %d iterations." % niter)
            logger.debug("Norm of residual: %s" % residual_norm)
            for var, tol in zip(self.VARS, tols):
                logger.debug("tolerance for %s: %s" % (var, tol))


def set_damping_factor(net, niter, errors):
    """
    Set the value of the damping factor (factor for the newton step width) from current results.

    :param net: the net for which to perform the pipeflow
    :type net: pandapipesNet
    :param niter:
    :type niter:
    :param errors: an array containing the current residuals of all field variables solved for
    :return: No Output.

    EXAMPLE:
        set_damping_factor(net, niter, [error_p, error_v])
    """
    error_increased = []
    for error in errors.values():
        error_increased.append(error[niter] > error[niter - 1])
    current_alpha = get_net_option(net, "alpha")
    if np.all(error_increased):
        set_net_option(net, "alpha", current_alpha / 10 if current_alpha >= 0.1 else current_alpha)
    else:
        set_net_option(net, "alpha", current_alpha * 10 if current_alpha <= 0.1 else 1.0)
    return error_increased
