# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.system_index import HydraulicSystemIndex, HeatSystemIndex, ComponentRegistry
from pandapipes.pf.calculation import Calculation
from pandapipes.pf.pipeflow_setup import (
    get_net_option, init_options,
    get_lookup, create_lookups, initialize_pit, reduce_pit,
    set_user_pf_options, init_all_result_tables, identify_active_nodes_branches,
    check_infeed_number, compute_infeed_nodes, PipeflowNotConverged
)
from pandapipes.pf.result_extraction import extract_all_results, extract_results_active_pit

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def set_logger_level_pipeflow(level):
    """
    Set logger level from outside to reduce/extend pipeflow() printout.
    :param level: levels according to 'logging' (i.e. DEBUG, INFO, WARNING, ERROR and CRITICAL)
    :type level: str
    :return: No output

    EXAMPLE:
        set_logger_level_pipeflow('WARNING')

    """
    logger.setLevel(level)


def pipeflow(net, sol_vec=None, **kwargs):
    """
    The main method used to start the solver to calculate the velocity, pressure and temperature\
    distribution for a given net. Different options can be entered for \\**kwargs, which control\
    the solver behaviour (see function :func:`init_options` for more information).

    :param net: The pandapipes net for which to perform the pipeflow
    :type net: pandapipesNet
    :param sol_vec: Initializes the start values for the heating network calculation
    :type sol_vec: numpy.ndarray, default None
    :param kwargs: A list of options controlling the solver behaviour
    :return: No output

    :Example:
        >>> pipeflow(net, mode="hydraulics")

    """
    init_pipeflow(net, **kwargs)

    net.converged = False
    calculation_mode = get_net_option(net, "mode")
    calculate_hydraulics = calculation_mode in ["hydraulics", 'sequential']
    calculate_heat = calculation_mode in ["heat", 'sequential']
    calculate_bidrect = calculation_mode == "bidirectional"


    # TODO: This is not necessary in every time step, but we need the result! The result of the
    #       connectivity check is currently not saved anywhere!
    # cannot be moved to calculate_hydraulics as the active node/branch hydraulics lookup is also required to
    # determine the active node/branch heat transfer lookup
    identify_active_nodes_branches(net)

    if calculation_mode == 'heat':
        use_given_hydraulic_results(net, sol_vec)

    if not (calculate_hydraulics | calculate_heat | calculate_bidrect):
        raise UserWarning("No proper calculation mode chosen.")
    elif calculate_bidrect:
        bidirectional(net)
    else:
        if calculate_hydraulics:
            hydraulics(net)
        if calculate_heat:
            heat_transfer(net)

    extract_all_results(net, calculation_mode)


def init_pipeflow(net, **kwargs):
    """
    Inputs & initialization of variables: physical constants/options, result tables,
    lookups and the internal PIT (pandapipes internal tables) arrays.

    :param net: The pandapipes net for which to perform the pipeflow
    :type net: pandapipesNet
    :param kwargs: A list of options controlling the solver behaviour
    :return: No output
    """
    # Init physical constants and options
    init_options(net, **kwargs)

    # init result tables
    init_all_result_tables(net)

    create_lookups(net)
    initialize_pit(net)


def use_given_hydraulic_results(net, sol_vec):
    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    if not net.user_pf_options["hyd_flag"]:
        raise UserWarning("Converged flag not set. Make sure that hydraulic calculation "
                          "results are available.")
    else:
        node_pit[:, IdxNode.PINIT] = sol_vec[:len(node_pit)]
        branch_pit[:, IdxBranch.MDOTINIT] = sol_vec[len(node_pit):]


class HydraulicCalculation(Calculation):
    """Newton-Raphson solve for pressure/mdot (see :func:`solve_hydraulics`)."""
    MODE = 'hydraulics'
    ITER = 'max_iter_hyd'
    VARS = ['mdot', 'p', 'mdotslack']
    TOLS = ['tol_m', 'tol_p', 'tol_m']
    PITS = ['branch', 'node', 'node']
    COLS = [IdxBranch.MDOTINIT, IdxNode.PINIT, IdxNode.MDOTSLACKINIT]

    def solve_step(self, net):
        return solve_hydraulics(net)


class ThermalCalculation(Calculation):
    """Newton-Raphson solve for branch outlet / node temperature (see :func:`solve_temperature`)."""
    MODE = 'heat'
    ITER = 'max_iter_therm'
    VARS = ['Tout', 'T']
    TOLS = ['tol_T', 'tol_T']
    PITS = ['branch', 'node']
    COLS = [IdxBranch.TOUTINIT, IdxNode.TINIT]

    def solve_step(self, net):
        return solve_temperature(net)


class BidirectionalCalculation(Calculation):
    """Newton-Raphson solve alternating hydraulics and heat transfer (see :func:`solve_bidirectional`)."""
    MODE = 'bidirectional'
    ITER = 'max_iter_bidirect'
    VARS = ['mdot', 'p', 'TOUT', 'T']
    TOLS = ['tol_m', 'tol_p', 'tol_T', 'tol_T']
    PITS = ['branch', 'node', 'branch', 'node']
    COLS = [IdxBranch.MDOTINIT, IdxNode.PINIT, IdxBranch.TOUTINIT, IdxNode.TINIT]

    def solve_step(self, net):
        return solve_bidirectional(net)


def bidirectional(net):
    net.converged = False
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()
    BidirectionalCalculation().run(net)
    if net.converged:
        set_user_pf_options(net, hyd_flag=True)
    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)
    if not net.converged:
        raise PipeflowNotConverged("The bidrectional calculation did not converge to a solution.")


def hydraulics(net):
    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    net.converged = False
    reduce_pit(net, mode="hydraulics")
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()
    HydraulicCalculation().run(net)
    if net.converged:
        set_user_pf_options(net, hyd_flag=True)
        rerun_hydraulics(net)

    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)

    if not net.converged:
        msg = "The hydraulic calculation did not converge to a solution."
        raise PipeflowNotConverged(msg)
    extract_results_active_pit(net, mode="hydraulics")


def heat_transfer(net):
    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    net.converged = False
    identify_active_nodes_branches(net, False)
    reduce_pit(net, mode="heat_transfer")
    if net.fluid.is_gas:
        logger.info("Caution! Temperature calculation does currently not affect hydraulic "
                    "properties!")
    ThermalCalculation().run(net)

    if net.converged:
        rerun_heat_transfer(net)

    if not net.converged:
        msg = "The heat transfer calculation did not converge to a solution."
        raise PipeflowNotConverged(msg)
    extract_results_active_pit(net, mode="heat_transfer")


def solve_bidirectional(net):
    reduce_pit(net, mode="hydraulics")
    res_hyd, residual_hyd, filter_hyd = solve_hydraulics(net)
    extract_results_active_pit(net, mode="hydraulics")
    identify_active_nodes_branches(net, False)
    reduce_pit(net, mode="heat_transfer")
    res_heat, residual_heat, filter_heat = solve_temperature(net)
    extract_results_active_pit(net, mode="heat_transfer")
    residual = np.concatenate([residual_hyd, residual_heat])
    res = res_hyd + res_heat
    filtered = filter_hyd + filter_heat
    return res, residual, filtered


def solve_hydraulics(net):
    """
    Create and solve the linearized system of equations (based on a jacobian in form of a scipy
    sparse matrix and a load vector in form of a numpy array) in order to calculate the hydraulic
    magnitudes (pressure and velocity) for the network nodes and branches.

    :param net: The pandapipesNet for which to solve the hydraulic matrix
    :type net: pandapipesNet
    :return:

    """
    options = net["_options"]

    connected_restarted = True
    while connected_restarted:
        branch_pit = net["_active_pit"]["branch"]
        node_pit = net["_active_pit"]["node"]
        connected_restarted = _restart_connectivity_check(net)

    sys_idx = HydraulicSystemIndex(node_pit, branch_pit)
    eq_registry = ComponentRegistry()

    for comp in net['component_list']:
        comp.register_hydraulic_equations(net, branch_pit, node_pit, sys_idx, eq_registry)

    sz = sys_idx.size()
    rows, cols, data, epsilon = eq_registry.assemble(sz)
    jacobian = csr_matrix((data, (rows, cols)), shape=(sz, sz))

    m_init_old = branch_pit[:, IdxBranch.MDOTINIT].copy()
    p_init_old = node_pit[:, IdxNode.PINIT].copy()
    slack_nodes = np.where(node_pit[:, IdxNode.NODE_TYPE] == IdxNode.P)[0]
    msl_init_old = node_pit[slack_nodes, IdxNode.MDOTSLACKINIT].copy()

    # Diagonal (Jacobi) scaling: D-columns can be many orders of magnitude larger/smaller
    # than mdot/p columns (terms scale like ~1/D**4..1/D**6), which makes spsolve badly
    # over/under-correct D on the first step. Scale rows/cols by 1/sqrt(|diagonal|) so all
    # variables are comparable before solving, then undo the scaling on the result - same
    # solution, better-conditioned linear system.
    diag = np.abs(jacobian.diagonal())
    diag[diag < 1e-30] = 1.0
    scale = 1.0 / np.sqrt(diag)
    scale_mat = diags(scale)
    jacobian_scaled = scale_mat @ jacobian @ scale_mat
    epsilon_scaled = scale * epsilon

    x_scaled = spsolve(jacobian_scaled, epsilon_scaled)
    x = scale * x_scaled

    branch_pit[:, IdxBranch.MDOTINIT] -= x[len(node_pit):len(node_pit) + len(branch_pit)] * options["alpha"]
    node_pit[:, IdxNode.PINIT] -= x[:len(node_pit)] * options["alpha"]
    node_pit[slack_nodes, IdxNode.MDOTSLACKINIT] -= x[len(node_pit) + len(branch_pit):]

    filtered = [None, None, slack_nodes]

    return [branch_pit[:, IdxBranch.MDOTINIT], m_init_old, node_pit[:, IdxNode.PINIT], p_init_old,
            node_pit[slack_nodes, IdxNode.MDOTSLACKINIT], msl_init_old], epsilon, filtered

def rerun_hydraulics(net):
    rerun = False
    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]
    branch_lookups = get_lookup(net, "branch", "from_to_active_hydraulics")
    for comp in net['component_list']:
        rerun |= comp.rerun_hydraulics(net, branch_pit, node_pit, branch_lookups, options)
    if rerun:
        extract_results_active_pit(net, 'hydraulics')
        identify_active_nodes_branches(net)
        hydraulics(net)

def rerun_heat_transfer(net):
    rerun = False
    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]
    branch_lookups = get_lookup(net, "branch", "from_to_active_heat_transfer")
    for comp in net['component_list']:
        rerun |= comp.rerun_hydraulics(net, branch_pit, node_pit, branch_lookups, options)
    if rerun:
        extract_results_active_pit(net, 'heat_transfer')
        identify_active_nodes_branches(net, False)
        heat_transfer(net)

def _restart_connectivity_check(net):
    nodes_connected = get_lookup(net, "node", "active_hydraulics")
    branches_connected = get_lookup(net, "branch", "active_hydraulics")
    rows_nodes = np.arange(net["_pit"]["node"].shape[0])[nodes_connected]
    rows_branches = np.arange(net["_pit"]["branch"].shape[0])[branches_connected]
    active_node_pit = net["_active_pit"]["node"]
    active_branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_pit"]["node"][rows_nodes, IdxNode.ACTIVE]
    branch_pit = net["_pit"]["branch"][rows_branches, IdxBranch.ACTIVE]
    mask_diff_node = active_node_pit[:, IdxNode.ACTIVE] != node_pit
    mask_diff_branch = active_branch_pit[:, IdxBranch.ACTIVE]  != branch_pit
    if np.any(mask_diff_node) | np.any(mask_diff_branch):
        net["_pit"]["node"][rows_nodes, IdxNode.ACTIVE] = active_node_pit[:, IdxNode.ACTIVE]
        net["_pit"]["node"][rows_nodes, IdxNode.NODE_TYPE] = active_node_pit[:, IdxNode.NODE_TYPE]
        net["_pit"]["branch"][rows_branches, IdxBranch.ACTIVE] = active_branch_pit[:, IdxBranch.ACTIVE]
        net["_pit"]["branch"][rows_branches, IdxBranch.BRANCH_TYPE] = active_branch_pit[:, IdxBranch.BRANCH_TYPE]
        identify_active_nodes_branches(net, True)
        reduce_pit(net, mode='hydraulics')
        return True
    return False


def solve_temperature(net):
    """
    This function contains the procedure to build and solve a linearized system of equation based on
    an underlying net and the necessary graph data structures. Temperature values are calculated.
    Returned are the solution vectors for the new iteration, the original solution vectors and a
    vector containing component indices for the system matrix entries

    :param net: The pandapipesNet for which to solve the temperature matrix
    :type net: pandapipesNet
    :return: branch_pit

    """

    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]

    # Negative velocity values are turned to positive ones (including exchange of from_node and
    # to_node for temperature calculation
    branch_pit[:, IdxBranch.FROM_NODE_T_SWITCHED] = branch_pit[:, IdxBranch.MDOTINIT] < -2e-11

    node_pit[:, IdxNode.INFEED] = False
    compute_infeed_nodes(branch_pit, node_pit)

    sys_idx = HeatSystemIndex(node_pit, branch_pit)
    eq_registry = ComponentRegistry()

    for comp in net['component_list']:
        comp.register_thermal_equations(net, branch_pit, node_pit, sys_idx, eq_registry)

    t_init_old = node_pit[:, IdxNode.TINIT].copy()
    t_out_old = branch_pit[:, IdxBranch.TOUTINIT].copy()
    filtered = [None, None]
    if not check_infeed_number(node_pit):
        return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], np.array([
            np.nan]), filtered

    sz = sys_idx.size()
    rows, cols, data, epsilon = eq_registry.assemble(sz)
    jacobian = csr_matrix((data, (rows, cols)), shape=(sz, sz))

    x = spsolve(jacobian, epsilon)

    if np.any(np.isnan(x)):
        return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], np.array([
            np.nan]), filtered

    node_pit[:, IdxNode.TINIT] -= x[:len(node_pit)] * options["alpha"]
    branch_pit[:, IdxBranch.TOUTINIT] -= x[len(node_pit):] * options["alpha"]

    return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], epsilon, filtered
