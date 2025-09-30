import numpy as np

from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected, \
    _sum_by_group
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density, get_branch_real_eta, \
    get_branch_cp


def calculate_derivatives_hydraulic(net, branch_pit, node_pit, options):
    """
    Function which creates derivatives.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit:
    :type branch_pit:
    :param node_pit:
    :type node_pit:
    :param options:
    :type options:
    :return: No Output.
    """
    fluid = get_fluid(net)
    gas_mode = fluid.is_gas
    friction_model = options["friction_model"]
    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    eta = get_branch_real_eta(fluid, node_pit, branch_pit)

    # Darcy Friction factor: lambda
    lambda_, re = calc_lambda(
        branch_pit[:, IdxBranch.MDOTINIT], eta, branch_pit[:, IdxBranch.D],
        branch_pit[:, IdxBranch.K], gas_mode, friction_model, branch_pit[:, IdxBranch.LENGTH],
        options, branch_pit[:, IdxBranch.AREA])
    der_lambda = calc_der_lambda(branch_pit[:, IdxBranch.MDOTINIT], eta,
                                 branch_pit[:, IdxBranch.D], branch_pit[:, IdxBranch.K],
                                 friction_model, lambda_, branch_pit[:, IdxBranch.AREA])
    branch_pit[:, IdxBranch.RE] = re
    branch_pit[:, IdxBranch.LAMBDA] = lambda_

    from_nodes = branch_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, IdxBranch.TO_NODE].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = \
        get_derived_values(node_pit, from_nodes, to_nodes, options["use_numba"])

    if not gas_mode:
        if options["use_numba"]:
            from pandapipes.pf.derivative_toolbox_numba import derivatives_hydraulic_incomp_numba \
                as derivatives_hydraulic_incomp
        else:
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_incomp_np_branches \
                as derivatives_hydraulic_incomp_branches
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_incomp_np_nodes \
                as derivatives_hydraulic_incomp_nodes

        f, df_dp_f, df_dp_t, df_dm, df_dmslack_f, df_dmslack_t = \
            derivatives_hydraulic_incomp_branches(branch_pit, der_lambda,
                                                  der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho)

        branch_pit[:, IdxBranch.F_B] = f
        branch_pit[:, IdxBranch.DF_DM_B] = df_dm
        branch_pit[:, IdxBranch.DF_DP_F_B] = df_dp_f
        branch_pit[:, IdxBranch.DF_DP_T_B] = df_dp_t
        branch_pit[:, IdxBranch.DF_DMSLACK_F_B] = df_dmslack_f
        branch_pit[:, IdxBranch.DF_DMSLACK_T_B] = df_dmslack_t

        f1_f, f1_t, df1_dm_f, df1_dm_t, df1_dp, df1_dmslack, \
        f2_f, f2_t, df2_dm_f, df2_dm_t, df2_dp, df2_dmslack = \
            derivatives_hydraulic_incomp_nodes(branch_pit, der_lambda)

        branch_pit[:, IdxBranch.F1_F_N] = f1_f
        branch_pit[:, IdxBranch.F1_T_N] = f1_t
        branch_pit[:, IdxBranch.DF1_DM_F_N] = df1_dm_f
        branch_pit[:, IdxBranch.DF1_DM_T_N] = df1_dm_t
        node_pit[:, IdxNode.DF1_DP_N] = df1_dp
        node_pit[:, IdxNode.DF1_DMSLACK_N] = df1_dmslack
        
        branch_pit[:, IdxBranch.F2_F_N] = f2_f
        branch_pit[:, IdxBranch.F2_T_N] = f2_t
        branch_pit[:, IdxBranch.DF2_DM_F_N] = df2_dm_f
        branch_pit[:, IdxBranch.DF2_DM_T_N] = df2_dm_t
        node_pit[:, IdxNode.DF2_DP_N] = df2_dp
        node_pit[:, IdxNode.DF2_DMSLACK_N] = df2_dmslack

    else:
        if options["use_numba"]:
            from pandapipes.pf.derivative_toolbox_numba import derivatives_hydraulic_comp_numba \
                as derivatives_hydraulic_comp, calc_medium_pressure_with_derivative_numba as \
                calc_medium_pressure_with_derivative
        else:
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_comp_np \
                as derivatives_hydraulic_comp, calc_medium_pressure_with_derivative_np as \
                calc_medium_pressure_with_derivative
        p_m, der_p_m, der_p_m1 = calc_medium_pressure_with_derivative(p_init_i_abs, p_init_i1_abs)
        rho_n = np.full(len(branch_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m)
        # TODO: this might not be required
        der_comp = fluid.get_der_compressibility() * der_p_m
        der_comp1 = fluid.get_der_compressibility() * der_p_m1
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1 = (
            derivatives_hydraulic_comp(
            node_pit, branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference,
            comp_fact, der_comp, der_comp1, rho, rho_n))

    branch_pit[:, IdxBranch.LOAD_VEC_BRANCHES] = load_vec
    branch_pit[:, IdxBranch.JAC_DERIV_DM] = df_dm
    branch_pit[:, IdxBranch.JAC_DERIV_DP] = df_dp
    branch_pit[:, IdxBranch.JAC_DERIV_DP1] = df_dp1
    branch_pit[:, IdxBranch.LOAD_VEC_NODES_FROM] = load_vec_nodes_from
    branch_pit[:, IdxBranch.LOAD_VEC_NODES_TO] = load_vec_nodes_to
    branch_pit[:, IdxBranch.JAC_DERIV_DM_NODE] = df_dm_nodes


def calculate_derivatives_thermal(net, branch_pit, node_pit, _):
    fluid = get_fluid(net)
    cp = get_branch_cp(fluid, node_pit, branch_pit)
    m_init_i = np.abs(branch_pit[:, IdxBranch.MDOTINIT])
    m_init_i1 = np.abs(branch_pit[:, IdxBranch.MDOTINIT])
    from_nodes = get_from_nodes_corrected(branch_pit)
    to_nodes = get_to_nodes_corrected(branch_pit)
    t_init_i = node_pit[from_nodes, IdxNode.TINIT]
    t_init_i1 = branch_pit[:, IdxBranch.TOUTINIT]
    t_init_n = node_pit[:, IdxNode.TINIT]
    cp_i = fluid.get_heat_capacity(t_init_i)
    cp_i1 = fluid.get_heat_capacity(t_init_i1)
    cp_n = fluid.get_heat_capacity(t_init_n)
    t_amb = branch_pit[:, IdxBranch.TEXT]
    length = branch_pit[:, IdxBranch.LENGTH]
    alpha = branch_pit[:, IdxBranch.ALPHA] * np.pi * branch_pit[:, IdxBranch.D]
    tl = branch_pit[:, IdxBranch.TL]
    qext = branch_pit[:, IdxBranch.QEXT]
    infeed_node = None

    node_pit[:, IdxNode.LOAD_T] = (node_pit[:, IdxNode.LOAD] * cp_n * t_init_n +
                                   node_pit[:, IdxNode.MDOTSLACKINIT] * cp_n * t_init_n)
    node_pit[:, IdxNode.JAC_DERIV_DT_LOAD] = - node_pit[:, IdxNode.LOAD] * cp_n
    node_pit[:, IdxNode.JAC_DERIV_DT_SLACK] = - node_pit[:, IdxNode.MDOTSLACKINIT] * cp_n

    branch_pit[:, IdxBranch.JAC_DERIV_DT_NODE] = - m_init_i * cp_i
    branch_pit[:, IdxBranch.JAC_DERIV_DTOUT_NODE] = m_init_i1 * cp_i1
    branch_pit[:, IdxBranch.LOAD_VEC_N_FROM_T] = m_init_i1 * t_init_i * cp_i
    branch_pit[:, IdxBranch.LOAD_VEC_N_TO_T] = m_init_i1 * t_init_i1 * cp_i1

    if get_net_option(net, "transient"):
        rho = get_branch_real_density(fluid, node_pit, branch_pit)
        area = branch_pit[:, IdxBranch.AREA]
        tvor = branch_pit[:, IdxBranch.T_OUT_OLD]
        delta_t = get_net_option(net, "dt")

        branch_pit[:, IdxBranch.LOAD_VEC_B_T] = (
                rho * area * cp * (t_init_i1 - tvor) * (1 / delta_t) * length
                + cp * m_init_i * (-t_init_i + t_init_i1 - tl)
                - alpha * (t_amb - t_init_i1) * length + qext
        )

        branch_pit[:, IdxBranch.JAC_DERIV_DT] = - cp * m_init_i
        branch_pit[:, IdxBranch.JAC_DERIV_DTOUT] = rho * area * cp / delta_t * length + cp * m_init_i + alpha

        branches_active_ht = get_lookup(net, "branch", "active_heat_transfer")
        branches_zero_fl = get_lookup(net, "branch", "zero_flow")[branches_active_ht]
        if np.any(branches_zero_fl):
            # TODO: maybe replace this statement with a component lookup
            zero_length = np.isclose(branch_pit[:, IdxBranch.LENGTH], 0, rtol=1e-6, atol=1e-10)
            mask = zero_length & branches_zero_fl
            if np.any(mask):
                branch_pit[mask, IdxBranch.LOAD_VEC_B_T] = (
                        rho[mask] * area[mask] * cp[mask] * (t_init_i1[mask] - tvor[mask]) * (1 / delta_t)
                        - alpha[mask] * (t_amb[mask] - t_init_i1[mask]) + qext[mask]
                )
                branch_pit[mask, IdxBranch.JAC_DERIV_DT] = 0
                branch_pit[mask, IdxBranch.JAC_DERIV_DTOUT] = (rho[mask] * area[mask] * cp[mask] / delta_t +
                                                     alpha[mask])

        nodes_active_ht = get_lookup(net, "node", "active_heat_transfer")
        nodes_zero_fl = get_lookup(net, "node", "zero_flow")[nodes_active_ht]

        if np.any(nodes_zero_fl):
            fn_zero = nodes_zero_fl[from_nodes]
            tn_zero = nodes_zero_fl[to_nodes]

            t_from_node_vor_zero = node_pit[from_nodes[fn_zero], IdxNode.TINIT_OLD]
            t_to_node_vor_zero = node_pit[to_nodes[tn_zero], IdxNode.TINIT_OLD]
            t_to_node = node_pit[to_nodes[tn_zero], IdxNode.TINIT]

            fn_eq = (rho[fn_zero] * area[fn_zero] * cp[fn_zero] * (1 / delta_t)
                     * (t_init_i[fn_zero] - t_from_node_vor_zero)
                     - alpha[fn_zero] * (t_amb[fn_zero] - t_init_i[fn_zero]))

            tn_eq = (rho[tn_zero] * area[tn_zero] * cp[tn_zero] * (1 / delta_t)
                     * (t_to_node - t_to_node_vor_zero)
                     - alpha[tn_zero] * (t_amb[tn_zero] - t_to_node))

            fn_deriv = (rho[fn_zero] * area[fn_zero] * cp[fn_zero] * (1 / delta_t) + alpha[fn_zero])
            tn_deriv = (rho[tn_zero] * area[tn_zero] * cp[tn_zero] * (1 / delta_t) + alpha[tn_zero])

            fn_nodes, fn_eq_sum, fn_deriv_sum= _sum_by_group(
                get_net_option(net, "use_numba"),
                from_nodes[fn_zero], fn_eq, fn_deriv
            )

            tn_nodes, tn_eq_sum, tn_deriv_sum = _sum_by_group(
                get_net_option(net, "use_numba"),
                to_nodes[tn_zero], tn_eq, tn_deriv
            )

            node_pit[nodes_zero_fl, IdxNode.LOAD_T] = 0
            node_pit[fn_nodes, IdxNode.LOAD_T] += fn_eq_sum
            node_pit[tn_nodes, IdxNode.LOAD_T] += tn_eq_sum
            node_pit[nodes_zero_fl, IdxNode.JAC_DERIV_DT_LOAD] = 0
            node_pit[fn_nodes, IdxNode.JAC_DERIV_DT_LOAD] -= fn_deriv_sum
            node_pit[tn_nodes, IdxNode.JAC_DERIV_DT_LOAD] -= tn_deriv_sum
            node_pit[nodes_zero_fl, IdxNode.JAC_DERIV_DT_SLACK] = 0

            branch_pit[fn_zero, IdxBranch.JAC_DERIV_DT_NODE] = 0
            branch_pit[tn_zero, IdxBranch.JAC_DERIV_DTOUT_NODE] = 0
            branch_pit[fn_zero, IdxBranch.LOAD_VEC_N_FROM_T] = 0
            branch_pit[tn_zero, IdxBranch.LOAD_VEC_N_TO_T] = 0

            from_nodes_not_zero = from_nodes[~nodes_zero_fl[from_nodes]]
            to_nodes_not_zero = to_nodes[~nodes_zero_fl[to_nodes]]
            infeed_node = np.setdiff1d(from_nodes_not_zero, to_nodes_not_zero)

    else:
        t_m = (t_init_i1 + t_init_i) / 2
        m_m = (m_init_i + m_init_i1) / 2

        branch_pit[:, IdxBranch.JAC_DERIV_DT] = - cp * m_m + alpha / 2 * length
        branch_pit[:, IdxBranch.JAC_DERIV_DTOUT] = cp * m_m + alpha / 2 * length
        branch_pit[:, IdxBranch.LOAD_VEC_B_T] = cp * m_m * (-t_init_i + t_init_i1 - tl) - alpha * (
                    t_amb - t_m) * length + qext

    if infeed_node is None:
        infeed_node = np.setdiff1d(from_nodes, to_nodes)
    node_pit[:, IdxNode.INFEED] = False
    node_pit[infeed_node, IdxNode.INFEED] = True

    # This approach can be used if you consider the effect of sources with given temperature (checkout issue #656)

    # branch_pit[:, LOAD_VEC_NODES_FROM_T] = m_init_i * t_init_i * cp_i
    # --> cp_i is calculated by fluid.get_heat_capacity(t_init_i)
    # branch_pit[:, LOAD_VEC_NODES_TO_T] = m_init_i1 * t_init_i1 * cp_i1
    # --> still missing is the derivative of loads
    # t_init = node_pit[:, TINIT_NODE]
    # cp_n = fluid.get_heat_capacity(t_init)
    # node_pit[:, LOAD_T] = cp_n * node_pit[:, LOAD] * t_init



def get_derived_values(node_pit, from_nodes, to_nodes, use_numba):
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(node_pit, from_nodes, to_nodes)


def calc_lambda(m, eta, d, k, gas_mode, friction_model, lengths, options, area):
    """
    Function calculates the friction factor of a pipe. Turbulence is calculated based on
    Nikuradse. If v equals 0, a value of 0.001 is used in order to avoid division by zero.
    This should not be a problem as the pressure loss term will equal zero (lambda * u^2).

    :param m:
    :type m:
    :param eta:
    :type eta:
    :param d:
    :type d:
    :param k:
    :type k:
    :param gas_mode:
    :type gas_mode:
    :param friction_model:
    :type friction_model:
    :param lengths:
    :type lengths:
    :param options:
    :type options:
    :param area:
    :type area:
    :return:
    :rtype:
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import calc_lambda_nikuradse_incomp_numba as \
            calc_lambda_nikuradse_incomp, colebrook_numba as colebrook, \
            calc_lambda_nikuradse_comp_numba as calc_lambda_nikuradse_comp
    else:
        from pandapipes.pf.derivative_toolbox import calc_lambda_nikuradse_incomp_np as \
            calc_lambda_nikuradse_incomp, colebrook_np as colebrook, \
            calc_lambda_nikuradse_comp_np as calc_lambda_nikuradse_comp
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(m, d, k, eta, area)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(m, d, k, eta, area)

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        dummy = (lengths != 0).astype(np.float64)
        converged, lambda_colebrook = colebrook(re, d, k, lambda_nikuradse, dummy, max_iter)
        if not converged:
            raise PipeflowNotConverged(
                "The Colebrook-White algorithm did not converge. There might be model "
                "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                "argument to the pipeflow.")
        return lambda_colebrook, re
    elif friction_model == "swamee-jain":
        # 1.325 instead of 0.25???
        lambda_swamee_jain = 0.25 / ((np.log10(k / (3.7 * d) + 5.74 / (re ** 0.9))) ** 2)
        return lambda_swamee_jain, re
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_tot = lambda_laminar + lambda_nikuradse
        return lambda_tot, re


def calc_der_lambda(m, eta, d, k, friction_model, lambda_pipe, area):
    """
    Function calculates the derivative of lambda with respect to v. Turbulence is calculated based
    on Nikuradse. This should not be a problem as the pressure loss term will equal zero
    (lambda * u^2).

    :param m:
    :type m:
    :param eta:
    :type eta:
    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param lambda_pipe:
    :type lambda_pipe:
    :param area:
    :type area:
    :return:
    :rtype:
    """

    b_term = np.zeros_like(m)
    df_dm = np.zeros_like(m)
    df_dlambda = np.zeros_like(m)
    lambda_der = np.zeros_like(m)
    pos = ~np.isclose(m, 0)

    if friction_model == "colebrook":
        b_term[pos] = (2.51 * eta[pos] * area[pos] / (m[pos] * d[pos] * np.sqrt(lambda_pipe[pos])) +
                       k[pos] / (3.71 * d[pos]))

        df_dm[pos] = -2 * 2.51 * eta[pos] * area[pos] / (m[pos] ** 2 * np.sqrt(lambda_pipe[pos]) * d[pos]) \
                / (np.log(10) * b_term[pos])

        df_dlambda[pos] = -0.5 * lambda_pipe[pos] ** (-3 / 2) - (2.51 * eta[pos] * area[pos] / (d[pos] * m[pos])) \
                     * lambda_pipe[pos] ** (-3 / 2) / (np.log(10) * b_term[pos])

        lambda_der[pos] = df_dm[pos] / df_dlambda[pos]

        return lambda_der
    elif friction_model == "swamee-jain":
        param = (k[pos] / (3.7 * d[pos]) + 5.74 * ((eta[pos] * area[pos]) /
                 (np.abs(m[pos]) * d[pos])) ** 0.9)
        # 0.5 / (log(10) * log(param)^3 * param) * 5.166 * abs(eta)^0.9  / (abs(rho * d)^0.9
        # * abs(v_corr)^1.9)
        lambda_der[pos] = 0.5 * np.log(10) ** 2 / (np.log(param) ** 3) / param * 5.166 \
                                 * ((eta[pos] * area[pos]) / (d[pos])) ** 0.9 * np.abs(m[pos]) ** -1.9
        return lambda_der
    else:
        lambda_der[pos] = -(64 * eta[pos] * area[pos]) / (m[pos] ** 2 * d[pos])
        return lambda_der
