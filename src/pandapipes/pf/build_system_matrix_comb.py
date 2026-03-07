# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.sparse import csr_matrix

from pandapipes.idx_branch import (FROM_NODE, TO_NODE, JAC_DERIV_DM, JAC_DERIV_DP, JAC_DERIV_DP1, \
                                   JAC_DERIV_DM_NODE, LOAD_VEC_NODES_FROM, LOAD_VEC_NODES_TO, LOAD_VEC_BRANCHES,
                                   JAC_DERIV_DT, JAC_DERIV_DTOUT,
                                   PC as PC_BRANCH, JAC_DERIV_DT_NODE, JAC_DERIV_DTOUT_NODE, LOAD_VEC_NODES_FROM_T,
                                   LOAD_VEC_NODES_TO_T, \
                                   LOAD_VEC_BRANCHES_T, BRANCH_TYPE, JAC_DERIV_DM_FROM_NODE, JAC_DERIV_DM_TO_NODE,
                                   JAC_DERIV_T_DM, JAC_DERIV_P_FROM_T, JAC_DERIV_P_TO_T)

from pandapipes.idx_node import (P, PC as PC_NODE, NODE_TYPE, T, NODE_TYPE_T, LOAD, LOAD_T, INFEED,
                                 MDOTSLACKINIT, JAC_DERIV_MSL, JAC_DERIV_DT_SLACK, JAC_DERIV_DT_LOAD)
from pandapipes.pf.internals_toolbox import _sum_by_group, \
    get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_net_option


def build_system_matrix_comb(net, branch_pit, node_pit):
    """
    Builds the system matrix.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit: pandapipes internal table for branching components such as pipes or valves
    :type branch_pit: numpy.ndarray
    :param node_pit:  pandapipes internal table for node components
    :type node_pit: numpy.ndarray
    :return: system_matrix, load_vector
    :rtype: system_matrix - scipy.sparse.csr.csr_matrix, load_vector - numpy.ndarray
    """
    use_numba = get_net_option(net, "use_numba")

    NUM_DER_HYD = 5
    NUM_DER_THERM = 3

    len_b = len(branch_pit)
    len_n = len(node_pit)
    branch_matrix_indices_hyd = np.arange(len_b) + len_n
    pc_nodes_hyd = np.where(node_pit[:, NODE_TYPE] == PC_NODE)[0]

    fn_hyd = branch_pit[:, FROM_NODE].astype(np.int32)
    tn_hyd = branch_pit[:, TO_NODE].astype(np.int32)
    pc_branch_mask_hyd = branch_pit[:, BRANCH_TYPE] == PC_BRANCH
    slack_nodes_hyd = np.where(node_pit[:, NODE_TYPE] == P)[0]
    pc_matrix_indices_hyd = branch_matrix_indices_hyd[pc_branch_mask_hyd]

    # size of the matrix
    len_sl_hyd = len(slack_nodes_hyd)
    slack_mass_matrix_indices_hyd = np.arange(len_sl_hyd) + len_b + len_n
    slack_masses_from_hyd, slack_branches_from_hyd = np.where(branch_pit[:, FROM_NODE] == slack_nodes_hyd[:, None])
    slack_masses_to_hyd, slack_branches_to_hyd = np.where(branch_pit[:, TO_NODE] == slack_nodes_hyd[:, None])
    not_slack_fn_branch_mask_hyd = node_pit[fn_hyd, NODE_TYPE] != P
    not_slack_tn_branch_mask_hyd = node_pit[tn_hyd, NODE_TYPE] != P
    len_fn_not_slack_hyd = np.sum(not_slack_fn_branch_mask_hyd)
    len_tn_not_slack_hyd = np.sum(not_slack_tn_branch_mask_hyd)
    len_fn1_hyd = NUM_DER_HYD * len_b + len_fn_not_slack_hyd
    len_tn1_hyd = len_fn1_hyd + len_tn_not_slack_hyd
    len_pc_hyd = len_tn1_hyd + pc_nodes_hyd.shape[0]
    len_slack_hyd = len_pc_hyd + slack_nodes_hyd.shape[0]
    len_fsb_hyd = len_slack_hyd + len(slack_branches_from_hyd)
    len_tsb_hyd = len_fsb_hyd + len(slack_branches_to_hyd)
    full_len_hyd = len_tsb_hyd + slack_nodes_hyd.shape[0]

    len_hyd = len_n + len_b + len_sl_hyd
    branch_matrix_indices_therm = np.arange(len_b) + len_n + len_hyd

    fn_therm = get_from_nodes_corrected(branch_pit)
    tn_therm = get_to_nodes_corrected(branch_pit)
    slack_nodes_therm = np.where(node_pit[:, NODE_TYPE_T] == T)[0]

    not_slack_fn_branch_mask_therm = ~node_pit[fn_therm, INFEED].astype(np.bool_)
    not_slack_tn_branch_mask_therm = ~node_pit[tn_therm, INFEED].astype(np.bool_)
    not_slack_mask_therm = ~node_pit[:, INFEED].astype(np.bool_)
    len_fn_not_slack_therm = np.sum(not_slack_fn_branch_mask_therm)
    len_tn_not_slack_therm = np.sum(not_slack_tn_branch_mask_therm)
    len_not_slack_therm = np.sum(not_slack_mask_therm)
    infeed_node_therm = np.arange(len_n)[node_pit[:, INFEED].astype(np.bool_)]
    len_fn1_therm = NUM_DER_THERM * len_b + len_fn_not_slack_therm + full_len_hyd
    len_tload_therm = len_fn1_therm + len_not_slack_therm
    len_tslack_therm = len_tload_therm + len_not_slack_therm
    len_tout_therm = len_tslack_therm + len_tn_not_slack_therm
    len_mfrom_therm = len_tout_therm + len_fn_not_slack_therm
    len_mto_therm = len_mfrom_therm + len_tn_not_slack_therm
    full_len_therm = len_mto_therm + slack_nodes_therm.shape[0]

    system_data = np.zeros(full_len_therm, dtype=np.float64)

    # entries in the matrix
    # HYDRAULIC
    # branch equations
    # ----------------
    # branch_dF_dm
    system_data[:len_b] = branch_pit[:, JAC_DERIV_DM]
    # branch_dF_dp_from
    system_data[len_b:2 * len_b] = branch_pit[:, JAC_DERIV_DP]
    # branch_dF_dp_to
    system_data[2 * len_b:3 * len_b] = branch_pit[:, JAC_DERIV_DP1]
    # branch_dF_dt_from
    system_data[3 * len_b:4 * len_b] = branch_pit[:, JAC_DERIV_P_FROM_T]
    # branch_dF_dt_to
    system_data[4 * len_b:5 * len_b] = branch_pit[:, JAC_DERIV_P_TO_T]

    # node equations
    # --------------
    # from_node_dF_dm
    system_data[5 * len_b:len_fn1_hyd] = branch_pit[not_slack_fn_branch_mask_hyd, JAC_DERIV_DM_NODE] * (-1)
    # to_node_dF_dm
    system_data[len_fn1_hyd:len_tn1_hyd] = branch_pit[not_slack_tn_branch_mask_hyd, JAC_DERIV_DM_NODE]

    # fixed pressure equations
    # ------------------------
    # pc_nodes and slack_nodes
    system_data[len_tn1_hyd:len_slack_hyd] = 1

    # mass flow slack equation
    # --------------
    # from_slack_dF_dm
    system_data[len_slack_hyd:len_fsb_hyd] = branch_pit[slack_branches_from_hyd, JAC_DERIV_DM_NODE] * (-1)
    # to_slack_dF_dm
    system_data[len_fsb_hyd:len_tsb_hyd] = branch_pit[slack_branches_to_hyd, JAC_DERIV_DM_NODE]
    # slackmass_dF_dmslack
    system_data[len_tsb_hyd:full_len_hyd] = node_pit[slack_nodes_hyd, JAC_DERIV_MSL]
    # THERMAL
    # branch equations
    # ----------------
    # branch_dF_dT_from
    system_data[full_len_hyd:len_b + full_len_hyd] = branch_pit[:, JAC_DERIV_DT]
    # branch_dF_dT_out
    system_data[len_b + full_len_hyd:2 * len_b + full_len_hyd] = branch_pit[:, JAC_DERIV_DTOUT]
    # branch_dF_dm
    system_data[2 * len_b + full_len_hyd: 3*len_b + full_len_hyd] = branch_pit[:, JAC_DERIV_T_DM]

    # node equations
    # --------------
    # node_dF_dT_from
    system_data[3 * len_b + full_len_hyd:len_fn1_therm] = branch_pit[not_slack_fn_branch_mask_therm, JAC_DERIV_DT_NODE]
    # node_dFload_dT
    system_data[len_fn1_therm:len_tload_therm] = node_pit[not_slack_mask_therm, JAC_DERIV_DT_LOAD]
    # node_dFslack_dT
    system_data[len_tload_therm:len_tslack_therm] = node_pit[not_slack_mask_therm, JAC_DERIV_DT_SLACK]
    # node_dF_dT_out
    system_data[len_tslack_therm:len_tout_therm] = branch_pit[not_slack_tn_branch_mask_therm, JAC_DERIV_DTOUT_NODE]
    # node_dF_dm_from
    system_data[len_tout_therm:len_mfrom_therm] = branch_pit[not_slack_fn_branch_mask_therm, JAC_DERIV_DM_FROM_NODE]
    # node_dF_dm_out
    system_data[len_mfrom_therm:len_mto_therm] = branch_pit[not_slack_tn_branch_mask_therm, JAC_DERIV_DM_TO_NODE]

    # fixed temperature equations
    # ---------------------------
    # t_nodes
    system_data[len_mto_therm:] = 1

    # position in the matrix
    system_cols = np.zeros(full_len_therm, dtype=np.int32)
    system_rows = np.zeros(full_len_therm, dtype=np.int32)


    # position in the matrix
    # HYDRAULIC
    # branch equations
    # ----------------
    # branch_dF_dm
    system_cols[:len_b] = branch_matrix_indices_hyd
    system_rows[:len_b] = branch_matrix_indices_hyd
    # branch_dF_dp_from
    system_cols[len_b:2 * len_b] = fn_hyd
    system_rows[len_b:2 * len_b] = branch_matrix_indices_hyd
    # branch_dF_dp_to
    system_cols[2 * len_b:3 * len_b] = tn_hyd
    system_rows[2 * len_b:3 * len_b] = branch_matrix_indices_hyd
    # branch_dF_dt_from
    system_cols[3 * len_b:4 * len_b] = fn_therm  + len_hyd
    system_rows[3 * len_b:4 * len_b] = branch_matrix_indices_hyd
    # branch_dF_dt_to
    system_cols[4 * len_b:5 * len_b] = branch_matrix_indices_therm
    system_rows[4 * len_b:5 * len_b] = branch_matrix_indices_hyd

    # node equations
    # --------------
    # from_node_dF_dm
    system_cols[5 * len_b:len_fn1_hyd] = branch_matrix_indices_hyd[not_slack_fn_branch_mask_hyd]
    system_rows[5 * len_b:len_fn1_hyd] = fn_hyd[not_slack_fn_branch_mask_hyd]
    # to_node_dF_dm
    system_cols[len_fn1_hyd:len_tn1_hyd] = branch_matrix_indices_hyd[not_slack_tn_branch_mask_hyd]
    system_rows[len_fn1_hyd:len_tn1_hyd] = tn_hyd[not_slack_tn_branch_mask_hyd]

    # fixed pressure equations
    # -----------------------
    # pc_nodes
    system_cols[len_tn1_hyd:len_pc_hyd] = pc_nodes_hyd
    system_rows[len_tn1_hyd:len_pc_hyd] = pc_matrix_indices_hyd
    # slack_nodes
    system_cols[len_pc_hyd:len_slack_hyd] = slack_nodes_hyd
    system_rows[len_pc_hyd:len_slack_hyd] = slack_nodes_hyd

    # mass flow slack equation
    # --------------
    # from_slack_dF_dm
    system_cols[len_slack_hyd:len_fsb_hyd] = branch_matrix_indices_hyd[slack_branches_from_hyd]
    system_rows[len_slack_hyd:len_fsb_hyd] = slack_mass_matrix_indices_hyd[slack_masses_from_hyd]
    # to_slack_dF_dm
    system_cols[len_fsb_hyd:len_tsb_hyd] = branch_matrix_indices_hyd[slack_branches_to_hyd]
    system_rows[len_fsb_hyd:len_tsb_hyd] = slack_mass_matrix_indices_hyd[slack_masses_to_hyd]
    # slackmass_dF_dmslack
    system_cols[len_tsb_hyd:full_len_hyd] = slack_mass_matrix_indices_hyd
    system_rows[len_tsb_hyd:full_len_hyd] = slack_mass_matrix_indices_hyd

    # THERMAL
    # branch equations
    # ----------------
    # branch_dF_dT_from
    system_cols[full_len_hyd:len_b+full_len_hyd] = fn_therm  + len_hyd
    system_rows[full_len_hyd:len_b+full_len_hyd] = branch_matrix_indices_therm
    # branch_dF_dT_out
    system_cols[len_b+full_len_hyd:2 * len_b+full_len_hyd] = branch_matrix_indices_therm
    system_rows[len_b+full_len_hyd:2 * len_b+full_len_hyd] = branch_matrix_indices_therm
    # branch_dF_dm
    system_cols[2 * len_b+full_len_hyd:3 * len_b+full_len_hyd] = branch_matrix_indices_hyd
    system_rows[2 * len_b+full_len_hyd:3 * len_b+full_len_hyd] = branch_matrix_indices_therm

    # node equations
    # --------------
    # node_dF_dT_from
    system_cols[3 * len_b+full_len_hyd:len_fn1_therm] = fn_therm[not_slack_fn_branch_mask_therm] + len_hyd
    system_rows[3 * len_b+full_len_hyd:len_fn1_therm] = fn_therm[not_slack_fn_branch_mask_therm] + len_hyd
    # node_dFload_dT
    system_cols[len_fn1_therm:len_tload_therm] = np.arange(len_n)[not_slack_mask_therm] + len_hyd
    system_rows[len_fn1_therm:len_tload_therm] = np.arange(len_n)[not_slack_mask_therm] + len_hyd
    # node_dFslack_dT
    system_cols[len_tload_therm:len_tslack_therm] = np.arange(len_n)[not_slack_mask_therm] + len_hyd
    system_rows[len_tload_therm:len_tslack_therm] = np.arange(len_n)[not_slack_mask_therm] + len_hyd
    # node_dF_dT_out
    system_cols[len_tslack_therm:len_tout_therm] = branch_matrix_indices_therm[not_slack_tn_branch_mask_therm]
    system_rows[len_tslack_therm:len_tout_therm] = tn_therm[not_slack_tn_branch_mask_therm] + len_hyd
    # node_dF_dm_from
    system_cols[len_tout_therm:len_mfrom_therm] = branch_matrix_indices_hyd[not_slack_fn_branch_mask_therm]
    system_rows[len_tout_therm:len_mfrom_therm] = fn_therm[not_slack_fn_branch_mask_therm] + len_hyd
    # node_dF_dm_to
    system_cols[len_mfrom_therm:len_mto_therm] = branch_matrix_indices_hyd[not_slack_tn_branch_mask_therm]
    system_rows[len_mfrom_therm:len_mto_therm] = tn_therm[not_slack_tn_branch_mask_therm] + len_hyd

    # fixed temperature equations
    # ---------------------------
    # t_nodes (overwrites only infeeding nodes' equation)
    system_cols[len_mto_therm:] = slack_nodes_therm
    system_rows[len_mto_therm:] = infeed_node_therm

    system_matrix = csr_matrix((system_data, (system_rows, system_cols)),
                               shape=(len_n * 2 + len_b * 2 + len_sl_hyd, len_n * 2 + len_b * 2 + len_sl_hyd))

    # load vector on the right side
    load_vector = np.empty(len_n * 2 + len_b * 2 + len_sl_hyd)
    load_vector[:len_n] = node_pit[:, LOAD] * (-1)
    load_vector[len_n:len_b + len_n] = branch_pit[:, LOAD_VEC_BRANCHES]
    fn_unique_hyd, fn_sums_hyd = _sum_by_group(use_numba, fn_hyd, branch_pit[:, LOAD_VEC_NODES_FROM])
    tn_unique_hyd, tn_sums_hyd = _sum_by_group(use_numba, tn_hyd, branch_pit[:, LOAD_VEC_NODES_TO])
    load_vector[fn_unique_hyd] -= fn_sums_hyd
    load_vector[tn_unique_hyd] += tn_sums_hyd
    load_vector[slack_nodes_hyd] = 0
    load_vector[pc_matrix_indices_hyd] = 0

    load_vector[slack_mass_matrix_indices_hyd] = node_pit[slack_nodes_hyd, LOAD] * (-1)
    fsb_unique_hyd, fsb_sums_hyd = _sum_by_group(use_numba, slack_masses_from_hyd,
                                         branch_pit[slack_branches_from_hyd, LOAD_VEC_NODES_FROM])
    tsb_unique_hyd, tsb_sums_hyd = _sum_by_group(use_numba, slack_masses_to_hyd,
                                         branch_pit[slack_branches_to_hyd, LOAD_VEC_NODES_TO])
    load_vector[slack_mass_matrix_indices_hyd[fsb_unique_hyd]] -= fsb_sums_hyd
    load_vector[slack_mass_matrix_indices_hyd[tsb_unique_hyd]] += tsb_sums_hyd
    load_vector[slack_mass_matrix_indices_hyd] -= node_pit[slack_nodes_hyd, MDOTSLACKINIT]

    load_vector[len_hyd:len_n+len_hyd] = node_pit[:, LOAD_T] * (-1)
    load_vector[len_n+len_hyd:] = branch_pit[:, LOAD_VEC_BRANCHES_T]

    fn_unique_therm, fn_sums_therm = _sum_by_group(use_numba, fn_therm, branch_pit[:, LOAD_VEC_NODES_FROM_T])
    tn_unique_therm, tn_sums_therm = _sum_by_group(use_numba, tn_therm, branch_pit[:, LOAD_VEC_NODES_TO_T])
    load_vector[fn_unique_therm+len_hyd] -= fn_sums_therm
    load_vector[tn_unique_therm+len_hyd] += tn_sums_therm
    load_vector[infeed_node_therm] = 0

    return system_matrix, load_vector
