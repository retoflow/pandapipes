from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode

def default_system_matrix_branch(branch_pit, node_pit):
   
    from_nodes = branch_pit[:, IdxBranch.FROM_NODE]
    to_nodes = branch_pit[:, IdxBranch.TO_NODE]

    jac_m = branch_pit[:, IdxBranch.DF_DM_B]
    jac_p_f = branch_pit[:, IdxBranch.DF_DP_F_B]
    jac_p_t = branch_pit[:, IdxBranch.DF_DP_T_B]
    jac_ms_f = branch_pit[:, IdxBranch.DF_DMSLACK_F_B]
    jac_ms_t = branch_pit[:, IdxBranch.DF_DMSLACK_T_B]

    c_m = branch_pit[:, IdxBranch.M_COL]
    c_p_f = node_pit[from_nodes, IdxNode.P_COL]
    c_p_t = node_pit[to_nodes, IdxNode.P_COL]
    c_ms_f = node_pit[from_nodes, IdxNode.MSLACK_COL]
    c_ms_t = node_pit[to_nodes, IdxNode.MSLACK_COL]

    r = branch_pit[:, IdxBranch.ROW_B]

    res = branch_pit[:, IdxBranch.F_B]

    res_r = branch_pit[:, IdxBranch.ROW_B]
    return [jac_m, jac_p_f, jac_p_t, jac_ms_f, jac_ms_t], \
           [c_m, c_p_f, c_p_t, c_ms_f, c_ms_t],\
           [r, r, r, r, r], \
           [res], \
           [res_r]

def default_system_matrix_node(branch_pit, node_pit):

    from_nodes = branch_pit[:, IdxBranch.FROM_NODE]
    to_nodes = branch_pit[:, IdxBranch.TO_NODE]

    l_f = node_pit[from_nodes, IdxNode.NODE_TYPE] == IdxNode.L
    l_t = node_pit[to_nodes, IdxNode.NODE_TYPE] == IdxNode.L
    p_f = node_pit[from_nodes, IdxNode.NODE_TYPE] == IdxNode.P
    p_t = node_pit[to_nodes, IdxNode.NODE_TYPE] == IdxNode.P

    jac1_m_f = branch_pit[:, IdxBranch.DF1_DM_F_N] * l_f
    jac1_m_t = branch_pit[:, IdxBranch.DF1_DM_T_N] * l_t
    jac1_p = node_pit[:, IdxNode.DF1_DP_N] * l_f
    jac1_ms = node_pit[:, IdxNode.DF1_DMSLACK_N] * l_f

    jac2_m_f = branch_pit[:, IdxBranch.DF2_DM_F_N] * p_f
    jac2_m_t = branch_pit[:, IdxBranch.DF2_DM_T_N] * p_t
    jac2_p = node_pit[:, IdxNode.DF2_DP_N] * p_f
    jac2_ms = node_pit[:, IdxNode.DF2_DMSLACK_N] * p_f

    c_m_f = branch_pit[:, IdxBranch.M_COL]
    c_m_t = branch_pit[:, IdxBranch.M_COL]
    c_p = node_pit[:, IdxNode.P_COL]
    c_ms = node_pit[:, IdxNode.MSLACK_COL]

    r1 = node_pit[:, IdxNode.ROW1_N]
    r1_f = node_pit[from_nodes, IdxNode.ROW1_N]
    r1_t = node_pit[to_nodes, IdxNode.ROW1_N]
    r2 = node_pit[:, IdxNode.ROW2_N]
    r2_f = node_pit[from_nodes, IdxNode.ROW2_N]
    r2_t = node_pit[to_nodes, IdxNode.ROW2_N]

    res1_f = branch_pit[:, IdxBranch.F1_F_N]
    res1_t = branch_pit[:, IdxBranch.F1_T_N]
    res2_f = branch_pit[:, IdxBranch.F2_F_N]
    res2_t = branch_pit[:, IdxBranch.F2_T_N]

    res1_r_f = node_pit[from_nodes, IdxNode.ROW1_N]
    res1_r_t = node_pit[to_nodes, IdxNode.ROW1_N]
    res2_r_f = node_pit[from_nodes, IdxNode.ROW2_N]
    res2_r_t = node_pit[to_nodes, IdxNode.ROW2_N]
    
    return [jac1_m_f, jac1_m_t, jac1_p, jac1_ms, jac2_m_f, jac2_m_t, jac2_p, jac2_ms], \
           [c_m_f, c_m_t, c_p, c_ms, c_m_f, c_m_t, c_p, c_ms], \
           [r1_f, r1_t, r1, r1, r2_f, r2_t, r2, r2], \
           [res1_f, res1_t, res2_f, res2_t], \
           [res1_r_f, res1_r_t, res2_r_f, res2_r_t]