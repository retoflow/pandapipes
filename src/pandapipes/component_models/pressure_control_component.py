# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.branch_wo_internals_models import \
    BranchWOInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, standard_branch_wo_internals_result_lookup,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import (
    calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal,
)
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import (
    ComponentEquations, HydVarEq, PitEntries, PitWriteMode, ThermVarEq,
)
from pandapipes.properties.fluids import get_fluid


class PressureControlComponent(BranchWOInternalsComponent):
    """

    """
    JUNCTS = 0
    IN_SERVICE = 1
    CONTROLLED = 2

    internal_cols = 3

    @classmethod
    def table_name(cls):
        return "press_control"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def get_component_input(cls):
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("controlled_junction", "u4"),
                ("controlled_p_bar", "f8"),
                ("control_active", "bool"),
                ("loss_coefficient", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        pcs = net[cls.table_name()]
        controlled = pcs.control_active.values & pcs.in_service.values
        if not np.any(controlled):
            return
        juncts = pcs['controlled_junction'].values[controlled]
        press = pcs['controlled_p_bar'].values[controlled]
        junction_idx_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()
        ]
        index_pc = junction_idx_lookup[juncts]
        registry.add_override(PitEntries(
            index_pc.astype(np.int32),
            np.full(len(index_pc), IdxNode.PINIT, dtype=np.int32),
            press.astype(np.float64),
            mode=PitWriteMode.MEAN,
        ))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)
        registry.add(PitEntries(*build_pit_entries(
            rows, [IdxBranch.LOSS_COEFFICIENT, IdxBranch.DIRECTED], [tbl.loss_coefficient.values, True],
        )))

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        pc_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        pc_array[:, cls.JUNCTS] = tbl["controlled_junction"].values
        pc_array[:, cls.CONTROLLED] = tbl.control_active.values
        pc_array[:, cls.IN_SERVICE] = tbl.in_service.values
        component_pits[cls.table_name()] = pc_array

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return

        options = {"use_numba": get_net_option(net, "use_numba"),
                   "friction_model": get_net_option(net, "friction_model")}

        b_pit = branch_pit[f:t]
        tbl_idx = b_pit[:, IdxBranch.ELEMENT_IDX].astype(np.int32)
        tbl = net[cls.table_name()]

        ctrl_active = tbl.control_active.values[tbl_idx].astype(bool)
        in_service_arr = tbl.in_service.values[tbl_idx].astype(bool)
        ctrl_juncts = tbl['controlled_junction'].values[tbl_idx].astype(np.int32)

        junction_idx_active = get_lookup(net, "node", "index_active_hydraulics")[
            cls.get_connected_node_type().table_name()
        ]
        index_pc = junction_idx_active[ctrl_juncts]

        if np.any(index_pc[in_service_arr] == -1):
            raise UserWarning(
                f"Controlled junction(s) are disconnected while the pressure controller is in "
                f"service: {ctrl_juncts[in_service_arr][index_pc[in_service_arr] == -1]}"
            )

        df_dm, df_dp, df_dp1, df_dm_node, load, load_fn, load_tn = (
            calculate_derivatives_hydraulic(net, b_pit, node_pit, options)
        )

        fn = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        tn = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)

        # Zero out branch equation contributions for ctrl_active branches (replaced by PC constraint)
        df_dm[ctrl_active]  = 0.0
        df_dp[ctrl_active]  = 0.0
        df_dp1[ctrl_active] = 0.0
        load[ctrl_active]   = 0.0

        # variables
        mdot_col   = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)
        p_from_col = sys_idx.idx(HydVarEq.PINIT, fn)
        p_to_col   = sys_idx.idx(HydVarEq.PINIT, tn)

        # equation position branch
        branch_eq  = sys_idx.idx(HydVarEq.BRANCH, branch_idx)

        # system matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq, branch_eq]).astype(np.int32)
        cols_branch = np.concatenate([mdot_col, p_from_col, p_to_col]).astype(np.int32)
        data_branch = np.concatenate([df_dm, df_dp, df_dp1]).astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = load.astype(np.float64)

        # equation position node
        fn_eq      = sys_idx.idx(HydVarEq.NODE, fn)
        tn_eq      = sys_idx.idx(HydVarEq.NODE, tn)

        # system matrix node
        rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
        cols_node = np.concatenate([mdot_col, mdot_col]).astype(np.int32)
        data_node = np.concatenate([-df_dm_node, df_dm_node]).astype(np.float64)
        load_rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
        load_node = np.concatenate([-load_fn, load_tn]).astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
        ))

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))

        # Pressure constraint for ctrl_active branches: P_ctrl_node = P_target
        if np.any(ctrl_active):
            ca_branch_eq = branch_eq[ctrl_active]
            ca_index_pc = index_pc[ctrl_active]
            p_ctrl_col = sys_idx.idx(HydVarEq.PINIT, ca_index_pc)
            p_target = tbl.controlled_p_bar.values[tbl_idx[ctrl_active]]
            p_ctrl_val = node_pit[ca_index_pc, IdxNode.PINIT]

            registry.add(ComponentEquations(
                rows=ca_branch_eq.astype(np.int32),
                cols=p_ctrl_col.astype(np.int32),
                data=np.ones(len(ca_branch_eq), dtype=np.float64),
                load_rows=ca_branch_eq.astype(np.int32),
                load_data=(p_ctrl_val - p_target).astype(np.float64),
            ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        options = {"use_numba": get_net_option(net, "use_numba")}
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = (
            calculate_derivatives_branch_thermal(net, branch_pit[f:t], node_pit,
                                          branch_pit_old[f:t], options)
        )

        b_pit = branch_pit[f:t]
        fn = get_from_nodes_corrected(b_pit).astype(np.int32)
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        # variables
        t_out_col  = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)
        t_from_col = sys_idx.idx(ThermVarEq.TINIT, fn)
        t_tn_col   = sys_idx.idx(ThermVarEq.TINIT, tn)

        # equation position branch
        branch_eq  = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # system matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq]).astype(np.int32)
        cols_branch = np.concatenate([t_from_col, t_out_col]).astype(np.int32)
        data_branch = np.concatenate([dfb_dt, dfb_dtout]).astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = fb.astype(np.float64)

        # equation position node
        tn_eq = sys_idx.idx(ThermVarEq.NODE, tn)

        # system matrix node
        rows_node = np.concatenate([tn_eq, tn_eq]).astype(np.int32)
        cols_node = np.concatenate([t_tn_col, t_out_col]).astype(np.int32)
        data_node = np.concatenate([dfnt_dt, dfnt_dtout]).astype(np.float64)
        load_rows_node = tn_eq.astype(np.int32)
        load_node = fnt.astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
        ))

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

        res_table = net["res_" + cls.table_name()]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        p_to = branch_results["p_to"][f:t]
        p_from = branch_results["p_from"][f:t]
        res_table["deltap_bar"].values[:] = p_to - p_from

    @classmethod
    def get_result_table(cls, net):
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ["deltap_bar"]
        return output, True
