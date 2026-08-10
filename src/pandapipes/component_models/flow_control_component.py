# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWOInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, standard_branch_wo_internals_result_lookup, get_component_array,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import (
    ELEMENT_IDX, FROM_NODE, TO_NODE, MDOTINIT, FLOW_RETURN_CONNECT,
)
from pandapipes.pf.derivative_calculation import (
    calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal,
)
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup, get_net_option
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import (
    ComponentEquations, HydVarEq, PitEntries, ThermVarEq,
)


class FlowControlComponent(BranchWOInternalsComponent):
    """

    """
    CONTROL_ACTIVE = 0

    internal_cols = 1

    @classmethod
    def table_name(cls):
        return "flow_control"

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
                ("controlled_mdot_kg_per_s", "f8"),
                ("control_active", "bool"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)

        registry.add(PitEntries(*build_pit_entries(
            rows,
            [MDOTINIT],
            [tbl.controlled_mdot_kg_per_s.values],
        )))

        ctrl_active = tbl.control_active.values.astype(bool)
        if np.any(ctrl_active):
            registry.add(PitEntries(*build_pit_entries(
                rows[ctrl_active],
                [FLOW_RETURN_CONNECT],
                [1.0],
            )))

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        fc_pit = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        fc_pit[:, cls.CONTROL_ACTIVE] = tbl.control_active.values
        component_pits[cls.table_name()] = fc_pit

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return

        options = {"use_numba": get_net_option(net, "use_numba"),
                   "friction_model": get_net_option(net, "friction_model")}

        # Compute derivatives for all active branches (writes RE, LAMBDA back to pit)
        df_dm, df_dp, df_dp1, df_dm_node, load, load_fn, load_tn = (
            calculate_derivatives_hydraulic(net, branch_pit[f:t], node_pit, options)
        )

        b_pit = branch_pit[f:t]
        fn = b_pit[:, FROM_NODE].astype(np.int32)
        tn = b_pit[:, TO_NODE].astype(np.int32)
        tbl_idx = b_pit[:, ELEMENT_IDX].astype(np.int32)

        ctrl_active = net[cls.table_name()].control_active.values[tbl_idx].astype(bool)
        controlled_mdot = net[cls.table_name()].controlled_mdot_kg_per_s.values[tbl_idx]

        # For control-active branches: prescribe mass flow (override branch equation)
        df_dm[ctrl_active] = 1.0
        df_dp[ctrl_active] = 0.0
        df_dp1[ctrl_active] = 0.0
        load[ctrl_active] = b_pit[ctrl_active, MDOTINIT] - controlled_mdot[ctrl_active]
        # load_fn / load_tn keep their MDOTINIT values — correct for all node mass balances

        mdot_col   = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)
        p_from_col = sys_idx.idx(HydVarEq.PINIT, fn)
        p_to_col   = sys_idx.idx(HydVarEq.PINIT, tn)
        branch_eq  = sys_idx.idx(HydVarEq.BRANCH, branch_idx)
        fn_eq      = sys_idx.idx(HydVarEq.NODE, fn)
        tn_eq      = sys_idx.idx(HydVarEq.NODE, tn)

        rows = np.concatenate([branch_eq, branch_eq, branch_eq, fn_eq, tn_eq])
        cols = np.concatenate([mdot_col, p_from_col, p_to_col, mdot_col, mdot_col])
        data = np.concatenate([df_dm, df_dp, df_dp1, -df_dm_node, df_dm_node])
        load_rows = np.concatenate([branch_eq, fn_eq, tn_eq])
        load_data = np.concatenate([load, -load_fn, load_tn])

        registry.add(ComponentEquations(
            rows.astype(np.int32), cols.astype(np.int32), data.astype(np.float64),
            load_rows.astype(np.int32), load_data.astype(np.float64),
        ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        options = {"use_numba": get_net_option(net, "use_numba")}
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = (
            calculate_derivatives_branch_thermal(net, branch_pit[f:t], node_pit, branch_pit_old[f:t], options)
        )

        b_pit = branch_pit[f:t]
        fn = get_from_nodes_corrected(b_pit).astype(np.int32)
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        # variables
        t_out_col  = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)
        t_from_col = sys_idx.idx(ThermVarEq.TINIT, fn)
        t_tn_col   = sys_idx.idx(ThermVarEq.TINIT, tn)

        # equation positions branch
        branch_eq  = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # jacobi matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq])
        cols_branch = np.concatenate([t_from_col, t_out_col])
        data_branch = np.concatenate([dfb_dt, dfb_dtout])

        # load vector branch
        load_rows_branch = branch_eq
        load_branch      = fb

        # equation positions node
        tn_eq = sys_idx.idx(ThermVarEq.NODE, tn)

        # jacobi matrix node
        rows_node = np.concatenate([tn_eq, tn_eq])
        cols_node = np.concatenate([t_tn_col, t_out_col])
        data_node = np.concatenate([dfnt_dt, dfnt_dtout])

        # load vector node
        load_rows_node = tn_eq
        load_node      = fnt

        # fuse
        rows = np.concatenate([rows_branch, rows_node]).astype(np.int32)
        cols = np.concatenate([cols_branch, cols_node]).astype(np.int32)
        data = np.concatenate([data_branch, data_node]).astype(np.float64)
        load_rows = np.concatenate([load_rows_branch, load_rows_node]).astype(np.int32)
        load_data = np.concatenate([load_branch, load_node]).astype(np.float64)

        registry.add(ComponentEquations(
            rows, cols, data,
            load_rows, load_data,
        ))

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

    @classmethod
    def get_result_table(cls, net):
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s"]
        return output, True
