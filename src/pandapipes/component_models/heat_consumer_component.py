# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import (get_fluid, BranchWOInternalsComponent, get_component_array,
                                         standard_branch_wo_internals_result_lookup)
from pandapipes.component_models.component_toolbox import build_pit_entries
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import (MDOTINIT, QEXT, TOUTINIT, FLOW_RETURN_CONNECT, FROM_NODE, TO_NODE)
from pandapipes.idx_node import TINIT
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.pf.derivative_calculation import calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, HydVarEq, ThermVarEq, PitEntries
from pandapipes.properties.properties_toolbox import get_branch_cp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class HeatConsumer(BranchWOInternalsComponent):
    """

    """
    # columns for internal array
    MASS = 0
    QEXT = 1
    DELTAT = 2
    TRETURN = 3
    MODE = 4

    internal_cols = 5

    # heat consumer modes (sum of combinations of given parameters)
    MF_DT = 1
    MF_TR = 2
    QE_MF = 3
    QE_DT = 4
    QE_TR = 5

    @classmethod
    def table_name(cls):
        return "heat_consumer"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("from_junction", "u4"), ("to_junction", "u4"), ("qext_w", "f8"),
                ("controlled_mdot_kg_per_s", "f8"), ("deltat_k", "f8"), ("treturn_k", "f8"),
                ("in_service", "bool"), ("type", dtype(object))]

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)

        qext = tbl.qext_w.values
        mask_qext = ~np.isnan(qext)
        if np.any(mask_qext):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_qext], [QEXT], [qext[mask_qext]])))

        mdot = tbl.controlled_mdot_kg_per_s.values
        mask_mdot = ~np.isnan(mdot)
        if np.any(mask_mdot):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_mdot], [MDOTINIT], [mdot[mask_mdot]])))

        treturn = tbl.treturn_k.values
        mask_tr = ~np.isnan(treturn)
        if np.any(mask_tr):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_tr], [TOUTINIT], [treturn[mask_tr]])))

        registry.add_override(PitEntries(*build_pit_entries(rows, [FLOW_RETURN_CONNECT], [np.ones(len(rows))])))

        mask_q0 = (qext == 0) & np.isnan(mdot)
        if np.any(mask_q0):
            logger.warning(r'qext_w is equals to zero for heat consumers with index %s. '
                           r'Therefore, the defined temperature control cannot be maintained.'
                           % tbl.index[mask_q0])

    @classmethod
    def create_component_array(cls, net, component_pits):
        """
        Function which creates an internal array of the component in analogy to the pit, but with
        component specific entries, that are not needed in the pit.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param component_pits: dictionary of component specific arrays
        :type component_pits: dict
        :return:
        :rtype:
        """
        tbl = net[cls.table_name()]
        consumer_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        consumer_array[:, cls.DELTAT] = tbl.deltat_k.values
        consumer_array[:, cls.TRETURN] = tbl.treturn_k.values
        consumer_array[:, cls.QEXT] = tbl.qext_w.values
        consumer_array[:, cls.MASS] = tbl.controlled_mdot_kg_per_s.values
        mf = tbl.controlled_mdot_kg_per_s.values
        tr = tbl.treturn_k.values
        dt = tbl.deltat_k.values
        qe = tbl.qext_w.values
        mf = ~np.isnan(mf)
        tr = ~np.isnan(tr)
        dt = ~np.isnan(dt)
        qe = ~np.isnan(qe)
        consumer_array[mf & dt, cls.MODE] = cls.MF_DT
        consumer_array[mf & tr, cls.MODE] = cls.MF_TR
        consumer_array[qe & mf, cls.MODE] = cls.QE_MF
        consumer_array[qe & dt, cls.MODE] = cls.QE_DT
        consumer_array[qe & tr, cls.MODE] = cls.QE_TR
        component_pits[cls.table_name()] = consumer_array

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        b_pit = branch_pit[f:t]
        fn = b_pit[:, FROM_NODE].astype(np.int32)
        tn = b_pit[:, TO_NODE].astype(np.int32)

        consumer_array = get_component_array(net, cls.table_name(), mode='hydraulics')

        # variables
        mdot_col  = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)

        # equation position branch
        branch_eq = sys_idx.idx(HydVarEq.BRANCH,   branch_idx)

        # derivative and load vector branch
        df_dm = np.ones_like(branch_idx, dtype=np.float64)
        load = np.zeros_like(branch_idx, dtype=np.float64)

        mask_qe_dt = consumer_array[:, cls.MODE] == cls.QE_DT
        if np.any(mask_qe_dt):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit[mask_qe_dt])
            deltat = consumer_array[mask_qe_dt, cls.DELTAT]
            mdot = b_pit[mask_qe_dt, QEXT] / (cp * deltat)
            load[mask_qe_dt] = - mdot + b_pit[mask_qe_dt, MDOTINIT]

        mask_qe_tr = consumer_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask_qe_tr):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit)
            from_nodes = get_from_nodes_corrected(b_pit).astype(np.int32)
            t_in  = node_pit[from_nodes, TINIT]
            t_out = b_pit[:, TOUTINIT]
            df_dm_qetr = -cp * (t_out - t_in)
            mask_equal = t_out >= t_in
            mask_zero  = b_pit[:, QEXT] == 0
            mask_ign   = mask_equal | mask_zero

            df_dm[mask_qe_tr & ~mask_ign] = df_dm_qetr[mask_qe_tr & ~mask_ign]
            load[mask_qe_tr] = (-b_pit[mask_qe_tr, QEXT] + df_dm_qetr[mask_qe_tr] * b_pit[mask_qe_tr, MDOTINIT])

        # jacobi matrix branch
        rows_branch = branch_eq.astype(np.int32)
        cols_branch = mdot_col.astype(np.int32)
        data_branch = df_dm.astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = load.astype(np.float64)

        # equation positions nodes
        fn_eq     = sys_idx.idx(HydVarEq.NODE, fn)
        tn_eq     = sys_idx.idx(HydVarEq.NODE, tn)

        # derivative and load vector nodes
        df_dm_node = np.ones_like(branch_idx)
        load_fn = -b_pit[:, MDOTINIT]
        load_tn = b_pit[:, MDOTINIT]
        if np.any(mask_qe_tr):
            load_fn[mask_qe_tr & mask_ign] = 0
            load_tn[mask_qe_tr & mask_ign] = 0

        # jacobi matrix node
        rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
        cols_node = np.concatenate([mdot_col, mdot_col]).astype(np.int32)
        data_node = np.concatenate([-df_dm_node, df_dm_node]).astype(np.float64)
        load_rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
        load_node = np.concatenate([load_fn, load_tn]).astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
            mode=EqWriteMode.UNIQUE,
        ))

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        options = {"use_numba": get_net_option(net, "use_numba")}
        branch_pit_old = net["_active_old_pit"]["branch"]

        b_pit = branch_pit[f:t]
        consumer_array = get_component_array(net, cls.table_name(), mode='heat_transfer')

        # preparation
        mask_mf_dt = consumer_array[:, cls.MODE] == cls.MF_DT
        if np.any(mask_mf_dt):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit)
            b_pit[mask_mf_dt, QEXT] = (cp[mask_mf_dt] * b_pit[mask_mf_dt, MDOTINIT]
                                         * consumer_array[mask_mf_dt, cls.DELTAT])

        mask_mf_tr = consumer_array[:, cls.MODE] == cls.MF_TR
        if np.any(mask_mf_tr):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit)
            fn_t = get_from_nodes_corrected(b_pit[mask_mf_tr]).astype(np.int32)
            t_in = node_pit[fn_t, TINIT]
            t_out = consumer_array[mask_mf_tr, cls.TRETURN]
            b_pit[mask_mf_tr, QEXT] = cp[mask_mf_tr] * b_pit[mask_mf_tr, MDOTINIT] * (t_in - t_out)


        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = calculate_derivatives_branch_thermal(
            net, branch_pit[f:t], node_pit, branch_pit_old[f:t], options
        )

        fn = get_from_nodes_corrected(b_pit).astype(np.int32)
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        # variables
        t_out_col = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)
        t_from_col = sys_idx.idx(ThermVarEq.TINIT, fn)
        t_to_col = sys_idx.idx(ThermVarEq.TINIT, tn)

        # equation positions branch
        branch_eq = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # derivative and load vector branches
        mask_qe_tr = consumer_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask_qe_tr):
            mask_ign = b_pit[:, QEXT] == 0
            mask = mask_qe_tr & ~mask_ign
            dfb_dt[mask] = 0
            dfb_dtout[mask] = 1
            fb[mask] = 0

        # jacobi matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq]).astype(np.int32)
        cols_branch = np.concatenate([t_from_col, t_out_col]).astype(np.int32)
        data_branch = np.concatenate([dfb_dt, dfb_dtout]).astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = fb.astype(np.float64)

        # equation positions node
        tn_eq = sys_idx.idx(ThermVarEq.NODE, tn)

        # jacobi matrix node
        rows_node = np.concatenate([tn_eq, tn_eq]).astype(np.int32)
        cols_node = np.concatenate([t_to_col, t_out_col]).astype(np.int32)
        data_node = np.concatenate([dfnt_dt, dfnt_dtout]).astype(np.float64)
        load_rows_node = tn_eq.astype(np.int32)
        load_node = fnt.astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
            mode=EqWriteMode.UNIQUE,
        ))

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))



    @classmethod
    def get_result_table(cls, net):
        """

        Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ['deltat_k', 'qext_w']
        return output, True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """

        :param net:
        :type net:
        :param options:
        :type options:
        :param branch_results:
        :type branch_results:
        :param mode:
        :type mode:
        :return:
        :rtype:
        """
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
            cls.table_name(), mode)

        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[cls.table_name()]

        res_table = net["res_" + cls.table_name()]

        res_table['qext_w'].values[:] = branch_pit[f:t, QEXT]
        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, TINIT]
        tout = branch_pit[f:t, TOUTINIT]
        res_table['deltat_k'].values[:] = t_from - tout
