# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.component_toolbox import build_pit_entries
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import GRAVITATION_CONSTANT, P_CONVERSION
from pandapipes.idx_branch import FROM_NODE, TO_NODE, MDOTINIT, PL
from pandapipes.idx_node import PINIT, PAMB, HEIGHT
from pandapipes.pf.derivative_calculation import calculate_derivatives_branch_thermal
from pandapipes.pf.internals_toolbox import get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup, get_fluid, get_net_option
from pandapipes.properties.properties_toolbox import get_branch_real_density
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, PitEntries, HydVarEq, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpPressure(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_pressure"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("return_junction", "u4"), ("flow_junction", "u4"), ("p_flow_bar", "f8"),
                ("t_flow_k", "f8"), ("plift_bar", "f8"), ("in_service", 'bool'), ("type", dtype(object))]

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)
        registry.add(PitEntries(*build_pit_entries(
            rows, [PL], [tbl['plift_bar'].values],
        )))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        cls._register_node_continuity(net, branch_pit, node_pit, sys_idx, registry)

        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        if f == t:
            return

        branch_idx = np.arange(f, t, dtype=np.int32)
        b_pit = branch_pit[f:t]
        fn = b_pit[:, FROM_NODE].astype(np.int32)
        tn = b_pit[:, TO_NODE].astype(np.int32)

        p_from_col = sys_idx.idx(HydVarEq.PINIT, fn)
        p_to_col = sys_idx.idx(HydVarEq.PINIT, tn)
        branch_eq = sys_idx.idx(HydVarEq.BRANCH, branch_idx)

        # Pressure residual: p_from - p_to + PL + height_correction
        p_from_abs = node_pit[fn, PINIT] + node_pit[fn, PAMB]
        p_to_abs = node_pit[tn, PINIT] + node_pit[tn, PAMB]
        fluid = get_fluid(net)
        rho = get_branch_real_density(fluid, node_pit, b_pit)
        height_diff = node_pit[fn, HEIGHT] - node_pit[tn, HEIGHT]
        const_height = rho * GRAVITATION_CONSTANT * height_diff / P_CONVERSION
        load = p_from_abs - p_to_abs + b_pit[:, PL] + const_height

        # Branch equation: 1 * δp_from - 1 * δp_to = load
        rows = np.concatenate([branch_eq, branch_eq])
        cols = np.concatenate([p_from_col, p_to_col])
        data = np.concatenate([np.ones(len(branch_idx)), -np.ones(len(branch_idx))])

        registry.add_override(ComponentEquations(
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            data=data.astype(np.float64),
            load_rows=branch_eq.astype(np.int32),
            load_data=load.astype(np.float64),
            mode=EqWriteMode.UNIQUE,
        ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        if f == t:
            return

        branch_idx = np.arange(f, t, dtype=np.int32)
        options = {"use_numba": get_net_option(net, "use_numba")}
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, _, _, _ = calculate_derivatives_branch_thermal(
            net, branch_pit[f:t], node_pit, branch_pit_old[f:t], options
        )

        b_pit = branch_pit[f:t]
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        t_out_col = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)
        tn_eq     = sys_idx.idx(ThermVarEq.NODE, tn)
        branch_eq = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # Outlet temperature fixed at t_flow_k
        registry.add_override(ComponentEquations(
            rows=branch_eq.astype(np.int32),
            cols=branch_eq.astype(np.int32),
            data=np.ones(len(branch_idx), dtype=np.float64),
            load_rows=branch_eq.astype(np.int32),
            load_data=np.zeros(len(branch_idx), dtype=np.float64),
            mode=EqWriteMode.UNIQUE,
        ))

        # Node energy balance at the receiving (to) node
        rows_node = np.concatenate([tn_eq, tn_eq])
        cols_node = np.concatenate([tn_eq, t_out_col])
        data_node = np.concatenate([dfnt_dt, dfnt_dtout])
        registry.add(ComponentEquations(
            rows_node.astype(np.int32), cols_node.astype(np.int32), data_node.astype(np.float64),
            tn_eq.astype(np.int32), fnt.astype(np.float64),
        ))
