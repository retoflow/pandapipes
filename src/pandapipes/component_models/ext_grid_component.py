# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.component_models.component_toolbox import build_pit_entries
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.idx_node import IdxNode
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, PitEntries, PitWriteMode, HydVarEq, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ExtGrid(NodeElementComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "ext_grid"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def sign(cls):
        return -1.

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

    @classmethod
    def get_node_col(cls):
        return "junction"

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("in_service", "bool"),
                ('type', dtype(object))]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids[cls.active_identifier()].values]
        if not len(ext_grids):
            return

        junction = ext_grids[cls.get_node_col()].values
        types = ext_grids.type.values
        junction_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        mask_p = np.isin(types, ["p", "pt"])
        mask_t = np.isin(types, ["t", "pt"])
        index_p = junction_lookup[junction[mask_p]]
        index_t = junction_lookup[junction[mask_t]]

        registry.add_override(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.PINIT, IdxNode.NODE_TYPE],
            [ext_grids.p_bar.values[mask_p], float(IdxNode.P)],
        ), mode=PitWriteMode.MEAN))
        registry.add_override(PitEntries(*build_pit_entries(
            index_t,
            [IdxNode.TINIT, IdxNode.NODE_TYPE_T],
            [ext_grids.t_k.values[mask_t], float(IdxNode.T)],
        ), mode=PitWriteMode.MEAN))
        registry.add_override(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.EXT_GRID_OCCURENCE],
            [1.]),
            mode=PitWriteMode.ADDITIVE))
        registry.add_override(PitEntries(*build_pit_entries(
            index_t,
            [IdxNode.EXT_GRID_OCCURENCE_T],
            [1.]),
            mode=PitWriteMode.ADDITIVE))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        slack_nodes = sys_idx.slack_nodes  # all P-type node_pit rows, sorted
        if not len(slack_nodes):
            return

        ranks = np.arange(len(slack_nodes), dtype=np.int32)

        # variables
        p_col     = sys_idx.idx(HydVarEq.PINIT,         slack_nodes)
        slack_col = sys_idx.idx(HydVarEq.MDOTSLACKINIT, ranks)

        # equation position slack
        slack_eq = sys_idx.idx(HydVarEq.SLACK, ranks)

        # system matrix slack: pressure fix — δPINIT = 0 (override)
        rows_slack = slack_eq.astype(np.int32)
        cols_slack = p_col.astype(np.int32)
        data_slack = np.ones(len(slack_eq), dtype=np.float64)
        load_rows_slack = slack_eq.astype(np.int32)
        load_slack = np.zeros(len(slack_eq), dtype=np.float64)

        # equation position node
        n_eq = sys_idx.idx(HydVarEq.NODE, slack_nodes)

        # system matrix node: MDOTSLACKINIT participates in mass balance
        rows_node = n_eq.astype(np.int32)
        cols_node = slack_col.astype(np.int32)
        data_node = np.ones(len(n_eq), dtype=np.float64)
        load_rows_node = n_eq.astype(np.int32)
        load_node = node_pit[slack_nodes, IdxNode.MDOTSLACKINIT].astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_slack,
            cols=cols_slack,
            data=data_slack,
            load_rows=load_rows_slack,
            load_data=load_slack,
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
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids[cls.active_identifier()].values]
        if not len(ext_grids):
            return

        junction = ext_grids[cls.get_node_col()].values
        types = ext_grids.type.values
        mask_t = np.isin(types, ["t", "pt"])

        junction_lookup = get_lookup(net, "node", "index_active_heat_transfer")[
            cls.get_connected_node_type().table_name()
        ]
        ext_nodes = junction_lookup[junction[mask_t].astype(np.int32)]
        ext_nodes = ext_nodes[ext_nodes != -1]  # drop disconnected, sort to match infeed_nodes order

        if not len(ext_nodes):
            return

        infeed_mask = node_pit[:, IdxNode.INFEED].astype(bool)
        infeed_nodes = np.where(infeed_mask)[0].astype(np.int32)

        if not len(infeed_nodes):
            return

        # variables
        t_col = sys_idx.idx(ThermVarEq.TINIT, ext_nodes)

        # equation position node
        n_eq = sys_idx.idx(ThermVarEq.NODE, infeed_nodes)

        # system matrix node
        rows_node = n_eq.astype(np.int32)
        cols_node = t_col.astype(np.int32)
        data_node = np.ones(len(n_eq), dtype=np.float64)
        load_rows_node = n_eq.astype(np.int32)
        load_node = np.zeros(len(n_eq), dtype=np.float64)

        registry.add_override(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
            mode=EqWriteMode.MEAN,
        ))

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param mode:
        :type mode:
        :return: No Output.
        """
        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        res_table = net["res_" + cls.table_name()]

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        p_grids = np.isin(ext_grids.type.values, ["p", "pt"]) & ext_grids.in_service.values
        junction = cls.get_connected_junction(net).values
        # get indices in internal structure for junctions in ext_grid tables which are "active"
        eg_nodes = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()][
            junction[p_grids]]
        node_uni, inverse_nodes, counts = np.unique(eg_nodes, return_counts=True, return_inverse=True)
        sum_mass_flows = node_pit[node_uni, IdxNode.MDOTSLACKINIT]

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[p_grids] = \
            cls.sign() * (sum_mass_flows / counts)[inverse_nodes]
        return res_table, ext_grids, node_pit, branch_pit
