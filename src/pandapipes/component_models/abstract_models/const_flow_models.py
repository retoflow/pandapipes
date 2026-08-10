# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.idx_node import LOAD, ELEMENT_IDX
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.pf.system_index import PitEntries, PitWriteMode, ComponentEquations, HydVarEq
from pandapipes.pf.derivative_calculation import calculate_load_hydraulic


class ConstFlow(NodeElementComponent):

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def sign(cls):
        raise NotImplementedError()

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("mdot_kg_per_s", "f8"),
                ("scaling", "f8"),
                ("in_service", "bool"),
                ("type", dtype(object))]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        loads = net[cls.table_name()]
        helper = loads.in_service.values * loads.scaling.values * cls.sign()
        mf = np.nan_to_num(loads.mdot_kg_per_s.values)
        mass_flow_loads = mf * helper
        juncts, loads_sum = _sum_by_group(get_net_option(net, "use_numba"), loads.junction.values,
                                          mass_flow_loads)
        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        index = junction_idx_lookups[juncts].astype(np.int32)
        registry.add(PitEntries(
            index,
            np.full(len(index), LOAD, dtype=np.int32),
            loads_sum.astype(np.float64),
            mode=PitWriteMode.ADDITIVE,
        ))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        index, loads_sum = calculate_load_hydraulic(
            net, net[cls.table_name()], cls.sign(), cls.get_connected_node_type().table_name())
        if not len(index):
            return

        # equation positions node
        n_eq = sys_idx.idx(HydVarEq.NODE, index)

        # load vector node
        registry.add(ComponentEquations(
            rows=np.empty(0, dtype=np.int32),
            cols=np.empty(0, dtype=np.int32),
            data=np.empty(0, dtype=np.float64),
            load_rows=n_eq.astype(np.int32),
            load_data=loads_sum.astype(np.float64),
        ))

    @classmethod
    def get_result_table(cls, net):
        """Get results.

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

        :param mode:
        :type mode:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        res_table = net["res_" + cls.table_name()]

        loads = net[cls.table_name()]

        is_loads = loads.in_service.values
        fj, tj = get_lookup(net, "node", "from_to")[cls.get_connected_node_type().table_name()]
        junct_pit = net["_pit"]["node"][fj:tj, :]
        nodes_connected_hyd = get_lookup(net, "node", "active_hydraulics")[fj:tj]
        is_juncts = np.isin(loads.junction.values, junct_pit[nodes_connected_hyd, ELEMENT_IDX])

        is_calc = is_loads & is_juncts
        res_table["mdot_kg_per_s"].values[is_calc] = loads.mdot_kg_per_s.values[is_calc] \
            * loads.scaling.values[is_calc]
