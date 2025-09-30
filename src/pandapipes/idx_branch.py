# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.idx import IndexMeta

class IdxBranch(metaclass=IndexMeta):
    # branch types
    PC = 1  # Pressure controller branch
    
    # branch indices
    TABLE_IDX = 0  # number of the table that this branch belongs to
    ELEMENT_IDX = 1  # index of the element that this branch belongs to (within the given table)
    BRANCH_TYPE = 2  # branch type relevant for the pressure controller
    FROM_NODE = 3  # f, from bus number
    TO_NODE = 4  # t, to bus number
    ACTIVE = 5
    LENGTH = 6  # Pipe length in [m]
    D = 7  # Diameter in [m]
    AREA = 8  # Area in [m²]
    K = 9  # Pipe roughness in [m]
    RE = 10 # Reynolds number
    LAMBDA = 11  # Lambda
    LOSS_COEFFICIENT = 12
    ALPHA = 13  # Slot for heat transfer coefficient
    QEXT = 14  # heat input into the branch [W]
    TEXT = 15 # temperature of surrounding [K]
    PL = 16 # Pressure lift [bar]
    TL = 17 # Temperature lift [K]
    
    MDOTINIT = 18  # mass in  [m/s]
    MDOTINIT_T = 19
    FROM_NODE_T_SWITCHED = 20 # flag to indicate if the from and to node are switched in the thermal calculation
    TOUTINIT = 21  # Internal slot for outlet pipe temperature
    FLOW_RETURN_CONNECT = 22 # Make sure that return and flow side are connected to the central pump, respectively
    
    DF_DM_B = 23  # Slot for the derivative by mass
    DF_DP_F_B = 24  # Slot for the derivative by pressure from_node
    DF_DP_T_B = 25  # Slot for the derivative by pressure to_node
    DF_DMSLACK_F_B = 25  # Slot for the derivative by pressure to_node
    DF_DMSLACK_T_B = 25  # Slot for the derivative by pressure to_node
    
    DF1_DM_F_N = 23  # Slot for the derivative by mass
    DF1_DM_T_N = 23  # Slot for the derivative by mass
    
    DF2_DM_F_N = 23  # Slot for the derivative by mass
    DF2_DM_T_N = 23  # Slot for the derivative by mass

    F_B = 27  # Slot for the load vector for the branches
    F1_F_N = 28  # Slot for the first node equation of the from nodes connected to branch
    F1_T_N = 29  # Slot for the first node equation of the to nodes connected to branch
    F2_F_N = 28  # Slot for the second node equation of the from nodes connected to branch
    F2_T_N = 29  # Slot for the second node equation of the to nodes connected to branch
    
    M_COL = 30
    
    ROW_B = 31
    
    JAC_DERIV_DT = 30
    JAC_DERIV_DTOUT = 31
    JAC_DERIV_DT_NODE = 32  # Slot for the node equation derivative of T for the nodes branch is connected to
    JAC_DERIV_DTOUT_NODE = 33  # Slot for the node equation derivative of T for the corresponding branch
    LOAD_VEC_B_T = 34
    LOAD_VEC_N_FROM_T = 35 # Slot for the load vector of the from nodes connected to branch
    LOAD_VEC_N_TO_T = 36 # Slot for the load vector of the to nodes connected to branch
    
    T_OUT_OLD = 37

    branch_cols = 38