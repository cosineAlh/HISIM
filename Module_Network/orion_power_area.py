import numpy as np
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import matplotlib.pyplot as plt
import pandas as pd

# 2007 ITRS predictions for a 32nm high-performance library
H_INVD2 = 8
W_INVD2 = 3
H_DFQD1 = 8
W_DFQD1 = 16
H_ND2D1 = 8
W_ND2D1 = 3
H_SRAM = 8
W_SRAM = 6
Vdd = 0.9
R = 606.321
IoffP = 0.00000102
IoffN = 0.00000102
IoffSRAM = 0.00000032
Cg_pwr = 0.000000000000000534
Cd_pwr = 0.000000000000000267
Cgdl = 0.0000000000000001068
Cg = 0.000000000000000534
Cd = 0.000000000000000267
LAMBDA = 0.016
MetalPitch = 0.000080
Rw = 0.0435644

Ci_delay = 3 * (Cg + Cgdl)
Co_delay = 3 * Cd
Ci = (1.0 + 2.0) * Cg_pwr
Co = (1.0 + 2.0) * Cd_pwr
FO4 = R * (3.0 * Cd + 12 * Cg + 12 * Cgdl)
tCLK = 20 * FO4
fCLK = 1.0 / tCLK
ChannelPitch = 2.0 * MetalPitch
CrossbarPitch = 2.0 * MetalPitch
numVC = 3
buf_size = 10
depthVC = 10
output_buffer_size = 1
input_switch = 6
output_switch = 6


def interconnect_type(type):
    """Return interconnect parameters for TSV or 2D_NoC."""
    if type == "TSV":
        # TSV wire optimization
        Cw_gnd = 1.509e-14
        Cw_cpl = 1.509e-14
        K, M, N = 1, 1, 1
        Cw = 2.0 * (Cw_cpl + Cw_gnd)
        wire_length = 0.05
    elif type == "2D_NoC":
        # 2D NoC optimization
        Cw_gnd = 2.67339e-13
        Cw_cpl = 2.67339e-13
        K, M, N = 8.1, 2, 1
        Cw = 2.0 * (Cw_cpl + Cw_gnd)
        wire_length = 2.0
    else:
        raise ValueError(f"Unknown interconnect type: {type}")
    return K, M, N, wire_length, Cw

#---------------------------------------------------------------------#
# Channel power and area
#---------------------------------------------------------------------#
def Power_Module_powerRepeatedWire(L, K, M, N, Cw):
    """Calculate dynamic power for repeated wire segments."""
    segments = M * N
    Ca = K * (Ci + Co) + Cw * (L / segments)
    Pa = 0.5 * Ca * Vdd**2 * fCLK
    return Pa * segments

def Power_Module_powerWireClk(M, W, Cw):
    """Calculate clock wire power."""
    columns = H_DFQD1 * MetalPitch / ChannelPitch
    clockLength = W * ChannelPitch
    Cclk = (1 + 5.0/16.0 * (1 + Co_delay / Ci_delay)) * (clockLength * Cw * columns + W * Ci_delay)
    return M * Cclk * Vdd**2 * fCLK

def Power_Module_powerRepeatedWireLeak(K, M, N):
    """Calculate leakage power for repeated wire segments."""
    Pl = K * 0.5 * (IoffN + 2.0 * IoffP) * Vdd
    return Pl * M * N

def Power_Module_powerWireDFF(M, W, alpha=1):
    """Calculate DFF power for wire."""
    Cdin = 2 * 0.8 * (Ci + Co) + 2 * (2.0/3.0 * 0.8 * Co)
    Cclk = 2 * 0.8 * (Ci + Co) + 2 * (2.0/3.0 * 0.8 * Cg_pwr)
    Cint = (alpha * 0.5) * Cdin + alpha * Cclk
    return Cint * M * W * Vdd**2 * fCLK

def Power_Module_calcChannel(channel_width, K, M, N, wire_length, Cw):
    """Calculate channel power and area."""
    wire_power = Power_Module_powerRepeatedWire(wire_length, K, M, N, Cw) * channel_width
    clk_power = Power_Module_powerWireClk(M, channel_width, Cw)
    dff_power = Power_Module_powerWireDFF(M, channel_width, alpha=1)
    leak_power = Power_Module_powerRepeatedWireLeak(K, M, N) * channel_width
    area = Power_Module_areaChannel(K, M, N, channel_width)
    return wire_power, clk_power, dff_power, leak_power, area

def Power_Module_areaChannel(K, N, M, channel_width):
    """Calculate channel area."""
    Adff = M * W_DFQD1 * H_DFQD1
    Ainv = M * N * (W_INVD2 + 3 * K) * H_INVD2
    return channel_width * (Adff + Ainv) * MetalPitch**2

#---------------------------------------------------------------------#
# Memory power and area
#---------------------------------------------------------------------#
def Power_Module_calcBuffer(channel_width, input_switch, Cw):
    """Calculate buffer power and area."""
    depth = numVC * depthVC
    Pleak = depth * IoffSRAM * Vdd
    inputArea = Power_Module_areaInputModule(depth, channel_width) * input_switch
    Pwl = Power_Module_powerWordLine(channel_width, depth, channel_width, Cw)
    Prd = Power_Module_powerMemoryBitRead(depth, Cw) * channel_width
    Pwr = Power_Module_powerMemoryBitWrite(depth, Cw) * channel_width
    inputReadPower = Pwl + Prd
    inputWritePower = Pwl + Pwr
    return inputReadPower, inputWritePower, Pleak, inputArea

def Power_Module_areaInputModule(Words, channel_width):
    """Calculate input module area."""
    Asram = channel_width * H_SRAM * Words * W_SRAM
    return Asram * MetalPitch**2

def Power_Module_powerWordLine(memoryWidth, memoryDepth, channel_width, Cw):
    """Calculate wordline power."""
    Ccell = 2 * (4.0 * LAMBDA) * Cg_pwr + 6 * MetalPitch * Cw
    Cwl = memoryWidth * Ccell
    Warray = 8 * MetalPitch + memoryDepth
    x = 1.0 + (5.0/16.0) * (1 + Co / Ci)
    Cpredecode = x * (Cw * Warray * Ci)
    Cdecode = x * Cwl
    Harray = 6 * memoryWidth * MetalPitch
    y = (1 + 0.25) * (1 + Co / Ci)
    Cprecharge = y * (Cw * Harray + 3 * channel_width * Ci)
    Cwren = y * (Cw * Harray + 2 * channel_width * Ci)
    Cbd = Cprecharge + Cwren
    Cwd = 2 * Cpredecode + Cdecode
    return (Cbd + Cwd) * Vdd**2 * fCLK

def Power_Module_powerMemoryBitRead(memoryDepth, Cw):
    """Calculate memory bit read power."""
    Ccell = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw
    Cbl = memoryDepth * Ccell
    return Cbl * Vdd**2 * fCLK

def Power_Module_powerMemoryBitWrite(memoryDepth, Cw):
    """Calculate memory bit write power."""
    Ccell = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw
    Cbl = memoryDepth * Ccell
    Ccc = 2 * (Co + Ci)
    return 0.5 * Ccc * Vdd**2 + Cbl * Vdd**2 * fCLK

#---------------------------------------------------------------------#
# Switch power and area
#---------------------------------------------------------------------#
def Power_Module_calcSwitch(channel_width, Cw):
    """Calculate switch power and area."""
    switchArea = Power_Module_areaCrossbar(input_switch, output_switch, channel_width)
    outputArea = Power_Module_areaOutputModule(output_switch, channel_width)
    switchPowerLeak = Power_Module_powerCrossbarLeak(channel_width, input_switch, output_switch, Cw)
    switchPower = channel_width * Power_Module_powerCrossbar(channel_width, input_switch, output_switch, Cw)
    switchPowerCtrl = Power_Module_powerCrossbarCtrl(channel_width, input_switch, input_switch, Cw)
    outputPowerClk = Power_Module_powerWireClk(1, channel_width, Cw)
    outputPower = Power_Module_powerWireDFF(1, channel_width, 1.0)
    outputCtrlPower = Power_Module_powerOutputCtrl(channel_width, Cw)
    return switchPower, switchPowerCtrl, switchPowerLeak, switchArea, outputPower, outputPowerClk, outputCtrlPower, outputArea

def Power_Module_areaCrossbar(Inputs, Outputs, channel_width):
    """Calculate crossbar area."""
    return (Inputs * channel_width * CrossbarPitch) * (Outputs * channel_width * CrossbarPitch)

def Power_Module_areaOutputModule(Outputs, channel_width):
    """Calculate output module area."""
    Adff = Outputs * W_DFQD1 * H_DFQD1
    return channel_width * Adff * MetalPitch**2

def Power_Module_powerCrossbarLeak(width, inputs, outputs, Cw):
    """Calculate crossbar leakage power."""
    Wxbar = width * outputs * CrossbarPitch
    Hxbar = width * inputs * CrossbarPitch
    CwIn = Wxbar * Cw
    CwOut = Hxbar * Cw
    Cxi = (1.0 / 16.0) * CwOut
    Cti = (1.0 / 16.0) * CwIn
    return 0.5 * (IoffN + 2 * IoffP) * width * (inputs * outputs * Cxi + inputs * Cti + outputs * Cti) / Ci

def Power_Module_powerCrossbar(width, inputs, outputs, Cw):
    """Calculate crossbar dynamic power."""
    Wxbar = width * outputs * CrossbarPitch
    Hxbar = width * inputs * CrossbarPitch
    CwIn = Wxbar * Cw
    CwOut = Hxbar * Cw
    Cxi = (1.0 / 16.0) * CwOut
    Cxo = 4.0 * Cxi * (Co_delay / Ci_delay)
    Cti = (1.0 / 16.0) * CwIn
    Cto = 4.0 * Cti * (Co_delay / Ci_delay)
    CinputDriver = 5.0 / 16.0 * (1 + Co_delay / Ci_delay) * (0.5 * Cw * Wxbar + Cti)
    Cin = CinputDriver + CwIn + Cti + (outputs * Cxi)
    Cout = CwOut + Cto + (inputs * Cxo)
    return 0.5 * (Cin + Cout) * Vdd**2 * fCLK

def Power_Module_powerCrossbarCtrl(width, inputs, outputs, Cw):
    """Calculate crossbar control power."""
    Wxbar = width * outputs * CrossbarPitch
    Hxbar = width * inputs * CrossbarPitch
    CwIn = Wxbar * Cw
    Cti = (5.0 / 16.0) * CwIn
    Cctrl = width * Cti + (Wxbar + Hxbar) * Cw
    Cdrive = (5.0 / 16.0) * (1 + Co_delay / Ci_delay) * Cctrl
    return (Cdrive + Cctrl) * Vdd**2 * fCLK

def Power_Module_powerOutputCtrl(width, Cw):
    """Calculate output control power."""
    Woutmod = width * ChannelPitch
    Cen = Ci
    Cenable = (1 + 5.0 / 16.0) * (1.0 + Co / Ci) * (Woutmod * Cw + width * Cen)
    return Cenable * Vdd**2 * fCLK

def power_summary_router(channel_width, input_switch, output_switch, hop, trc, tva, tsa, tst, tl, tenq, Q, N_chiplet, mesh_edge):
    """Summarize router power and area for given parameters."""
    type = "TSV" if input_switch == 6 else "2D_NoC"
    K, M, N, wire_length, Cw = interconnect_type(type)
    channel_wire_power, channel_clk_power, channel_DFFPower, channelLeakPower, channelArea = Power_Module_calcChannel(channel_width, wire_length, K, M, N, Cw)
    inputReadPower, inputWritePower, Pleak, inputArea = Power_Module_calcBuffer(channel_width, input_switch, Cw)
    (switchPower, switchPowerCtrl, switchPowerLeak, switchArea,
     outputPower, outputPowerClk, outputCtrlPower, outputArea) = Power_Module_calcSwitch(channel_width, Cw)

    Latency_cycle = hop * (trc + tva + tsa + tst + tl) + tenq * (Q / channel_width)

    if input_switch == 6:
        factor = mesh_edge * mesh_edge * (N_chiplet - 1)
        channel_wire_power *= (hop * tl + Q / channel_width) / Latency_cycle * factor
        channel_DFFPower *= (hop * tl + Q / channel_width) / Latency_cycle * factor
        inputReadPower *= tenq * (Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        inputWritePower *= tenq * (Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        switchPower *= hop * (trc + tva + tsa) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        switchPowerCtrl *= (hop * (trc + tva + tsa) + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        outputPower *= (hop * tst + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        outputCtrlPower *= (hop * tst + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
    elif input_switch == 5:
        factor = mesh_edge * (mesh_edge * 2 - 2) * N_chiplet
        channel_wire_power *= (hop * tl + Q / channel_width) / Latency_cycle * factor
        channel_DFFPower *= (hop * tl + Q / channel_width) / Latency_cycle * factor
        inputReadPower *= tenq * (Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        inputWritePower *= tenq * (Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        switchPower *= hop * (trc + tva + tsa) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        switchPowerCtrl *= (hop * (trc + tva + tsa) + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        outputPower *= (hop * tst + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet
        outputCtrlPower *= (hop * tst + Q / channel_width) / Latency_cycle * mesh_edge * mesh_edge * N_chiplet

    channel_clk_power *= (2 * mesh_edge * mesh_edge * N_chiplet + 2 * mesh_edge * mesh_edge * (N_chiplet - 1))
    outputPowerClk *= hop * tst / Latency_cycle * (2 * mesh_edge * mesh_edge * (N_chiplet - 1) + 2 * mesh_edge * mesh_edge * N_chiplet)
    channelArea *= (mesh_edge * mesh_edge * N_chiplet * 2 + 2 * mesh_edge * (mesh_edge * 2 - 2) * N_chiplet + 2 * mesh_edge * mesh_edge * (N_chiplet - 1))
    switchArea *= mesh_edge * mesh_edge * N_chiplet
    inputArea *= mesh_edge * mesh_edge * N_chiplet
    outputArea *= mesh_edge * mesh_edge * N_chiplet
    total_area_router = channelArea + switchArea + inputArea + outputArea

    total_dynamic_power = channel_wire_power + channel_clk_power + channel_DFFPower
    total_other_power = (inputReadPower + inputWritePower + switchPower + switchPowerCtrl + outputPower + outputPowerClk + outputCtrlPower)

    return total_area_router, total_dynamic_power, total_other_power

