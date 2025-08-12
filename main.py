# -*- coding: utf-8 -*-
# HISIM: Analytical Performance Modeling and Design Exploration of 2.5D/3D Heterogeneous Integration for AI Computing
# Copyright (c) Arizona State University & University of Minnesota

import os
import shutil
import csv
import time
import argparse
import sys

from Module_Compute.functions import imc_analysis
from Module_Thermal.thermal_model import thermal_model
from Module_Network.network_model import network_model
from Module_Compute.compute_IMC_model import compute_IMC_model
from Module_AI_Map.util_chip.util_mapping import model_mapping, load_ai_network, smallest_square_greater_than


def prepare_dirs():
    os.makedirs('./Results', exist_ok=True)
    os.makedirs('./Results/result_thermal', exist_ok=True)
    if os.path.exists('./Results/result_thermal/1stacks'):
        shutil.rmtree('./Results/result_thermal/1stacks')
    os.makedirs('./Results/result_thermal/1stacks')

def clean_cache(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_dir = os.path.join(root, dir_name)
                shutil.rmtree(pycache_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Design Space Search', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ai_model', type=str, default="vit", help='AI models:vit, gcn, resnet50, resnet110, vgg16, densenet121, test, roofline')
    parser.add_argument('--chip_architect', type=str, default="M3D", help='hardware architecture:M3D,M2D')
    parser.add_argument('--N_tier', type=int, default=4, help='how many tiers')
    parser.add_argument('--N_stack', type=int, default=1, help='Number of 3D stacks in 3.5D design') 
    parser.add_argument('--N_tile', type=int, default=256, help='how many tiles in tier')
    parser.add_argument('--xbar_size', type=int, default=128, help='crossbar size')
    parser.add_argument('--N_crossbar', type=int, default=1, help='how many crossbar in one PE')
    parser.add_argument('--N_pe', type=int, default=16, help='how many PEs in tile')
    parser.add_argument('--quant_weight', type=int, default=8, help='Precision of quantized weight of AI model')
    parser.add_argument('--quant_act', type=int, default=8, help='Precision of quantized activation of AI model')
    parser.add_argument('--freq_computing', type=float, default=1, help='Computing unit operation frequency')
    parser.add_argument('--fclk_noc', type=float, default=1, help='network data communication operation frequency')
    parser.add_argument('--tsvPitch', type=float, default=10, help='TSV pitch um')
    parser.add_argument('--W2d', type=int, default=32, help='Number of links of 2D NoC')
    parser.add_argument('--voltage', type=float, default=0.5, help='Operating Voltage in voltage')
    parser.add_argument('--placement_method', type=int, default=5, help='computing tile placement method')
    parser.add_argument('--routing_method', type=int, default=2, help='3D routing method')
    parser.add_argument('--percent_router', type=float, default=0.5, help='when data route from one tier to next tier, the system will choose how much percent routers for 3D communication')
    parser.add_argument('--router_times_scale', type=int, default=1, help='Scaling factor for time components of router: trc, tva, tsa, tst,tl, tenq')
    parser.add_argument('--thermal', action='store_true', help='Run thermal simulation or not')
    return parser.parse_args()

def configure(args):
    config = vars(args).copy()
    config['ai_model'] = args.ai_model
    config['thermal'] = args.thermal
    config['relu'] = True
    config['sigmoid'] = False
    config['result_list'] = [args.freq_computing, args.fclk_noc, args.xbar_size, args.N_tile, args.N_pe]
    config['result_dictionary'] = {}
    config['filename_results'] = "./Results/PPA.csv"
    config['sim_name'] = args.ai_model+"_placement_"+str(args.placement_method)

    config['placement_method'] = args.placement_method
    config['routing_method'] = args.routing_method
    config['percent_router'] = args.percent_router
    config['router_times_scale'] = args.router_times_scale
    if args.chip_architect == 'M2D':
        config['N_tier'] = 1
        config['N_stack'] = 1
    elif args.chip_architect == 'M3D':
        config['N_stack'] = 1
    return config

def mapping_step(config):
    network_params = load_ai_network(config['ai_model'])
    filename = "./Results/layer_inform.csv"
    total_tiles_real, tiles_each_tier, tiles_each_stack = model_mapping(
        filename, config['placement_method'], network_params,
        config['quant_act'], config['xbar_size'], config['N_crossbar'],
        config['N_pe'], config['quant_weight'], config['N_tile'],
        config['N_tier'], config['N_stack']
    )
    N_stack_real = len(tiles_each_stack)
    # placement_method:
    # 1: Tier/Chiplet Edge to Tier/Chiplet Edge connection 
    # 2: tile-to-tile 3D connection
    if config['placement_method'] == 2:
        N_tier_real = config['N_tier']
        N_tile_real = smallest_square_greater_than(max(max(tiles_each_tier)))
    elif config['placement_method'] == 1:
        N_tile_real = config['N_tile']
        if N_stack_real>1:
            N_tier_real=config['N_tier']
        else:
            N_tier_real = -(-total_tiles_real // config['N_tile'])  # Ceiling division
            # if total_tiles_real%config['N_tile']==0:
            #     N_tier_real=int(total_tiles_real//config['N_tile'])       
            # else:
            #     N_tier_real=int(total_tiles_real//config['N_tile'])+1
    else:
        print("Alert!!! Invalid placement method!")
        sys.exit()
    config['result_list'].append(N_tile_real)
    if N_tier_real > 4:
        print("Alert!!! Too many number of tiers!")
        sys.exit()
    config['result_list'].append(N_tier_real)
    if N_stack_real == 1:
        config['chip_architect'] = "M3D" if N_tier_real > 1 else "M2D"
    else:
        print("Alert!!! Too many number of stacks!")
        sys.exit()
    config['result_list'].append(N_stack_real)
    config.update({
        'N_tier_real': N_tier_real,
        'N_tile_real': N_tile_real,
        'N_stack_real': N_stack_real,
        'tiles_each_tier': tiles_each_tier,
        'tiles_each_stack': tiles_each_stack,
        'network_params': network_params,
        'total_tiles_real': total_tiles_real
    })

def compute_step(config):
    args = config
    N_tier_real, computing_data, area_single_tile, voltage, total_model_L, result_list, out_peripherial, A_peri = compute_IMC_model(
        args['xbar_size'], args['voltage'], args['freq_computing'],
        args['quant_act'], args['quant_weight'], args['N_crossbar'], args['N_pe'],
        args['N_tier_real'], args['N_stack_real'], args['N_tile'], args['result_list'],
        args['result_dictionary'], args['network_params'], args['relu']
    )
    config.update({
        'N_tier_real': N_tier_real,
        'computing_data': computing_data,
        'area_single_tile': area_single_tile,
        'voltage': voltage,
        'total_model_L': total_model_L,
        'result_list': result_list,
        'out_peripherial': out_peripherial,
        'A_peri': A_peri
    })

def network_step(config):
    args = config
    chiplet_num, tier_2d_hop_list_power, tier_3d_hop_list_power, single_router_area, mesh_edge, result_list = network_model(
        args['N_tier_real'], args['N_stack_real'], args['N_tile'], args['N_tier'],
        args['computing_data'], args['placement_method'], args['percent_router'],
        args['chip_architect'], args['tsvPitch'], args['area_single_tile'],
        args['result_list'], args['result_dictionary'], args['voltage'],
        args['fclk_noc'], args['total_model_L'], args['router_times_scale'],
        args['tiles_each_tier'], args['routing_method'], args['W2d']
    )
    config.update({
        'chiplet_num': chiplet_num,
        'tier_2d_hop_list_power': tier_2d_hop_list_power,
        'tier_3d_hop_list_power': tier_3d_hop_list_power,
        'single_router_area': single_router_area,
        'mesh_edge': mesh_edge,
        'result_list': result_list
    })

def thermal_step(config):
    args = config
    peak_temp = thermal_model(
        args['thermal'], args['chip_architect'], args['chiplet_num'],
        args['N_tile'], args['placement_method'], args['tier_2d_hop_list_power'],
        args['tier_3d_hop_list_power'], args['area_single_tile'],
        args['single_router_area'], args['mesh_edge'], args['sim_name']
    )
    config['peak_temp'] = peak_temp
    config['result_list'].append(peak_temp)
    config['result_list'].append(args['placement_method'])
    config['result_list'].append(args['percent_router'])

def write_results(config):
    result_list = config['result_list']
    with open(config['filename_results'], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tile(real)","N_tier", "N_Stack","W2d","W3d","Computing_latency", "Computing_energy","compute_area","chip_area","chip_Architecture","2d NoC latency","3d NoC latency","2.5d NoC latency", "network_latency","2d NoC energy","3d NoC energy","2.5d NoC energy","network_energy","rcc","TFLOPS", "2D_3D_NoC_power","2_5D_power","2d_3d_router_area","peak_temperature","placement_method","percent_router"])
        writer.writerow(result_list)

def main():
    prepare_dirs()
    args = parse_args()
    config = configure(args)

    print("=============== Start HISIM Simulation ===============","\n")
    #---------------------------------------------------------------------#
    #     Mapping: from AI model -> hardware mapping                      #
    #---------------------------------------------------------------------#
    start = time.time()
    print("--------------------------------------------------------")
    print("start mapping", args.ai_model, "...\n")
    mapping_step(config)
    #---------------------------------------------------------------------#
    #     Computing: generate PPA for IMC/GPU/CPU/ASIC computing units    #
    #---------------------------------------------------------------------#
    compute_start = time.time()
    compute_step(config)
    compute_end = time.time()
    print("\nComputing model simulation time is:", compute_end - compute_start, "s")
    print("--------------------------------------------------------")
    #---------------------------------------------------------------------#
    #     Network: generate PPA for NoC/NoP (2D/2.5D/3D/heterogeneous)    #
    #---------------------------------------------------------------------#
    network_start = time.time()
    network_step(config)
    network_end = time.time()
    print("\nThe NOC simulation time is:", network_end - network_start, "s")
    print("The total simulation time is:", network_end - start, "s\n")
    #---------------------------------------------------------------------------------#
    #     Thermal: generate temperature of the chip (based on power,area)             #
    #---------------------------------------------------------------------------------#
    if args.thermal:
        print("--------------------------------------------------------")
        print("Thermal Analysis Results")
        print("--------------------------------------------------------")
        thermal_step(config)
    write_results(config)

    clean_cache(os.getcwd())

if __name__ == "__main__":
    main()
