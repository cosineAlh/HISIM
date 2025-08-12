import math
import collections
import torch

from Module_Thermal.util import *

def _get_power_dict(power_list):
    power_dict = {}
    for i, power in enumerate(power_list):
        power_dict[i] = power
    return power_dict

def _get_dict_k(chip_architect, alpha):
    dict_k = dict()
    if chip_architect == "M3D":
        dict_k.update({
            'k_imc_0': 110/alpha, 'k_imc_1': 142.8/alpha, 'k_imc_2': 4/alpha,
            'k_r_0': 110/alpha, 'k_r_1': 142.8/alpha, 'k_r_2': 4/alpha,
            'k_tsv_0': 142.8/alpha, 'k_tsv_1': 200/alpha, 'k_tsv_2': 7.9/alpha,
            'cu': 398/alpha, 'air': 0.0243/alpha, 'subs': 142.8/alpha
        })
    elif chip_architect == "M2D":
        dict_k.update({
            'k_imc_0': 110/alpha, 'k_imc_1': 142.8/alpha, 'k_imc_2': 4/alpha,
            'k_r_0': 110/alpha, 'k_r_1': 142.8/alpha, 'k_r_2': 4/alpha,
            'k_tsv_0': 110/alpha, 'k_tsv_1': 142.8/alpha, 'k_tsv_2': 4/alpha,
            'cu': 398/alpha, 'air': 0.0243/alpha, 'subs': 142.8/alpha
        })
    return dict_k

def _get_dict_size(imc_size, r_size):
    return {
        "imc": (imc_size, imc_size),
        "r": (r_size, r_size),
        "tsv0": (r_size, imc_size),
        "tsv1": (imc_size, r_size)
    }

def _get_dict_z():
    return {
        'heatsink': 40,
        'heatspread': 20,
        'device': (0.002, 0.1, 0.02),
        'subs': 1,
        'air': 50
    }

def thermal_model(thermal, chip_architect, chiplet_num_list, N_tile, placement_method, tier_2d_hop_list_power_1, tier_3d_hop_list_power_1, area_single_tile, single_router_area, mesh_edge, sim_name):
    #---------------------------------------------------------------------#
    # Thermal simulation (based on power, area)
    #---------------------------------------------------------------------#
    if not thermal:
        return 'NA'

    if chip_architect in ("M3D", "M2D"):
        chiplet_num = chiplet_num_list[0]
        tier_2d_hop_list_power = tier_2d_hop_list_power_1[0]
        tier_3d_hop_list_power = tier_3d_hop_list_power_1[0]
        torch.set_printoptions(threshold=50_000)

        alpha = 3.5
        power_router = _get_power_dict(tier_2d_hop_list_power)
        power_tsv = _get_power_dict(tier_3d_hop_list_power)
        dict_k = _get_dict_k(chip_architect, alpha)

        imc_size = math.sqrt(area_single_tile) / 1000
        r_size = math.sqrt(single_router_area) / 1000
        dict_size = _get_dict_size(imc_size, r_size)
        dict_z = _get_dict_z()
        heatsinkair_resolution = 0.5

        devicemap = collections.defaultdict(list)
        devicemap['1stacks'].append([(0, 1, 'heatsink')])
        devicemap['1stacks'].append([(0, 1, 'heatspread')])
        for _ in range(chiplet_num):
            devicemap['1stacks'].append([(0, 1, 'device')])
        devicemap['1stacks'].append([(0, 1, 'subs')])
        devicemap['1stacks'].append([(0, 1, 'air')])

        numofdevicelayer = {'1stacks': chiplet_num}
        devicemap_sanitycheck(devicemap)
        xdim, _ = get_unitsize(dict_size, mesh_edge)
        cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict, cube_layertype_dict = create_cube(dict_size, dict_z, dict_k, xdim, devicemap, heatsinkair_resolution, mesh_edge)
        cube_power_dict = load_power(dict_z, devicemap, cube_n_dict, power_tsv, power_router, numofdevicelayer, cube_layertype_dict, mesh_edge, chiplet_num, placement_method)
        cube_G_dict = get_conductance_G(cube_geo_dict, cube_k_dict, cube_z_dict)
        peak_temp = solver(cube_G_dict, cube_n_dict, cube_power_dict, cube_layertype_dict, xdim, sim_name)
    else:
        peak_temp = 'NA'

    return peak_temp
