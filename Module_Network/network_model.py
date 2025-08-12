import math
import numpy as np
import matplotlib.pyplot as plt

from Module_Network.orion_power_area import power_summary_router
from Module_AI_Map.util_chip.util_mapping import create_tile

def assign_tiles(computing_data, N_tier_real, N_stack_real, N_tile, placement_method):
    mesh_edge = int(math.sqrt(N_tile))
    layer_start_tile_tier = [[0]*N_tier_real]*N_stack_real
    layer_start_tile = 0
    tile_total = []
    count_tier = 0
    for idx, layer in enumerate(computing_data):
        prev_layer = computing_data[idx-1] if idx > 0 else layer
        if placement_method == 2:
            if prev_layer[15] != layer[15]:
                count_tier = 0
            if count_tier < N_tier_real:
                layer_start_tile_tier[int(layer[15])][int(layer[9])] = 0
                count_tier += 1
            layer_start_tile = layer_start_tile_tier[int(layer[15])][int(layer[9])]
        elif placement_method == 1:
            if layer[15] >= 1 and prev_layer[15] != layer[15]:
                layer_start_tile = 0
            elif layer[9] >= 1 and prev_layer[9] != layer[9]:
                layer_start_tile = 0
        else:
            print("Alert!!! Invalid placement method!")

        layer_end_tile = layer_start_tile + int(layer[1]) - 1
        tile_index = []
        for tile_num in range(layer_start_tile, layer_end_tile + 1):
            x_idx, y_idx = divmod(tile_num, mesh_edge)
            if x_idx % 2 == 1:
                y_idx = mesh_edge - y_idx - 1
            if placement_method != 2 and layer[9] % 2 == 1:
                x_idx = mesh_edge - x_idx - 1
                y_idx = mesh_edge - y_idx - 1
            tile_index.append([x_idx, y_idx, int(layer[9]), int(layer[15])])
        tile_index = np.array(tile_index)

        # Activation Q for next layer
        each_tile_activation_Q = int(computing_data[idx + 1][8] / layer[1]) if idx < len(computing_data) - 1 else 0
        tile_index = np.vstack([tile_index, [each_tile_activation_Q] * 4])
        tile_total.append(tile_index)
        if placement_method == 2:
            layer_start_tile_tier[int(layer[15])][int(layer[9])] = layer_end_tile + 1
        else:
            layer_start_tile = layer_end_tile + 1
    return tile_total, mesh_edge

def build_empty_tile_map(N_stack_real, chiplet_num, mesh_edge):
    return [np.array([[x, y, tier_index, stack_index] for x in range(mesh_edge) for y in range(mesh_edge)]) for stack_index in range(N_stack_real) for tier_index in range(chiplet_num)]

def compute_communication_stats(tile_total, empty_tile_total, routing_method, percent_router, N_tile, N_stack_real):
    hop2d = hop3d = Q_3d = Q_2d = 0
    layer_Q = []
    layer_HOP_2d, layer_HOP_3d = [], []
    N_stack_real = N_stack_real
    hop2d_stack = [0] * N_stack_real
    hop3d_stack = [0] * N_stack_real
    Q_2d_stack = [0] * N_stack_real
    Q_3d_stack = [0] * N_stack_real
    stack_index = 0

    for i in range(len(tile_total) - 1):
        prev_hop2d, prev_hop3d = hop2d, hop3d
        curr, nxt = tile_total[i], tile_total[i + 1]
        if curr[0][3] != nxt[0][3]:
            for x in range(len(curr) - 1):
                hop2d += (abs(curr[x][0] - empty_tile_total[curr[x][2]][-1][0]) + 1) * 2
            Q_2d += curr[-1][3] * (len(nxt) - 1)
            layer_Q.append(curr[-1][3] * (len(curr) - 1))
            stack_index += 1
        else:
            if routing_method == 1:
                for x in range(len(curr) - 1):
                    for y in range(len(nxt) - 1):
                        hop2d += abs(curr[x][0] - nxt[y][0]) + abs(curr[x][1] - nxt[y][1]) + 1
                        hop3d += abs(curr[x][2] - nxt[y][2])
                if curr[0][2] != nxt[0][2]:
                    Q_3d += curr[-1][2] * (len(nxt) - 1)
                    layer_Q.append(curr[-1][2] * (len(curr) - 1))
                else:
                    Q_2d += curr[-1][2] * (len(nxt) - 1)
                    layer_Q.append(curr[-1][2] * (len(curr) - 1))
            elif routing_method == 2:
                for x in range(len(curr) - 1):
                    for y in range(len(nxt) - 1):
                        hop2d += abs(curr[x][0] - nxt[y][0]) + abs(curr[x][1] - nxt[y][1]) + 1
                        hop3d += abs(curr[x][2] - nxt[y][2]) * N_tile * percent_router
                if curr[0][2] != nxt[0][2]:
                    for x in range(len(curr) - 1):
                        for y in range(int(len(empty_tile_total[int(curr[x][2])] ) * percent_router)):
                            hop2d += (abs(curr[x][0] - empty_tile_total[int(curr[x][2])][y][0]) + abs(curr[x][1] - empty_tile_total[int(curr[x][2])][y][1]) + 1) * 2
                    Q_3d += int(curr[-1][2] * (len(nxt) - 1) / (N_tile * percent_router))
                    layer_Q.append(curr[-1][2] * (len(curr) - 1))
                    Q_2d += int(curr[-1][2] * (len(curr) - 1) / (N_tile * percent_router))
                else:
                    Q_2d += curr[-1][2] * (len(nxt) - 1)
                    layer_Q.append(curr[-1][2] * (len(curr) - 1))
            hop2d_stack[stack_index] += hop2d
            hop3d_stack[stack_index] += hop3d
            Q_2d_stack[stack_index] += Q_2d
            Q_3d_stack[stack_index] += Q_3d
        layer_HOP_2d.append(hop2d - prev_hop2d)
        layer_HOP_3d.append(hop3d - prev_hop3d)

    return hop2d, hop3d, Q_2d, Q_3d, layer_Q, layer_HOP_2d, layer_HOP_3d, hop2d_stack, hop3d_stack, Q_2d_stack, Q_3d_stack

def calculate_area_power_latency(
    chip_architect, N_tier_real, N_stack_real, N_tile, mesh_edge, tsvPitch, area_single_tile, hop2d, hop3d, Q_2d, Q_3d, voltage, scale_factor, chiplet_num, W2d, result_list,
    result_dictionary, fclk_noc, total_model_L, layer_HOP_2d, layer_HOP_3d, hop2d_stack, hop3d_stack, Q_2d_stack, Q_3d_stack, tiles_each_tier, tile_total, computing_data):
    # Bandwidth and area calculations
    W3d_assume = 32
    trc = tva = tsa = tst = tl = 1 * scale_factor
    tenq = 2 * scale_factor

    if chip_architect == "M3D" and N_tier_real != 1:
        channel_width = 4 / 5 * W2d + 1 / 5 * W3d_assume
        total_router_area, _, _ = power_summary_router(channel_width, 6, 6, hop3d, trc, tva, tsa, tst, tl, tenq, Q_3d, chiplet_num, mesh_edge)
    elif chip_architect in ("M2D") or N_tier_real == 1:
        channel_width = W2d
        total_router_area, _, _ = power_summary_router(channel_width, 5, 5, hop2d, trc, tva, tsa, tst, tl, tenq, Q_2d, chiplet_num, mesh_edge)
    single_router_area = total_router_area / (mesh_edge * mesh_edge * chiplet_num)
    edge_single_router = math.sqrt(single_router_area)
    edge_single_tile = math.sqrt(area_single_tile)

    num_tsv_io = int(edge_single_router / tsvPitch * 1000) ** 2 * 2
    W3d = num_tsv_io
    if chip_architect == "M3D" and N_tier_real != 1:
        channel_width = 4 / 5 * W2d + 1 / 5 * W3d
        total_router_area, _, _ = power_summary_router(channel_width, 6, 6, hop3d, trc, tva, tsa, tst, tl, tenq, Q_3d, chiplet_num, mesh_edge)
    single_router_area = total_router_area / (mesh_edge * mesh_edge * chiplet_num)
    edge_single_router = math.sqrt(single_router_area)

    chip_area = (edge_single_router + edge_single_tile) ** 2 * N_tile * N_stack_real
    result_list.append(chip_area)
    result_dictionary['chip area (mm2)'] = chip_area
    result_list.insert(8, W2d)
    result_list.insert(9, W3d)
    result_dictionary['W2d'] = W2d
    result_dictionary['W3d'] = W3d

    # Latency calculations
    L_booksim_3d = 0
    L_booksim_2d = (hop2d * (trc + tva + tsa + tst + tl) + tenq * (Q_2d / W2d)) / fclk_noc
    if chip_architect == "M3D" and N_tier_real != 1:
        L_booksim_3d = (hop3d * (trc + tva + tsa + tst + tl) + tenq * (Q_3d / W3d)) / fclk_noc
    L_booksim = L_booksim_2d + L_booksim_3d

    result_list.extend([chip_architect, L_booksim_2d, L_booksim_3d])
    result_dictionary['chip_Architecture'] = chip_architect
    result_dictionary['2d NoC latency (ns)'] = L_booksim_2d
    result_dictionary['3d NoC latency (ns)'] = L_booksim_3d

    # Power calculations
    chiplet_num_list = [sum(item != 0 for item in row) for row in tiles_each_tier]
    tier_2d_hop_list_power, tier_3d_hop_list_power = [], []
    total_2d_channel_power = total_2d_router_power = total_tsv_channel_power = total_3d_router_power = 0

    for stack_index in range(N_stack_real):
        tier_2d_hop_list, tier_3d_hop_list = [], []
        tier_total_2d_hop = tier_total_3d_hop = num_layer = 0
        for i in range(chiplet_num_list[stack_index]):
            for layer_index in range(len(tile_total) - 1):
                if computing_data[layer_index][9] == i and computing_data[layer_index][15] == stack_index:
                    tier_total_2d_hop += layer_HOP_2d[layer_index]
                    tier_total_3d_hop += layer_HOP_3d[layer_index]
                num_layer += 1
            tier_2d_hop_list.append(tier_total_2d_hop / num_layer / N_tile if num_layer else 0)
            tier_3d_hop_list.append(tier_total_3d_hop)
            tier_total_2d_hop = tier_total_3d_hop = num_layer = 0

        if chip_architect == "M3D" and chiplet_num_list[stack_index] != 1:
            _, tsv_power, router_3d_power = power_summary_router(W3d, 6, 6, hop3d_stack[stack_index], trc, tva, tsa, tst, tl, tenq, Q_3d_stack[stack_index], chiplet_num_list[stack_index], mesh_edge)
            _, channel_power, router_2d_power = power_summary_router(W2d, 5, 5, hop2d_stack[stack_index], trc, tva, tsa, tst, tl, tenq, Q_2d_stack[stack_index], chiplet_num_list[stack_index], mesh_edge)
        else:
            tsv_power = router_3d_power = 0
            _, channel_power, router_2d_power = power_summary_router(W2d, 5, 5, hop2d_stack[stack_index], trc, tva, tsa, tst, tl, tenq, Q_2d_stack[stack_index], chiplet_num_list[stack_index], mesh_edge)

        total_router_power_stack = router_3d_power + router_2d_power + channel_power
        if len(tier_2d_hop_list) != 1:
            tier_2d_hop_list[-1] = tier_2d_hop_list[-2]
            tier_3d_hop_list[-1] = tier_3d_hop_list[-2]
        else:
            tier_3d_hop_list[-1] = total_router_power_stack / chiplet_num_list[stack_index] if chiplet_num_list[stack_index] else 0

        tier_2d_hop_list_power_stack = [
            i * total_router_power_stack / chiplet_num_list[stack_index] / i * fclk_noc if i else 0
            for i in tier_2d_hop_list
        ]
        if chip_architect == "M3D" and chiplet_num_list[stack_index] != 1:
            tier_3d_hop_list_power_stack = [
                i * tsv_power / (chiplet_num_list[stack_index] - 1) / i * fclk_noc if i else 0
                for i in tier_3d_hop_list
            ]
        else:
            tier_3d_hop_list_power_stack = [0 for _ in tier_3d_hop_list]

        total_2d_channel_power += channel_power
        total_2d_router_power += router_2d_power
        total_tsv_channel_power += tsv_power
        total_3d_router_power += router_3d_power
        tier_3d_hop_list_power.append(tier_3d_hop_list_power_stack)
        tier_2d_hop_list_power.append(tier_2d_hop_list_power_stack)

    total_router_power = total_3d_router_power + total_2d_router_power + total_2d_channel_power
    energy_2d = (total_2d_channel_power + total_2d_router_power) * L_booksim_2d * fclk_noc
    energy_3d = (total_tsv_channel_power + total_3d_router_power) * L_booksim_3d * fclk_noc
    total_energy = energy_2d + energy_3d

    result_list.extend([L_booksim, energy_2d, energy_3d, total_energy])
    result_dictionary['network_latency (ns)'] = L_booksim
    result_dictionary['2d NoC energy (pJ)'] = energy_2d
    result_dictionary['3d NoC energy (pJ)'] = energy_3d
    result_dictionary['network_energy (pJ)'] = total_energy

    # Area of routers and channels
    wire_length_2d = 2
    wire_pitch_2d = 0.0045
    Num_routers = N_tile * sum(chiplet_num_list)
    Total_area_routers = single_router_area * Num_routers
    Total_channel_area = wire_length_2d * wire_pitch_2d * W2d
    flops = sum(layer[14] for layer in computing_data)
    total_power = (total_router_power + total_tsv_channel_power) * fclk_noc

    result_list.append(flops * 1e-3 / (L_booksim + total_model_L * 1e-9))
    result_dictionary['Throughput (TFLOPS/s)'] = flops * 1e-3 / (L_booksim + total_model_L*1e-9)
    result_list.append(total_model_L * 1e9 / L_booksim)
    result_dictionary['rcc (compute latency/communciation latency)'] = total_model_L * 1e9 / L_booksim
    result_list.append((total_router_power + total_tsv_channel_power) * fclk_noc * 1e-3)
    result_dictionary['2D_3D_NoC_power (W)'] = total_power * 1e-3
    result_list.append(Total_area_routers + Total_channel_area)
    result_dictionary['2d_3d_router_area (mm2)'] = Total_area_routers + Total_channel_area

    return {
        'single_router_area': single_router_area,
        'edge_single_router': edge_single_router,
        'edge_single_tile': edge_single_tile,
        'chip_area': chip_area,
        'W3d': W3d,
        'L_booksim': L_booksim,
        'L_booksim_2d': L_booksim_2d,
        'L_booksim_3d': L_booksim_3d,
        'energy_2d': energy_2d,
        'energy_3d': energy_3d,
        'total_energy': total_energy,
        'total_power': total_power,
        'chiplet_num_list': chiplet_num_list,
        'tier_2d_hop_list_power': tier_2d_hop_list_power,
        'tier_3d_hop_list_power': tier_3d_hop_list_power,
        'Total_area_routers': Total_area_routers,
        'Total_channel_area': Total_channel_area
    }

def visualize_tile_map(empty_tile_total, tile_total, N_stack_real, N_tier_real):
    fig = plt.figure(figsize=(8, 8))
    axes = [fig.add_subplot(1, N_stack_real, i + 1, projection='3d') for i in range(N_stack_real)]
    tile_spacing = 0.7
    cube_height = 0.0
    for i, ax in enumerate(axes):
        start = i * N_tier_real
        for item in empty_tile_total[start:start + N_tier_real]:
            for tile in item:
                idx = next((x for x in range(len(tile_total)) if any((tile == tile_total[x][y]).all() for y in range(len(tile_total[x]) - 1))), '')
                create_tile(ax, *tile[:3], tile_spacing, tile_spacing, cube_height, 'lightblue', idx)
        ax.set_axis_off()
        # ax.set_title(f'3D Stack {i + 1}', fontsize=16)
    plt.savefig('./Results/tile_map.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    # plt.show()

def network_model(N_tier_real, N_stack_real, N_tile, N_tier, computing_data, placement_method, percent_router, chip_architect, tsvPitch, area_single_tile, result_list, result_dictionary, voltage, fclk_noc, total_model_L, scale_factor, tiles_each_tier, routing_method, W2d):
    chiplet_num = N_tier_real
    tile_total, mesh_edge = assign_tiles(computing_data, N_tier_real, N_stack_real, N_tile, placement_method)
    empty_tile_total = build_empty_tile_map(N_stack_real, chiplet_num, mesh_edge)

    hop2d, hop3d, Q_2d, Q_3d, layer_Q, layer_HOP_2d, layer_HOP_3d, hop2d_stack, hop3d_stack, Q_2d_stack, Q_3d_stack = compute_communication_stats(tile_total, empty_tile_total, routing_method, percent_router, N_tile, N_stack_real)

    print("\n--------------------------------------------------------")
    print("Network Performance Results")
    print("--------------------------------------------------------")
    print("Network Info:")
    print("Total data volumn for 2d communication:", Q_2d)
    print("Total HOP for 2d communication:", hop2d)
    if chip_architect == "M3D":
        print("Total data volume for 3d communication:", Q_3d)
        print("Total HOP for 3d communication:", hop3d)
    area_power_latency = calculate_area_power_latency(chip_architect, N_tier_real, N_stack_real, N_tile, mesh_edge, tsvPitch, area_single_tile, hop2d, hop3d, Q_2d, Q_3d, voltage, scale_factor, chiplet_num, W2d, result_list, result_dictionary, fclk_noc, total_model_L, layer_HOP_2d, layer_HOP_3d, hop2d_stack, hop3d_stack, Q_2d_stack, Q_3d_stack, tiles_each_tier, tile_total, computing_data)
    print("---------------------------------------------------------")
    print("Area Report:")
    print("Single tile area:", round(area_power_latency['edge_single_tile'] ** 2, 5), "mm2")
    print("Single router area:", round(area_power_latency['single_router_area'], 5), "mm2")
    print("Edge length single router:", round(area_power_latency['edge_single_router'], 5), "mm")
    print("Edge length single tile:", round(area_power_latency['edge_single_tile'], 5), "mm")
    print("Total 3d stack area:", round((area_power_latency['edge_single_router'] + area_power_latency['edge_single_tile']) ** 2 * N_tile * N_stack_real, 5), "mm2")
    print("---------------------------------------------------------")
    print('Latency Report:')
    print("2D NoC channel width (W2d):", W2d)
    print("3D TSV channel width (W3d):", area_power_latency['W3d'])
    print("Network total energy:", round(area_power_latency['total_energy'], 5), "pJ")
    print("Network power:", round((area_power_latency['total_power']), 5), "mW")
    print("Total NoC latency:", round(area_power_latency['L_booksim'], 5), "ns")
    print("Computing latency:", round(total_model_L * 1e9, 5), "ns")
    print("Total system latency:", round(area_power_latency['L_booksim'] + total_model_L * 1e9, 5), "ns")

    visualize_tile_map(empty_tile_total, tile_total, N_stack_real, N_tier_real)

    return area_power_latency['chiplet_num_list'], area_power_latency['tier_2d_hop_list_power'], area_power_latency['tier_3d_hop_list_power'], area_power_latency['single_router_area'], mesh_edge, result_list
