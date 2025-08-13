import numpy as np
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import matplotlib.pyplot as plt
import math
import pandas as pd
import collections
import time


#====================================================================================================
def plot_im(plot_data, save_name, vmin, vmax):
    """Plot and save a thermal map image."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(plot_data, cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.axis('off')  # Hide axes
    fig.savefig(save_name, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()

def checkneighbours(idxx, idxy, idxz, numx, numy, numz):
    """Return True if (idxx, idxy, idxz) is within bounds."""
    return 0 <= idxx < numx and 0 <= idxy < numy and 0 <= idxz < numz

def id_(idxx, idxy, idxz, numx, numy, numz):
    """Convert 3D indices to a linear index."""
    return idxz * (numx * numy) + idxx * numy + idxy

def iffallintodevicerange(xpos, devicelayer, tiles_edges_in_tier):
    """Check if xpos falls within any device segment in devicelayer."""
    for partstart, partend, parttype in devicelayer:
        if parttype == 'device':
            start = partstart * tiles_edges_in_tier * 2
            end = partend * tiles_edges_in_tier * 2
            if start <= xpos < end:
                return True
    return False

def findparttype(xpos, layer, tiles_edges_in_tier):
    """Return the part type for xpos in the given layer."""
    for partstart, partend, parttype in layer:
        start = partstart * tiles_edges_in_tier * 2
        end = partend * tiles_edges_in_tier * 2
        if start <= xpos < end:
            return parttype
    return None

# ====================================================================================================
def devicemap_sanitycheck(devicemap):
    """Sanity check for device map structure."""
    def check_layer(layer, total_width):
        prev = layer[0]
        total = prev[1] - prev[0]
        for part in layer[1:]:
            assert prev[1] == part[0], "Layer segments must be contiguous."
            assert part[1] > part[0], "Layer segment end must be greater than start."
            assert (part[1] - part[0]) % 0.5 == 0, "Layer segment width must be a multiple of 0.5."
            total += part[1] - part[0]
            prev = part
        assert total == total_width, "Total layer width mismatch."

    for design, layers in devicemap.items():
        top_layer = layers[0]
        assert top_layer[0][0] == 0, "Top layer must start at 0."
        total_width = top_layer[0][1]
        for layer in layers:
            check_layer(layer, total_width)

def get_unitsize(dict_size, tiles_edges_in_tier):
    """Calculate the total unit size for a single plane."""
    row_types = np.array(["imc", "tsv0"])
    col_types = np.array(["tsv1", "r"])
    plane_rows = np.tile(row_types, tiles_edges_in_tier)
    plane_cols = np.tile(col_types, tiles_edges_in_tier)
    plane = np.stack((plane_rows, plane_cols))
    oneplane = np.tile(plane, (tiles_edges_in_tier, 1))
    xdim = sum(dict_size[oneplane[0, y]][0] for y in range(oneplane.shape[1]))
    ydim = sum(dict_size[oneplane[x, 0]][1] for x in range(oneplane.shape[0]))
    return xdim, ydim

def basicblock(dict_size, dict_k, xdim, tiles_edges_in_tier):
    """Generate basic building blocks for geometry, conductivity, and naming."""
    tallair = dict_size["imc"][0]
    shortair = dict_size["r"][0]
    dict_k['cu'] = 15

    # Geometry columns
    tallunit = f"{xdim/(2*tiles_edges_in_tier)},{tallair}"
    shortunit = f"{xdim/(2*tiles_edges_in_tier)},{shortair}"
    onenormalcol = np.array([tallunit, shortunit] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)

    planeunit0_0 = f"{tallair},{tallair}"
    planeunit0_1 = f"{tallair},{shortair}"
    planeunit1_0 = f"{shortair},{tallair}"
    planeunit1_1 = f"{shortair},{shortair}"
    onedevicecol0 = np.array([planeunit0_0, planeunit0_1] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevicecol1 = np.array([planeunit1_0, planeunit1_1] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)

    # Conductivity columns
    def make_k_col(imc, tsv, r):
        return np.array([imc, tsv, tsv, r] * tiles_edges_in_tier // 2).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col0_layer0 = np.array([dict_k['k_imc_0'], dict_k['k_tsv_0']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col1_layer0 = np.array([dict_k['k_tsv_0'], dict_k['k_r_0']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col0_layer1 = np.array([dict_k['k_imc_1'], dict_k['k_tsv_1']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col1_layer1 = np.array([dict_k['k_tsv_1'], dict_k['k_r_1']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col0_layer2 = np.array([dict_k['k_imc_2'], dict_k['k_tsv_2']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_k_col1_layer2 = np.array([dict_k['k_tsv_2'], dict_k['k_r_2']] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)

    oneheatsinkcol = np.full((tiles_edges_in_tier*2, 1), dict_k['cu'])
    oneheatspreadcol = np.full((tiles_edges_in_tier*2, 1), dict_k['cu'])
    onesubscol = np.full((tiles_edges_in_tier*2, 1), dict_k['subs'])
    oneaircol = np.full((tiles_edges_in_tier*2, 1), dict_k['air'])

    dict_colk = {
        'onesubs_k_col': onesubscol,
        'oneheatspread_k_col': oneheatspreadcol,
        'oneheatsink_k_col': oneheatsinkcol,
        'oneair_k_col': oneaircol,
        'onedevice_k_col0_layer0': onedevice_k_col0_layer0,
        'onedevice_k_col1_layer0': onedevice_k_col1_layer0,
        'onedevice_k_col0_layer1': onedevice_k_col0_layer1,
        'onedevice_k_col1_layer1': onedevice_k_col1_layer1,
        'onedevice_k_col0_layer2': onedevice_k_col0_layer2,
        'onedevice_k_col1_layer2': onedevice_k_col1_layer2,
    }

    # Naming columns
    onedevice_n_col0_layer0 = np.array(['imc0', 'tsv'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_n_col1_layer0 = np.array(['tsv', 'r'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_n_col0_layer1 = np.array(['imc1', 'tsv'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_n_col1_layer1 = np.array(['tsv', 'r'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_n_col0_layer2 = np.array(['imc2', 'tsv'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    onedevice_n_col1_layer2 = np.array(['tsv', 'r'] * tiles_edges_in_tier).reshape(tiles_edges_in_tier*2, 1)
    oneheatsink_n_col = np.full((tiles_edges_in_tier*2, 1), 'cu')
    oneheatspread_n_col = np.full((tiles_edges_in_tier*2, 1), 'cu')
    onesubs_n_col = np.full((tiles_edges_in_tier*2, 1), 'subs')
    oneair_n_col = np.full((tiles_edges_in_tier*2, 1), 'air')

    dict_coln = {
        'onesubs_n_col': onesubs_n_col,
        'oneheatspread_n_col': oneheatspread_n_col,
        'oneheatsink_n_col': oneheatsink_n_col,
        'oneair_n_col': oneair_n_col,
        'onedevice_n_col0_layer0': onedevice_n_col0_layer0,
        'onedevice_n_col1_layer0': onedevice_n_col1_layer0,
        'onedevice_n_col0_layer1': onedevice_n_col0_layer1,
        'onedevice_n_col1_layer1': onedevice_n_col1_layer1,
        'onedevice_n_col0_layer2': onedevice_n_col0_layer2,
        'onedevice_n_col1_layer2': onedevice_n_col1_layer2,
    }

    return onenormalcol, onedevicecol0, onedevicecol1, dict_colk, dict_coln

def create_cube(dict_size, dict_z, dict_k, xdim, devicemap, heatsinkair_resoluation, tiles_edges_in_tier):
    onebasiccol, onedevicecol0, onedevicecol1, dict_colk, dict_coln = basicblock(dict_size, dict_k, xdim, tiles_edges_in_tier)
    cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict, cube_layertype_dict = {}, {}, {}, {}, {}

    for designname, layerlist in devicemap.items():
        layer_geo_l, layer_k_l, layer_z_l, layer_n_l, layer_type_l = [], [], [], [], []
        totalx = layerlist[0][0][1]

        # Precompute layer types and sublayer counts
        layer_types = []
        sublayer_counts = []
        for idx, _ in enumerate(layerlist):
            if idx == 0:
                layer_types.append('heatsink')
                sublayer_counts.append(5)
            elif idx == 1:
                layer_types.append('heatspread')
                sublayer_counts.append(5)
            elif idx == len(layerlist) - 2:
                layer_types.append('subs')
                sublayer_counts.append(5)
            elif idx == len(layerlist) - 1:
                layer_types.append('air')
                sublayer_counts.append(5)
            else:
                layer_types.append('device')
                sublayer_counts.append(3)

        for layer, layertype, numofsublayer in zip(layerlist, layer_types, sublayer_counts):
            for idx1 in range(numofsublayer):
                layer_type_l.append(layertype)
                layer_z_l.append(heatsinkair_resoluation / 1000 if layertype != 'device' else dict_z[layertype][idx1] / 1000)

                xpos = 0
                devicetrigger = False
                currlayer_dim, currlayer_k, currlayer_n = None, None, None

                while xpos <= totalx * tiles_edges_in_tier * 2 - 1:
                    parttype = findparttype(xpos, layer, tiles_edges_in_tier)
                    layer2dictk = f'one{parttype}_k_col'
                    layer2dictn = f'one{parttype}_n_col'

                    if iffallintodevicerange(xpos, layerlist[2], tiles_edges_in_tier):
                        # Device region
                        if not devicetrigger:
                            if parttype == 'device':
                                layer2dictk = f'one{parttype}_k_col0_layer{idx1}'
                                layer2dictn = f'one{parttype}_n_col0_layer{idx1}'
                            devicetrigger = True
                            block_dim, block_k, block_n = onedevicecol0, dict_colk[layer2dictk], dict_coln[layer2dictn]
                        else:
                            if parttype == 'device':
                                layer2dictk = f'one{parttype}_k_col1_layer{idx1}'
                                layer2dictn = f'one{parttype}_n_col1_layer{idx1}'
                            devicetrigger = False
                            block_dim, block_k, block_n = onedevicecol1, dict_colk[layer2dictk], dict_coln[layer2dictn]
                    else:
                        # Non-device region
                        block_dim, block_k, block_n = onebasiccol, dict_colk[layer2dictk], dict_coln[layer2dictn]

                    # Stack blocks horizontally
                    if currlayer_dim is None:
                        currlayer_dim, currlayer_k, currlayer_n = block_dim, block_k, block_n
                    else:
                        currlayer_dim = np.hstack([currlayer_dim, block_dim])
                        currlayer_k = np.hstack([currlayer_k, block_k])
                        currlayer_n = np.hstack([currlayer_n, block_n])
                    xpos += 1
                layer_geo_l.append(currlayer_dim)
                layer_k_l.append(currlayer_k)
                layer_n_l.append(currlayer_n)

        layer_geo = np.stack(layer_geo_l)
        layer_k = np.stack(layer_k_l)
        layer_n = np.stack(layer_n_l)
        assert layer_geo.shape == layer_k.shape == layer_n.shape
        cube_geo_dict[designname] = layer_geo
        cube_k_dict[designname] = layer_k
        cube_z_dict[designname] = layer_z_l
        cube_n_dict[designname] = layer_n
        cube_layertype_dict[designname] = layer_type_l

    return cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict, cube_layertype_dict

def snakewalk(numofnodex, numofnodey, start_pos, direction0, direction1):
    x, y = start_pos
    order_idx = []
    visited = set()
    directions = {
        'up':    (-1, 0),
        'down':  (1, 0),
        'left':  (0, -2),
        'right': (0, 2)
    }
    dir0, dir1 = direction0, direction1

    while True:
        order_idx.append((x, y))
        visited.add((x, y))
        # Move in primary direction
        dx, dy = directions[dir0]
        x_new, y_new = x + dx, y + dy

        # Check bounds and switch direction if needed
        if dir0 in ['up', 'down']:
            if not (0 <= x_new < numofnodex):
                dir0 = 'down' if dir0 == 'up' else 'up'
                x_new = max(0, min(numofnodex - 1, x_new))
                # Move in secondary direction
                if dir1 == 'right':
                    y += 2
                    if y >= start_pos[1] + numofnodey:
                        y -= 2
                        break
                elif dir1 == 'left':
                    y -= 2
                    if y < start_pos[1]:
                        y += 2
                        break
                else:
                    break
            x = x_new
        elif dir0 in ['left', 'right']:
            if not (start_pos[1] <= y_new < start_pos[1] + numofnodey):
                dir0 = 'right' if dir0 == 'left' else 'left'
                y_new = max(start_pos[1], min(start_pos[1] + numofnodey - 1, y_new))
                # Move in secondary direction
                if dir1 == 'up':
                    x -= 1
                    if x < 0:
                        x += 1
                        break
                elif dir1 == 'down':
                    x += 1
                    if x >= numofnodex:
                        x -= 1
                        break
                else:
                    break
            y = y_new
        else:
            break

    return order_idx, (x, y)

def load_power(dict_z, devicemap, cube_n_dict, power_tsv, power_router, numofdevicelayer_dict, cube_layertype_dict, tiles_edges_in_tier, chiplet_num, placement_method):
    power_inform = pd.read_csv("./Results/layer_performance.csv", header=None).to_numpy()
    computing_data = pd.read_csv("./Results/layer_inform.csv", header=None).to_numpy()
    power_l = []

    if placement_method != 2:
        for i in range(len(computing_data) - 1):
            power_l.extend([float(power_inform[i][5])] * int(computing_data[i][1]))
            missing = int((computing_data[i][9] + 1) * tiles_edges_in_tier * tiles_edges_in_tier - computing_data[i][7])
            if missing != 0 and missing < computing_data[i + 1][1]:
                power_l.extend([0] * missing)
        power_l.extend([float(power_inform[-1][5])] * int(computing_data[-1][1]))
        missing = int((computing_data[-1][9] + 1) * tiles_edges_in_tier * tiles_edges_in_tier - computing_data[-1][7])
        if missing != 0:
            power_l.extend([0] * missing)
    else:
        every_tier_real_tiles = [
            sum(int(computing_data[i][1]) for i in range(len(computing_data)) if computing_data[i][9] == tier_n)
            for tier_n in range(chiplet_num)
        ]
        for tier_n in range(chiplet_num):
            sum_tiles = 0
            for i in range(len(computing_data)):
                if computing_data[i][9] == tier_n:
                    power_l.extend([float(power_inform[i][5])] * int(computing_data[i][1]))
                    sum_tiles += int(computing_data[i][1])
            if every_tier_real_tiles[tier_n] == sum_tiles:
                power_l.extend([0] * ((tiles_edges_in_tier * tiles_edges_in_tier) - sum_tiles))
    power_l = np.array(power_l)
    assert len(power_l) == chiplet_num * tiles_edges_in_tier * tiles_edges_in_tier

    dict_power_container = {}
    for design, layer_n in cube_n_dict.items():
        layertype_l = cube_layertype_dict[design]
        curr_chip_zstart_copy = next(idx for idx, layertype in enumerate(layertype_l) if layertype == 'device')
        devicelayer = devicemap[design][2]
        numofdevicelayer = numofdevicelayer_dict[design]
        power_container = np.zeros(layer_n.shape)
        power_count = 0
        tsv_idx = 0
        router_idx = 0
        for partidx, part in enumerate(devicelayer):
            if part[2] != 'device':
                continue
            curr_chip_xystart = (0, part[0] * tiles_edges_in_tier * 2)
            curr_chip_zstart = curr_chip_zstart_copy
            order_idx, _ = snakewalk(tiles_edges_in_tier * 2, tiles_edges_in_tier * 2, curr_chip_xystart, 'down', 'right')
            for _ in range(numofdevicelayer):
                for x, y in order_idx:
                    for sublayer in range(3):
                        actual_layer = curr_chip_zstart + sublayer
                        tiletype = layer_n[actual_layer, x, y]
                        tiletype_next = layer_n[actual_layer, x, y + 1]
                        assert tiletype not in ('subs', 'air', 'cu')
                        assert tiletype_next not in ('subs', 'air', 'cu')

                        tilepower = 0
                        if tiletype == 'tsv':
                            tilepower = power_tsv[tsv_idx]
                        elif tiletype == 'imc0':
                            tilepower = power_l[power_count]
                            power_count += 1
                        elif tiletype == 'r' and sublayer == 0:
                            tilepower = power_router[router_idx]

                        tilepower_next = 0
                        if tiletype_next == 'tsv':
                            tilepower_next = power_tsv[tsv_idx]
                        elif tiletype_next == 'imc0':
                            tilepower_next = power_l[power_count]
                            power_count += 1
                        elif tiletype_next == 'r' and sublayer == 0:
                            tilepower_next = power_router[router_idx]
                        power_container[actual_layer, x, y] = tilepower / 1000
                        power_container[actual_layer, x, y + 1] = tilepower_next / 1000
                tsv_idx += 1
                router_idx += 1
                curr_chip_zstart += 3
        assert tsv_idx == chiplet_num
        assert router_idx == chiplet_num
        assert power_count == len(power_l)
        dict_power_container[design] = power_container

    return dict_power_container

def get_conductance_G(cube_geo_dict, cube_k_dict, cube_z_dict):
    direction_map = {
        0: (0, 1, 0),   # east
        1: (0, -1, 0),  # west
        2: (1, 0, 0),   # north
        3: (-1, 0, 0),  # south
        4: (0, 0, 1),   # top
        5: (0, 0, -1)   # bottom
    }

    cube_G_dict = {}
    for design, geo_info_np in cube_geo_dict.items():
        k_info_np = cube_k_dict[design]
        z_info_np = cube_z_dict[design]
        assert geo_info_np.shape == k_info_np.shape
        num_layers, num_x, num_y = geo_info_np.shape[0], geo_info_np.shape[1], geo_info_np.shape[2]
        num_total_nodes = (num_layers + 2) * num_x * num_y

        G_sparse = sparse_mat.dok_matrix((num_total_nodes, num_total_nodes))

        for z_idx in range(1, num_layers + 1):
            for x_idx in range(num_x):
                for y_idx in range(num_y):
                    dim_str = geo_info_np[z_idx - 1, x_idx, y_idx].split(',')
                    center_length, center_width = float(dim_str[0]), float(dim_str[1])
                    center_height = z_info_np[z_idx - 1]
                    center_k = k_info_np[z_idx - 1, x_idx, y_idx].item()
                    center_id = id_(x_idx, y_idx, z_idx, num_x, num_y, num_layers)

                    # Boundary connections
                    if z_idx == num_layers:
                        neighbor_id = id_(x_idx, y_idx, z_idx + 1, num_x, num_y, num_layers)
                        G_sparse[center_id, neighbor_id] = 1
                        G_sparse[neighbor_id, center_id] = 1
                    if z_idx == 1:
                        neighbor_id = id_(x_idx, y_idx, z_idx - 1, num_x, num_y, num_layers)
                        G_sparse[center_id, neighbor_id] = 1
                        G_sparse[neighbor_id, center_id] = 1

                    G_sparse[center_id, center_id] = 0

                    # Neighbor conductance
                    for direction in range(6):
                        nx, ny, nz = x_idx + direction_map[direction][0], y_idx + direction_map[direction][1], z_idx + direction_map[direction][2]
                        if checkneighbours(nx, ny, nz - 1, num_x, num_y, num_layers):
                            neighbor_id = id_(nx, ny, nz, num_x, num_y, num_layers)
                            neighbor_k = k_info_np[nz - 1, nx, ny]
                            dim_str_n = geo_info_np[nz - 1, nx, ny].split(',')
                            neighbor_length, neighbor_width = float(dim_str_n[0]), float(dim_str_n[1])

                            if direction in [0, 1]:  # east/west
                                assert center_width == neighbor_width
                                d1, d2 = center_length / 2, neighbor_length / 2
                                area = center_width * center_height
                            elif direction in [2, 3]:  # north/south
                                assert center_length == neighbor_length
                                d1, d2 = center_width / 2, neighbor_width / 2
                                area = center_length * center_height
                            else:  # top/bottom
                                assert center_width == neighbor_width
                                assert center_length == neighbor_length
                                neighbor_height = z_info_np[nz - 1]
                                d1, d2 = center_height / 2, neighbor_height / 2
                                area = center_width * center_length

                            dd = d1 + d2
                            k_avg = dd / (d1 / center_k + d2 / neighbor_k)
                            G_val = (k_avg * area) / dd
                            G_sparse[center_id, center_id] += G_val
                            G_sparse[center_id, neighbor_id] = -G_val
        cube_G_dict[design] = G_sparse
        print('INFO: Done generating G for', design)
    return cube_G_dict

def convert2realratio(t, namemap, xdim):
    assert t.shape == namemap.shape
    layers = []
    for layer in range(t.shape[0]):
        rows = []
        for row in range(t.shape[1]):
            blocks = []
            for col in range(t.shape[2]):
                value = t[layer, row, col]
                celltype = namemap[layer, row, col]
                if row % 2 == 0:
                    num_w = 2
                    num_l = 2 if celltype in ('imc0', 'imc1', 'imc2') else 1
                else:
                    num_w = 1
                    num_l = 2 if celltype == 'tsv' else 1
                blocks.append(np.full((num_w, num_l), value))
            row_matrix = np.hstack(blocks)
            rows.append(row_matrix)
        layer_matrix = np.vstack(rows)
        layers.append(layer_matrix[np.newaxis, ...])
    return np.vstack(layers)

def solver(cube_G_dict, cube_n_dict, cube_power_dict, cube_layertype_dict, xdim, sim_name):
    device_count = 0
    for design, G_sparse in cube_G_dict.items():
        namemap = cube_n_dict[design]
        p = cube_power_dict[design]
        numoflayer = p.shape[0]
        numofnodex = p.shape[1]
        numofnodey = p.shape[2]
        p = p.reshape(numoflayer*numofnodex*numofnodey, 1)
        newp = np.ones(((numoflayer+2)*numofnodex*numofnodey, 1))*298
        #====================================================================================================
        newp[numofnodex*numofnodey:(numoflayer+1)*numofnodex*numofnodey, :] = p
        p = newp
        G_sparse = G_sparse.tocsc()
        p = sparse_mat.csc_matrix(p)

        print('Starting solving...')
        start_time = time.time()       
        t = sparse_algebra.spsolve(G_sparse, p, permc_spec=None, use_umfpack=True)
        print("Running time: %s s" % (time.time() - start_time))
        t = t.reshape(numoflayer+2, numofnodex, numofnodey)
        t = t[1:numoflayer+1, :, :]
        #====================================================================================================
        layertype_l = cube_layertype_dict[design]
        devicestart = None
        deviceend = None
        start = False
        for i,layername in enumerate(layertype_l):
            if devicestart is None and layername == 'device':
                devicestart = i
                start = True
                continue
            if deviceend is None and  start and layername != 'device':
                deviceend = i
                break
        t = t[devicestart:deviceend, :, :]
        namemap = namemap[devicestart:deviceend, :, :]
        realratiot = convert2realratio(t, namemap, xdim) # TODO: needed?

        vmin  = t.min()
        vmax  = t.max()
        print(design, '\t\tMin T:', round(vmin, 2), '\t\tPeak T:', round(vmax,2), '\t\tAverage T:', round(t.mean(), 2))
        # Draw the thermal map
        plot_im(plot_data=realratiot[0,:,:], save_name='./Results/result_thermal/{}/thermal_map{}.png'.format(design, device_count), vmin=vmin, vmax=vmax)
        plot_im(plot_data=t[0,:,:], save_name='./Results/result_thermal/{}/thermal_map_raw{}.png'.format(design, device_count), vmin=vmin, vmax=vmax)
        # Draw the power map
        power_map = cube_power_dict[design][devicestart, :, :]
        power_vmin = power_map.min()
        power_vmax = power_map.max()
        plot_im(plot_data=power_map, save_name='./Results/result_thermal/{}/power_map{}.png'.format(design, device_count), vmin=power_vmin, vmax=power_vmax)
        device_count += 1

    return round(vmax, 2)
