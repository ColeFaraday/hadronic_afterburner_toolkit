#!/usr/bin/env python3
import sys
import json
import h5py
import numpy as np
from scipy.optimize import curve_fit
import os

hbarc = 0.19733
KT_values = ['0.15_0.25', '0.25_0.35', '0.35_0.45', '0.45_0.55']
q_cut_max_list = [0.05, 0.075, 0.1, 0.125, 0.15]
q_cut_min = 0.
eps = 1e-15

HEADER = (
    "# q_cut[GeV]  lambda  lambda_err  R_out[fm]  R_out_err[fm]  "
    "R_side[fm]  R_side_err[fm]  R_long [fm]  R_long_err[fm]  "
    "R_os[fm]  R_os_err[fm]  R_ol[fm]  R_ol_err[fm]"
)

def gaussian_3d(q_arr, lambda_, R_out, R_side, R_long, R_os, R_ol):
    q_out, q_side, q_long = q_arr
    R_out /= hbarc
    R_side /= hbarc
    R_long /= hbarc
    R_os /= hbarc
    R_ol /= hbarc
    gauss = lambda_ * np.exp(
        -((R_out * q_out) ** 2 + (R_side * q_side) ** 2 + (R_long * q_long) ** 2
          + 2. * q_out * q_side * R_os ** 2. + 2. * q_out * q_long * R_ol ** 2.)
    )
    return np.ravel(gauss)

def load_centrality_mapping(json_path, mapping_key=None):
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    if mapping_key is None:
        mapping_key = list(mapping['mappings'].keys())[0]
    return mapping['mappings'][mapping_key], mapping_key

def load_all_centrality_mappings(json_path):
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    return mapping['mappings']

def main():
    if len(sys.argv) < 4:
        print("Usage: {} input_events.h5 centrality_map.json output.h5 [mapping_key]".format(sys.argv[0]))
        sys.exit(1)
    input_h5 = sys.argv[1]
    centrality_json = sys.argv[2]
    output_h5 = sys.argv[3]
    mapping_key = sys.argv[4] if len(sys.argv) > 4 else None

    # Check if output file exists
    if os.path.exists(output_h5):
        resp = input(f"Output file '{output_h5}' exists. Overwrite? (y/n): ").strip().lower()
        if resp == 'y':
            os.remove(output_h5)
            print(f"Deleted '{output_h5}'. Proceeding...")
        else:
            print("Aborted.")
            sys.exit(0)

    with h5py.File(input_h5, 'r') as hf, h5py.File(output_h5, 'w') as out_h5:
        if mapping_key is not None:
            centrality_map, used_mapping_key = load_centrality_mapping(centrality_json, mapping_key)
            print(f"Using centrality mapping: {used_mapping_key}")
            mapping_group = out_h5.create_group(used_mapping_key)
            centrality_maps = {used_mapping_key: centrality_map}
        else:
            centrality_maps = load_all_centrality_mappings(centrality_json)
            print(f"Using all centrality mappings: {list(centrality_maps.keys())}")
        for used_mapping_key, centrality_map in centrality_maps.items():
            mapping_group = out_h5.create_group(used_mapping_key)
            for cent_class, event_dict in centrality_map.items():
                print(f"Processing centrality class: {cent_class} (mapping: {used_mapping_key})")
                class_group = mapping_group.create_group(cent_class)
                event_names = list(event_dict.keys())
                for iKT, KT in enumerate(KT_values):
                    file_name = f'HBT_correlation_function_KT_{KT}.dat'
                    # Collect all event data for this centrality class and KT bin
                    event_data_list = []
                    for event_name in event_names:
                        if event_name in hf:
                            event_group = hf[event_name]
                            if file_name in event_group:
                                data = np.nan_to_num(event_group[file_name][...])
                                event_data_list.append(data)
                    if not event_data_list:
                        print(f"No events found for {cent_class} {KT}")
                        continue
                    # Average over events
                    event_data_stack = np.stack(event_data_list)
                    event_avg_data = np.mean(event_data_stack, axis=0)
                    nev = len(event_data_list)
                    # Calculate num, denorm, etc. for fit
                    num = event_avg_data[:, 4]
                    denorm = event_avg_data[:, 5]
                    correlation = event_avg_data[:, 6] if event_avg_data.shape[1] > 6 else num / denorm
                    correlation_err = event_avg_data[:, 7] if event_avg_data.shape[1] > 7 else np.zeros_like(correlation)
                    # Save averaged histogram
                    class_group.create_dataset(file_name, data=event_avg_data, compression="gzip", compression_opts=9)
                    # Fit radii
                    nq = int(round(event_avg_data.shape[0] ** (1 / 3)))
                    try:
                        q_out = event_avg_data[:, 0].reshape(nq, nq, nq)
                        q_side = event_avg_data[:, 1].reshape(nq, nq, nq)
                        q_long = event_avg_data[:, 2].reshape(nq, nq, nq)
                        HBT_Corr = correlation.reshape(nq, nq, nq)
                        HBT_Corr_err = correlation_err.reshape(nq, nq, nq) + eps
                    except Exception as e:
                        print(f"Reshape error for {cent_class} {KT}: {e}")
                        continue
                    output = []
                    for q_cut in q_cut_max_list:
                        idx = ((np.sqrt(q_out ** 2 + q_side ** 2 + q_long ** 2) > q_cut_min) &
                               (np.sqrt(q_out ** 2 + q_side ** 2 + q_long ** 2) < q_cut))
                        q_arr = [q_out[idx], q_side[idx], q_long[idx]]
                        guess_vals = [1.0, 5., 5., 5., 0.1, 0.1]
                        try:
                            fit_params, cov_mat = curve_fit(
                                gaussian_3d, q_arr, np.ravel(HBT_Corr[idx]), p0=guess_vals,
                                sigma=np.ravel(HBT_Corr_err[idx]), absolute_sigma=True)
                            fit_errors = np.sqrt(np.diag(cov_mat))
                            temp = []
                            for x, y in zip(fit_params, fit_errors):
                                temp += [x, y]
                            output.append([q_cut] + temp)
                        except Exception as e:
                            print(f"Fit error for {cent_class} {KT} q_cut={q_cut}: {e}")
                            output.append([q_cut] + [np.nan] * 12)
                    radii_name = f'HBT_radii_KT_{KT}.dat'
                    radii_ds = class_group.create_dataset(radii_name, data=np.array(output), compression="gzip", compression_opts=9)
                    radii_ds.attrs.create("header", HEADER.encode('utf-8'))
    print("Done.")

if __name__ == "__main__":
    main()
