#!/usr/bin/env python3
"""
=============================================================================
Symmetry Function Feature Selection for RuNNer
=============================================================================

Automated backward elimination of symmetry functions (ACSF) for each element
independently, then simultaneously for all elements (Phase 2).

Algorithm (per element):
    i = 0
    model[0] = build_mlp(train_data)           # full feature set
    build_evaluate_RuNNer(train_data)           # mode1 -> mode2 -> mode3
    do:
        i++
        rank(element)                           # feature importance
        train_data = train_data - least_important_feature * n
        model[i] = build_mlp(train_data)
        build_evaluate_RuNNer(train_data)
        # save r2_model[i]
    while (r2(model[i]) > r2(model[0]) * 0.95)

    Repeat for each element (La, Zr, O) independently.
    Phase 2: Simultaneously reduce features from all elements.

Usage:
    python symf_feature_selection.py --work_dir . --element La --n_remove 1
    python symf_feature_selection.py --work_dir . --phase2
    python symf_feature_selection.py --work_dir . --element all --n_remove 1

=============================================================================
"""

import os
import sys
import re
import shutil
import subprocess
import argparse
import json
import time
import glob
import numpy as np
from collections import OrderedDict

# ============================================================================
# 1) PARSING: Read symmetry functions from input.nn template
# ============================================================================

def parse_symfunction_lines(filepath):
    """
    Parse an input.nn file and separate:
      - header_lines: all non-symfunction lines (settings, comments)
      - symf_lines: dict keyed by central element -> list of symfunction line strings
    Returns (header_lines, symf_dict, all_symf_lines_ordered)
    """
    header_lines = []
    symf_dict = {}  # element -> list of (line_string, line_index)
    all_lines = []

    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    symf_lines_ordered = []
    for i, line in enumerate(all_lines):
        stripped = line.strip()
        if stripped.startswith('symfunction_short'):
            parts = stripped.split()
            central_elem = parts[1]
            if central_elem not in symf_dict:
                symf_dict[central_elem] = []
            symf_dict[central_elem].append(stripped)
            symf_lines_ordered.append((central_elem, stripped))
        else:
            header_lines.append(line)

    return header_lines, symf_dict, symf_lines_ordered


def write_input_nn(filepath, header_lines, symf_dict, runner_mode, elements_order=None):
    """
    Write an input.nn file with specified runner_mode and symmetry functions.
    header_lines: the non-symfunction portion of the template
    symf_dict: element -> list of symfunction line strings
    runner_mode: 1, 2, or 3
    """
    if elements_order is None:
        elements_order = sorted(symf_dict.keys())

    with open(filepath, 'w') as f:
        for line in header_lines:
            # Replace runner_mode
            if line.strip().startswith('runner_mode'):
                # Preserve comment
                parts = line.split('#')
                comment = '#' + '#'.join(parts[1:]) if len(parts) > 1 else ''
                f.write(f'runner_mode {runner_mode}                                    {comment}\n')
            else:
                f.write(line)

        # Write symmetry functions grouped by element
        f.write('\n# SYMMETRY FUNCTIONS (auto-generated)\n')
        for elem in elements_order:
            if elem in symf_dict:
                f.write(f'\n## Features for {elem}\n')
                for sf_line in symf_dict[elem]:
                    f.write(sf_line + '\n')


# ============================================================================
# 2) RuNNer EXECUTION: Mode 1 -> Mode 2 -> Mode 3
# ============================================================================

def run_runner(work_dir, runner_exe, mode, timeout=None):
    """
    Run RuNNer in the specified mode.
    Returns (returncode, stdout, stderr)
    """
    print(f"\n{'='*60}")
    print(f"  Running RuNNer Mode {mode} in {work_dir}")
    print(f"{'='*60}")

    # Build the input.nn for this mode
    cmd = f"cd {work_dir} && {runner_exe}"
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        print(f"  Mode {mode} completed in {elapsed:.1f}s (exit code: {result.returncode})")
        
        # Save output
        output_file = os.path.join(work_dir, f"mode{mode}.out")
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        
        if result.returncode != 0:
            print(f"  WARNING: RuNNer mode {mode} returned non-zero exit code!")
            print(f"  stderr: {result.stderr[:500]}")
        
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  ERROR: RuNNer mode {mode} timed out!")
        return -1, "", "Timeout"


def run_full_pipeline(work_dir, runner_exe, header_lines, symf_dict,
                      elements_order, n_epochs=100, timeout_per_mode=None):
    """
    Execute the full RuNNer pipeline: Mode 1 -> Mode 2 -> Mode 3.
    
    1. Generate input.nn for mode 1, run it
    2. Generate input.nn for mode 2, run it
    3. Copy optimal weights, generate input.nn for mode 3, run it
    4. Extract metrics
    
    Returns a dict with metrics.
    """
    # --- Mode 1: Compute symmetry functions and split data ---
    write_input_nn(os.path.join(work_dir, 'input.nn'),
                   header_lines, symf_dict, runner_mode=1,
                   elements_order=elements_order)
    
    rc1, out1, err1 = run_runner(work_dir, runner_exe, mode=1, timeout=timeout_per_mode)
    if rc1 != 0:
        print("ERROR: Mode 1 failed!")
        return None

    # --- Mode 2: Training ---
    write_input_nn(os.path.join(work_dir, 'input.nn'),
                   header_lines, symf_dict, runner_mode=2,
                   elements_order=elements_order)
    
    rc2, out2, err2 = run_runner(work_dir, runner_exe, mode=2, timeout=timeout_per_mode)
    if rc2 != 0:
        print("ERROR: Mode 2 failed!")
        return None

    # --- Copy optimal weights for Mode 3 ---
    copy_optimal_weights(work_dir, elements_order)

    # --- Mode 3: Prediction/Evaluation ---
    write_input_nn(os.path.join(work_dir, 'input.nn'),
                   header_lines, symf_dict, runner_mode=3,
                   elements_order=elements_order)
    
    rc3, out3, err3 = run_runner(work_dir, runner_exe, mode=3, timeout=timeout_per_mode)
    if rc3 != 0:
        print("ERROR: Mode 3 failed!")
        return None

    # --- Extract metrics ---
    metrics = extract_metrics(work_dir, out2)
    return metrics


def copy_optimal_weights(work_dir, elements_order):
    """
    Copy optweights.XXX.out -> weights.XXX.data for each element.
    Element nuclear charges: La=57, Zr=40, O=8
    """
    elem_to_z = {'La': '057', 'Zr': '040', 'O': '008',
                 'H': '001', 'C': '006', 'N': '007', 'Mn': '025',
                 'Li': '003', 'Mg': '012', 'Au': '079'}

    for elem in elements_order:
        z_str = elem_to_z.get(elem)
        if z_str is None:
            print(f"WARNING: Unknown element {elem}, cannot copy weights")
            continue
        
        src = os.path.join(work_dir, f"optweights.{z_str}.out")
        dst = os.path.join(work_dir, f"weights.{z_str}.data")
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
        else:
            print(f"  WARNING: {src} not found!")


# ============================================================================
# 3) METRICS EXTRACTION
# ============================================================================

def extract_metrics(work_dir, mode2_stdout=None):
    """
    Extract evaluation metrics from RuNNer output:
    - RMSE (train/test) from mode2.out
    - R² from trainpoints/testpoints files (last epoch)
    
    Returns dict with keys:
        rmse_train, rmse_test, r2_train, r2_test, best_epoch,
        max_error_train, max_error_test
    """
    metrics = {}

    # --- Parse mode2.out for RMSE per epoch ---
    mode2_file = os.path.join(work_dir, 'mode2.out')
    if mode2_stdout is None and os.path.exists(mode2_file):
        with open(mode2_file, 'r') as f:
            mode2_stdout = f.read()
    
    if mode2_stdout:
        # Parse ENERGY lines: " ENERGY    60      0.001831     0.002081   10.57"
        energy_lines = re.findall(
            r'^\s*ENERGY\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+',
            mode2_stdout, re.MULTILINE
        )
        
        if energy_lines:
            epochs = []
            for epoch_str, train_rmse_str, test_rmse_str in energy_lines:
                epochs.append({
                    'epoch': int(epoch_str),
                    'train_rmse': float(train_rmse_str),
                    'test_rmse': float(test_rmse_str)
                })
            
            # Best epoch (lowest test RMSE)
            best = min(epochs, key=lambda x: x['test_rmse'])
            metrics['best_epoch'] = best['epoch']
            metrics['rmse_train'] = best['train_rmse']
            metrics['rmse_test'] = best['test_rmse']
            
            # Also store all epoch data
            metrics['epoch_data'] = epochs

        # Parse OPTSHORT line for final best RMSE
        optshort = re.search(
            r'OPTSHORT\s+([\d.]+)\s+([\d.]+)', mode2_stdout
        )
        if optshort:
            metrics['rmse_train_opt'] = float(optshort.group(1))
            metrics['rmse_test_opt'] = float(optshort.group(2))

    # --- Compute R² from trainpoints/testpoints of last epoch ---
    best_epoch = metrics.get('best_epoch', None)
    
    if best_epoch is not None:
        r2_train = compute_r2_from_points(work_dir, 'trainpoints', best_epoch)
        r2_test = compute_r2_from_points(work_dir, 'testpoints', best_epoch)
        
        if r2_train is not None:
            metrics['r2_train'] = r2_train
        if r2_test is not None:
            metrics['r2_test'] = r2_test
    
    # Also try the last available epoch if best_epoch doesn't have files
    if 'r2_test' not in metrics:
        # Find the last testpoints file
        test_files = sorted(glob.glob(os.path.join(work_dir, 'testpoints.*.out')))
        if test_files:
            last_epoch_str = os.path.basename(test_files[-1]).split('.')[1]
            last_epoch = int(last_epoch_str)
            r2_train = compute_r2_from_points(work_dir, 'trainpoints', last_epoch)
            r2_test = compute_r2_from_points(work_dir, 'testpoints', last_epoch)
            if r2_train is not None:
                metrics['r2_train'] = r2_train
            if r2_test is not None:
                metrics['r2_test'] = r2_test
            metrics['r2_epoch'] = last_epoch

    return metrics


def compute_r2_from_points(work_dir, prefix, epoch):
    """
    Compute R² from a trainpoints/testpoints file using energy PER ATOM.
    File format (whitespace-delimited):
        Conf.  atoms  Ref.total_energy(Ha)  NN_total_energy(Ha)  E(Ref.)-E(NN)(Ha/atom)
    
    R² = 1 - SS_res / SS_tot  (computed on per-atom energies)
    
    IMPORTANT: Must use per-atom energies, NOT total energies.
    Total energy R² is always ~0.9999 regardless of fit quality because
    inter-structure energy differences dominate. Per-atom R² is sensitive
    to the actual neural network fit quality.
    """
    filepath = os.path.join(work_dir, f'{prefix}.{epoch:06d}.out')
    
    if not os.path.exists(filepath):
        return None
    
    try:
        ref_per_atom = []
        nn_per_atom = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Conf') or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        n_atoms = int(parts[1])
                        ref_e   = float(parts[2]) / n_atoms  # per-atom energy
                        nn_e    = float(parts[3]) / n_atoms  # per-atom energy
                        ref_per_atom.append(ref_e)
                        nn_per_atom.append(nn_e)
                    except (ValueError, IndexError, ZeroDivisionError):
                        continue
        
        if len(ref_per_atom) < 2:
            return None
        
        ref = np.array(ref_per_atom)
        nn  = np.array(nn_per_atom)
        
        ss_res = np.sum((ref - nn) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2 = 1.0 - ss_res / ss_tot
        return r2
        
    except Exception as e:
        print(f"  Warning: Could not compute R² from {filepath}: {e}")
        return None


# ============================================================================
# 4) FEATURE IMPORTANCE RANKING
# ============================================================================

def rank_features_by_variance(work_dir, element):
    """
    Rank symmetry function features by their variance in the function.data file.
    Features with low variance carry little information and are candidates for removal.
    
    This is a simple, fast heuristic. More advanced methods (permutation importance,
    weight-based) can be substituted.
    
    Returns: list of feature indices sorted by importance (least important first)
    """
    # function.data contains the computed symmetry function values
    func_file = os.path.join(work_dir, 'function.data')
    
    if not os.path.exists(func_file):
        print(f"  WARNING: function.data not found, using random ranking")
        return None
    
    # Parse function.data format (from RuNNer source writeatomicsymfunctions.f90):
    #   Line 1: num_atoms (integer, 1 field)
    #   Atom lines: Z sf1 sf2 ... sfN (1 + N_features fields, N_features >> 4)
    #   Energy line: charge totalE/atom shortE/atom elecE/atom (4 fields)
    # No begin/end markers.
    elem_to_z = {'La': 57, 'Zr': 40, 'O': 8}
    target_z = elem_to_z.get(element)
    
    if target_z is None:
        print(f"  WARNING: Unknown element {element}")
        return None
    
    feature_values = []
    
    try:
        with open(func_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                # Atom lines have Z + many SF values (>>4 fields).
                # Skip num_atoms lines (1 field) and energy lines (4 fields).
                if len(parts) < 10:
                    continue
                try:
                    z_val = int(float(parts[0]))
                    if z_val == target_z:
                        features = [float(x) for x in parts[1:]]
                        feature_values.append(features)
                except (ValueError, IndexError):
                    continue
        
        if not feature_values:
            print(f"  WARNING: No feature values found for {element}")
            return None
        
        features_array = np.array(feature_values)
        variances = np.var(features_array, axis=0)
        
        # Sort by variance: least important (lowest variance) first
        ranked_indices = np.argsort(variances).tolist()
        
        print(f"  Feature ranking for {element}: {len(ranked_indices)} features")
        print(f"  Lowest variance features (indices): {ranked_indices[:5]}")
        print(f"  Highest variance features (indices): {ranked_indices[-5:]}")
        
        return ranked_indices
        
    except Exception as e:
        print(f"  WARNING: Error ranking features: {e}")
        return None


def rank_features_by_sensitivity(work_dir, element, header_lines, symf_dict,
                                 elements_order, runner_exe, baseline_r2,
                                 timeout_per_mode=None):
    """
    Rank features by leave-one-out sensitivity analysis.
    Remove each feature one at a time, re-run RuNNer, measure R² drop.
    Features causing the largest R² drop are the most important.
    
    This is expensive but more accurate than variance-based ranking.
    
    Returns: list of feature indices sorted by importance (least important first)
    """
    n_features = len(symf_dict[element])
    r2_drops = []
    
    print(f"\n  Sensitivity analysis for {element} ({n_features} features)...")
    
    for i in range(n_features):
        # Create a copy of symf_dict with feature i removed
        temp_symf = dict(symf_dict)
        temp_features = list(temp_symf[element])
        removed = temp_features.pop(i)
        temp_symf[element] = temp_features
        
        # Create a temporary working directory
        temp_dir = os.path.join(work_dir, f'_sensitivity_{element}_{i}')
        setup_work_dir(work_dir, temp_dir)
        
        # Run pipeline
        metrics = run_full_pipeline(temp_dir, runner_exe, header_lines,
                                    temp_symf, elements_order,
                                    timeout_per_mode=timeout_per_mode)
        
        if metrics and 'r2_test' in metrics:
            r2_drop = baseline_r2 - metrics['r2_test']
            r2_drops.append((i, r2_drop, metrics['r2_test']))
            print(f"    Feature {i}: R²_test={metrics['r2_test']:.6f}, "
                  f"drop={r2_drop:.6f} ({removed[:50]}...)")
        else:
            # If failed, assume this feature is important
            r2_drops.append((i, 999.0, None))
            print(f"    Feature {i}: FAILED (assuming important)")
        
        # Clean up temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Sort by R² drop: least important (smallest drop) first
    r2_drops.sort(key=lambda x: x[1])
    ranked_indices = [idx for idx, _, _ in r2_drops]
    
    return ranked_indices


def setup_work_dir(source_dir, target_dir, input_data_path=None):
    """
    Set up a working directory for RuNNer by copying/linking necessary files.
    input_data_path: optional path to input.data (default: source_dir/input.data)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Link input.data (use custom path if provided)
    data_src = input_data_path or os.path.join(source_dir, 'input.data')
    data_dst = os.path.join(target_dir, 'input.data')
    if os.path.exists(data_src) and not os.path.exists(data_dst):
        os.symlink(os.path.abspath(data_src), data_dst)
    
    # Copy other needed files
    for fname in ['RuNNer.serial.x', 'scaling.data']:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            # Ensure executable permission for RuNNer
            if fname == 'RuNNer.serial.x':
                os.chmod(dst, 0o755)


# ============================================================================
# 5) MAIN FEATURE SELECTION LOOP
# ============================================================================

def run_feature_selection(work_dir, element, runner_exe, n_remove=1,
                          r2_threshold=0.95, ranking_method='variance',
                          max_iterations=None, timeout_per_mode=None,
                          template_file=None, input_data_path=None,
                          min_features=0):
    """
    Backward feature elimination for a single element.
    
    Parameters:
        work_dir: base working directory (with input.data, RuNNer.serial.x)
        element: 'La', 'Zr', or 'O'
        runner_exe: path to RuNNer executable
        n_remove: number of features to remove per iteration (default: 1)
        r2_threshold: stop when R² < baseline_R² * threshold (default: 0.95)
        ranking_method: 'variance' or 'sensitivity'
        max_iterations: maximum number of iterations (None = unlimited)
        timeout_per_mode: timeout for each RuNNer mode in seconds
        template_file: path to input.nn template file
    
    Returns:
        results: dict with full history of the selection process
    """
    print(f"\n{'#'*70}")
    print(f"# FEATURE SELECTION for element: {element}")
    print(f"# Working directory: {work_dir}")
    print(f"# n_remove={n_remove}, threshold={r2_threshold}, min_features={min_features}")
    print(f"# Ranking method: {ranking_method}")
    print(f"{'#'*70}\n")

    # Parse template
    if template_file is None:
        template_file = os.path.join(work_dir, 'input.nn-1')
    
    header_lines, symf_dict, _ = parse_symfunction_lines(template_file)
    elements_order = ['La', 'Zr', 'O']  # Fixed order for La2Zr2O7

    print(f"Initial feature counts:")
    for elem in elements_order:
        print(f"  {elem}: {len(symf_dict.get(elem, []))} features")

    # Results storage
    results = {
        'element': element,
        'n_remove': n_remove,
        'min_features': min_features,
        'r2_threshold': r2_threshold,
        'ranking_method': ranking_method,
        'iterations': [],
        'initial_features': {e: len(symf_dict.get(e, [])) for e in elements_order},
    }

    # Output directory for this element's selection process
    output_dir = os.path.join(work_dir, f'feature_selection_{element}')
    os.makedirs(output_dir, exist_ok=True)

    # --- Iteration 0: Baseline with all features ---
    print(f"\n{'='*60}")
    print(f"  ITERATION 0: Baseline (all features)")
    print(f"{'='*60}")

    iter_dir = os.path.join(output_dir, 'iter_000_baseline')
    os.makedirs(iter_dir, exist_ok=True)
    setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

    metrics_0 = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                  symf_dict, elements_order,
                                  timeout_per_mode=timeout_per_mode)

    if metrics_0 is None or 'r2_test' not in metrics_0:
        print("ERROR: Baseline model failed! Cannot proceed.")
        return results

    baseline_r2 = metrics_0['r2_test']
    threshold_r2 = baseline_r2 * r2_threshold

    print(f"\n  Baseline R²_test = {baseline_r2:.6f}")
    print(f"  Threshold (95%)  = {threshold_r2:.6f}")
    print(f"  RMSE train/test  = {metrics_0.get('rmse_train', 'N/A')}/{metrics_0.get('rmse_test', 'N/A')} eV/atom")

    results['baseline_r2'] = baseline_r2
    results['threshold_r2'] = threshold_r2
    results['iterations'].append({
        'iteration': 0,
        'n_features': {e: len(symf_dict.get(e, [])) for e in elements_order},
        'removed_features': [],
        'metrics': {k: v for k, v in metrics_0.items() if k != 'epoch_data'},
    })

    # Save baseline metrics
    save_iteration_results(output_dir, 0, metrics_0, symf_dict, element)

    # --- Iterative feature removal ---
    current_symf = dict(symf_dict)
    iteration = 0

    while True:
        iteration += 1
        
        if max_iterations and iteration > max_iterations:
            print(f"\n  Reached max iterations ({max_iterations}). Stopping.")
            break

        n_current = len(current_symf.get(element, []))
        # Hard safety floor: never allow an element to drop below 1 feature.
        if n_current - n_remove < 1:
            print(f"\n  Cannot remove {n_remove} feature(s): {n_current} -> {n_current - n_remove} would leave fewer than 1 feature for {element}.")
            print("  Stopping to prevent invalid input.nn / missing function.data.")
            break

        if min_features > 0:
            if n_current <= min_features:
                print(f"\n  Reached minimum allowed features for {element}: {n_current} (min={min_features}).")
                print("  Stopping to prevent over-pruning.")
                break

            if n_current - n_remove < min_features:
                print(f"\n  Cannot remove {n_remove} feature(s): {n_current} -> {n_current - n_remove} would go below min_features={min_features}.")
                print("  Stopping to prevent over-pruning.")
                break

        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}: {n_current} -> {n_current - n_remove} features for {element}")
        print(f"{'='*60}")

        # Determine previous iteration directory for ranking
        prev_dir = os.path.join(output_dir,
                                f'iter_{iteration-1:03d}_baseline' if iteration == 1
                                else f'iter_{iteration-1:03d}')

        # --- Rank features ---
        if ranking_method == 'variance':
            ranked = rank_features_by_variance(prev_dir, element)
        elif ranking_method == 'sensitivity':
            last_r2 = results['iterations'][-1]['metrics'].get('r2_test', baseline_r2)
            ranked = rank_features_by_sensitivity(
                prev_dir,
                element, header_lines, current_symf,
                elements_order, runner_exe,
                last_r2,
                timeout_per_mode=timeout_per_mode
            )
        else:
            print(f"  Unknown ranking method: {ranking_method}")
            break

        if ranked is None:
            print("  WARNING: Ranking failed, using reverse order (remove last features)")
            ranked = list(range(n_current - 1, -1, -1))

        # --- Remove least important features ---
        features_to_remove = ranked[:n_remove]
        removed_lines = []
        
        current_features = list(current_symf[element])
        # Remove in reverse order to preserve indices
        for idx in sorted(features_to_remove, reverse=True):
            if idx < len(current_features):
                removed_lines.append(current_features.pop(idx))
        current_symf[element] = current_features

        print(f"  Removed {len(removed_lines)} feature(s) from {element}:")
        for rl in removed_lines:
            print(f"    - {rl}")

        # --- Run pipeline with reduced features ---
        iter_dir = os.path.join(output_dir, f'iter_{iteration:03d}')
        os.makedirs(iter_dir, exist_ok=True)
        setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

        metrics_i = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                      current_symf, elements_order,
                                      timeout_per_mode=timeout_per_mode)

        if metrics_i is None:
            print("  ERROR: Pipeline failed for this iteration!")
            # Restore the removed features and stop
            current_symf[element] = list(symf_dict[element])  # Reset to prevent corruption
            break

        # BUG FIX: Never fall back to 0.0 when r2_test is missing.
        # 0.0 is always below threshold and would silently remove all features.
        # Instead, warn and skip this iteration without stopping.
        if 'r2_test' not in metrics_i:
            print("  WARNING: R²_test could not be computed (testpoints file missing?).")
            print("  Skipping stopping-criterion check for this iteration.")
            r2_current = None
        else:
            r2_current = metrics_i['r2_test']

        print(f"\n  Iteration {iteration} Results:")
        if r2_current is not None:
            print(f"    R²_test     = {r2_current:.6f}  (baseline: {baseline_r2:.6f})")
            print(f"    R² ratio    = {r2_current/baseline_r2:.4f}  (threshold: {r2_threshold})")
        else:
            print(f"    R²_test     = N/A  (baseline: {baseline_r2:.6f})")
        print(f"    RMSE train  = {metrics_i.get('rmse_train', 'N/A')} eV/atom")
        print(f"    RMSE test   = {metrics_i.get('rmse_test', 'N/A')} eV/atom")
        print(f"    Features    = {len(current_symf[element])} ({element})")

        # Save iteration results
        iter_result = {
            'iteration': iteration,
            'n_features': {e: len(current_symf.get(e, [])) for e in elements_order},
            'removed_features': removed_lines,
            'metrics': {k: v for k, v in metrics_i.items() if k != 'epoch_data'},
            'r2_ratio': (r2_current / baseline_r2) if (r2_current is not None and baseline_r2 != 0) else None,
        }
        results['iterations'].append(iter_result)
        save_iteration_results(output_dir, iteration, metrics_i, current_symf, element)

        # --- Check stopping criterion ---
        # Stop ONLY by R² threshold (RMSE is still reported, but not used for stopping).
        # Only evaluate threshold when R² was successfully computed.
        if r2_current is not None and r2_current < threshold_r2:
            print(f"\n  STOPPING: R²_test per-atom ({r2_current:.6f}) < threshold ({threshold_r2:.6f})")
            print(f"  Feature selection completed after {iteration} iterations.")
            # The previous iteration's feature set is the optimal one
            results['optimal_iteration'] = iteration - 1
            results['optimal_n_features'] = len(current_symf[element]) + n_remove
            break

    # If loop ended without threshold breach, current state is optimal
    if 'optimal_iteration' not in results:
        results['optimal_iteration'] = iteration
        results['optimal_n_features'] = len(current_symf.get(element, []))

    # Save final results
    results['final_features'] = {e: len(current_symf.get(e, [])) for e in elements_order}
    
    results_file = os.path.join(output_dir, 'selection_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    # BUG FIX: Write the OPTIMAL (last-good) input.nn back to work_dir.
    # Without this, the over-pruned (threshold-breaching) feature set remains
    # as the active input.nn, which is the root cause of "removes everything".
    opt_iter = results['optimal_iteration']
    opt_symf_file = os.path.join(output_dir, f'symf_iter_{opt_iter:03d}_{element}.txt')
    if os.path.exists(opt_symf_file):
        with open(opt_symf_file, 'r') as f:
            optimal_features = [line.strip() for line in f if line.strip()]
        # Rebuild the symf_dict with the optimal feature set for this element
        final_symf = dict(symf_dict)   # start from original (all elements)
        final_symf[element] = optimal_features
        optimal_input_nn = os.path.join(work_dir, f'input.nn-optimal_{element}')
        write_input_nn(optimal_input_nn, header_lines, final_symf,
                       runner_mode=2, elements_order=elements_order)
        print(f"  Optimal input.nn written to: {optimal_input_nn}")
        print(f"  Optimal feature count for {element}: {len(optimal_features)}")
    else:
        print(f"  WARNING: Could not find optimal symf file: {opt_symf_file}")
        print(f"  Optimal input.nn was NOT written back.")

    # Generate summary report
    generate_summary_report(output_dir, results)

    return results


def save_iteration_results(output_dir, iteration, metrics, symf_dict, element):
    """Save per-iteration results to files."""
    iter_file = os.path.join(output_dir, f'metrics_iter_{iteration:03d}.json')
    
    save_data = {
        'iteration': iteration,
        'metrics': {k: v for k, v in metrics.items() if k != 'epoch_data'},
        'n_features': {e: len(v) for e, v in symf_dict.items()},
    }
    
    with open(iter_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    # Also save the current symfunction set
    symf_file = os.path.join(output_dir, f'symf_iter_{iteration:03d}_{element}.txt')
    with open(symf_file, 'w') as f:
        for sf_line in symf_dict.get(element, []):
            f.write(sf_line + '\n')


def generate_summary_report(output_dir, results):
    """Generate a human-readable summary report."""
    report_file = os.path.join(output_dir, 'SUMMARY_REPORT.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  FEATURE SELECTION SUMMARY - Element: {results['element']}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Baseline R²_test:    {results.get('baseline_r2', 'N/A')}\n")
        f.write(f"Threshold (95%):     {results.get('threshold_r2', 'N/A')}\n")
        f.write(f"Ranking method:      {results.get('ranking_method', 'N/A')}\n")
        f.write(f"n_remove per step:   {results.get('n_remove', 'N/A')}\n\n")
        
        f.write(f"Initial features:    {results.get('initial_features', {})}\n")
        f.write(f"Final features:      {results.get('final_features', {})}\n\n")
        
        if 'optimal_iteration' in results:
            f.write(f"Optimal iteration:   {results['optimal_iteration']}\n")
            f.write(f"Optimal n_features:  {results.get('optimal_n_features', 'N/A')}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Iter':>5}  {'N_feat':>7}  {'R²_test':>10}  {'R²_ratio':>10}  "
                f"{'RMSE_train':>11}  {'RMSE_test':>10}\n")
        f.write("-" * 70 + "\n")
        
        for it in results.get('iterations', []):
            n_feat = it['n_features'].get(results['element'], '?')
            r2 = it['metrics'].get('r2_test', None)
            r2_str = f"{r2:.6f}" if r2 is not None else "N/A"
            ratio = it.get('r2_ratio', None)
            ratio_str = f"{ratio:.4f}" if ratio is not None else "1.0000"
            rmse_tr = it['metrics'].get('rmse_train', None)
            rmse_tr_str = f"{rmse_tr:.6f}" if rmse_tr is not None else "N/A"
            rmse_te = it['metrics'].get('rmse_test', None)
            rmse_te_str = f"{rmse_te:.6f}" if rmse_te is not None else "N/A"
            
            f.write(f"{it['iteration']:>5}  {n_feat:>7}  {r2_str:>10}  {ratio_str:>10}  "
                    f"{rmse_tr_str:>11}  {rmse_te_str:>10}\n")
        
        f.write("-" * 70 + "\n")
    
    print(f"  Summary report saved to {report_file}")


def run_all_joint_selection(work_dir, runner_exe, n_remove=1,
                            r2_threshold=0.95, ranking_method='variance',
                            max_iterations=None, timeout_per_mode=None,
                            template_file=None, input_data_path=None,
                            min_features=0):
    """
    Joint backward elimination for all elements together.

    Each iteration:
      1) remove n_remove features from La, Zr, and O in the same iteration
      2) run one RuNNer pipeline (mode1->mode2->mode3)
      3) evaluate one R² and apply stopping criterion
    """
    print(f"\n{'#'*70}")
    print("# JOINT FEATURE SELECTION for all elements (La, Zr, O)")
    print(f"# Working directory: {work_dir}")
    print(f"# n_remove={n_remove}, threshold={r2_threshold}, min_features={min_features}")
    print(f"# Ranking method: {ranking_method}")
    print(f"{'#'*70}\n")

    if template_file is None:
        template_file = os.path.join(work_dir, 'input.nn-1')

    header_lines, symf_dict, _ = parse_symfunction_lines(template_file)
    elements_order = ['La', 'Zr', 'O']

    print("Initial feature counts:")
    for elem in elements_order:
        print(f"  {elem}: {len(symf_dict.get(elem, []))} features")

    output_dir = os.path.join(work_dir, 'feature_selection_all_joint')
    os.makedirs(output_dir, exist_ok=True)

    # Baseline
    iter_dir = os.path.join(output_dir, 'iter_000_baseline')
    os.makedirs(iter_dir, exist_ok=True)
    setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

    metrics_0 = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                  symf_dict, elements_order,
                                  timeout_per_mode=timeout_per_mode)

    if metrics_0 is None or 'r2_test' not in metrics_0:
        print("ERROR: Joint baseline failed!")
        return None

    baseline_r2 = metrics_0['r2_test']
    threshold_r2 = baseline_r2 * r2_threshold

    print(f"\n  Baseline R²_test = {baseline_r2:.6f}")
    print(f"  Threshold (95%)  = {threshold_r2:.6f}")
    print(f"  RMSE train/test  = {metrics_0.get('rmse_train', 'N/A')}/{metrics_0.get('rmse_test', 'N/A')} eV/atom")

    results = {
        'mode': 'all_joint',
        'n_remove': n_remove,
        'min_features': min_features,
        'r2_threshold': r2_threshold,
        'ranking_method': ranking_method,
        'baseline_r2': baseline_r2,
        'threshold_r2': threshold_r2,
        'initial_features': {e: len(symf_dict.get(e, [])) for e in elements_order},
        'iterations': [{
            'iteration': 0,
            'n_features': {e: len(symf_dict.get(e, [])) for e in elements_order},
            'removed_features': {},
            'metrics': {k: v for k, v in metrics_0.items() if k != 'epoch_data'},
        }],
    }

    current_symf = dict(symf_dict)
    iteration = 0

    while True:
        iteration += 1

        if max_iterations and iteration > max_iterations:
            print(f"\n  Reached max iterations ({max_iterations}). Stopping.")
            break

        # Check if all elements can be reduced in this joint iteration
        can_reduce_all = True
        for elem in elements_order:
            n_current = len(current_symf.get(elem, []))
            if n_current - n_remove < 1:
                can_reduce_all = False
                print(f"\n  Stopping: {elem} would drop from {n_current} to {n_current - n_remove}, leaving fewer than 1 feature.")
                break
            if min_features > 0 and n_current - n_remove < min_features:
                can_reduce_all = False
                print(f"\n  Stopping: {elem} would fall below min_features ({n_current}->{n_current-n_remove}, min={min_features}).")
                break

        if not can_reduce_all:
            break

        print(f"\n{'='*60}")
        print(f"  JOINT ITERATION {iteration}: removing {n_remove} from each element")
        print(f"{'='*60}")

        prev_dir = os.path.join(output_dir,
                                f'iter_{iteration-1:03d}_baseline' if iteration == 1
                                else f'iter_{iteration-1:03d}')

        removed_by_element = {}

        # Rank + remove for each element before a single pipeline run
        for elem in elements_order:
            n_current = len(current_symf[elem])

            if ranking_method == 'variance':
                ranked = rank_features_by_variance(prev_dir, elem)
            elif ranking_method == 'sensitivity':
                last_r2 = results['iterations'][-1]['metrics'].get('r2_test', baseline_r2)
                ranked = rank_features_by_sensitivity(
                    prev_dir, elem, header_lines, current_symf,
                    elements_order, runner_exe, last_r2,
                    timeout_per_mode=timeout_per_mode
                )
            else:
                print(f"  Unknown ranking method: {ranking_method}")
                return results

            if ranked is None:
                print(f"  WARNING: Ranking failed for {elem}, using reverse order")
                ranked = list(range(n_current - 1, -1, -1))

            features_to_remove = ranked[:n_remove]
            current_features = list(current_symf[elem])
            removed_lines = []
            for idx in sorted(features_to_remove, reverse=True):
                if idx < len(current_features):
                    removed_lines.append(current_features.pop(idx))
            current_symf[elem] = current_features
            removed_by_element[elem] = removed_lines

            print(f"  {elem}: removed {len(removed_lines)} feature(s), now {len(current_features)}")

        # Single pipeline evaluation per joint iteration
        iter_dir = os.path.join(output_dir, f'iter_{iteration:03d}')
        os.makedirs(iter_dir, exist_ok=True)
        setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

        metrics_i = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                      current_symf, elements_order,
                                      timeout_per_mode=timeout_per_mode)

        if metrics_i is None:
            print("  ERROR: Pipeline failed for this joint iteration!")
            break

        # BUG FIX: Never fall back to 0.0 — that always triggers threshold.
        if 'r2_test' not in metrics_i:
            print("  WARNING: R²_test could not be computed. Skipping stopping check.")
            r2_current = None
        else:
            r2_current = metrics_i['r2_test']

        print(f"\n  Joint Iteration {iteration} Results:")
        if r2_current is not None:
            print(f"    R²_test     = {r2_current:.6f}  (baseline: {baseline_r2:.6f})")
            print(f"    R² ratio    = {r2_current/baseline_r2:.4f}  (threshold: {r2_threshold})")
        else:
            print(f"    R²_test     = N/A  (baseline: {baseline_r2:.6f})")
        print(f"    RMSE train  = {metrics_i.get('rmse_train', 'N/A')} eV/atom")
        print(f"    RMSE test   = {metrics_i.get('rmse_test', 'N/A')} eV/atom")

        results['iterations'].append({
            'iteration': iteration,
            'n_features': {e: len(current_symf.get(e, [])) for e in elements_order},
            'removed_features': removed_by_element,
            'metrics': {k: v for k, v in metrics_i.items() if k != 'epoch_data'},
            'r2_ratio': (r2_current / baseline_r2) if (r2_current is not None and baseline_r2 != 0) else None,
        })

        if r2_current is not None and r2_current < threshold_r2:
            print(f"\n  STOPPING: R²_test per-atom ({r2_current:.6f}) < threshold ({threshold_r2:.6f})")
            print(f"  Joint feature selection completed after {iteration} iterations.")
            results['optimal_iteration'] = iteration - 1
            results['optimal_n_features'] = {
                e: len(current_symf.get(e, [])) + n_remove for e in elements_order
            }
            break

    if 'optimal_iteration' not in results:
        results['optimal_iteration'] = iteration
        results['optimal_n_features'] = {e: len(current_symf.get(e, [])) for e in elements_order}

    results['final_features'] = {e: len(current_symf.get(e, [])) for e in elements_order}

    results_file = os.path.join(output_dir, 'selection_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    report_file = os.path.join(output_dir, 'SUMMARY_REPORT.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  JOINT FEATURE SELECTION SUMMARY (La, Zr, O)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Baseline R²_test:    {results.get('baseline_r2', 'N/A')}\n")
        f.write(f"Threshold (95%):     {results.get('threshold_r2', 'N/A')}\n")
        f.write(f"Ranking method:      {results.get('ranking_method', 'N/A')}\n")
        f.write(f"n_remove per step:   {results.get('n_remove', 'N/A')}\n")
        f.write(f"min_features:        {results.get('min_features', 'N/A')}\n\n")
        f.write(f"Initial features:    {results.get('initial_features', {})}\n")
        f.write(f"Final features:      {results.get('final_features', {})}\n\n")
        f.write(f"Optimal iteration:   {results.get('optimal_iteration', 'N/A')}\n")
        f.write(f"Optimal n_features:  {results.get('optimal_n_features', 'N/A')}\n")

    print(f"\n  Joint results saved to {results_file}")
    print(f"  Joint summary saved to {report_file}")
    return results


# ============================================================================
# 6) PHASE 2: SIMULTANEOUS REDUCTION
# ============================================================================

def run_phase2(work_dir, runner_exe, phase1_results, n_remove=1,
               r2_threshold=0.95, timeout_per_mode=None, template_file=None,
               input_data_path=None, min_features=0):
    """
    Phase 2: Simultaneously reduce features from all elements.
    
    Uses the optimal feature sets from Phase 1 as starting point,
    then iteratively removes the least important feature from the
    element whose feature contributes least.
    """
    print(f"\n{'#'*70}")
    print(f"# PHASE 2: SIMULTANEOUS FEATURE REDUCTION")
    print(f"{'#'*70}\n")

    if template_file is None:
        template_file = os.path.join(work_dir, 'input.nn-1')
    
    header_lines, symf_dict, _ = parse_symfunction_lines(template_file)
    elements_order = ['La', 'Zr', 'O']

    # If phase1 results provided, start from their optimal feature sets
    if phase1_results:
        for elem, p1_result in phase1_results.items():
            opt_iter = p1_result.get('optimal_iteration', 0)
            # Load the optimal symfunction set from phase 1
            p1_dir = os.path.join(work_dir, f'feature_selection_{elem}')
            symf_file = os.path.join(p1_dir, f'symf_iter_{opt_iter:03d}_{elem}.txt')
            if os.path.exists(symf_file):
                with open(symf_file, 'r') as f:
                    symf_dict[elem] = [line.strip() for line in f if line.strip()]
                print(f"  Loaded optimal {elem} features from Phase 1: {len(symf_dict[elem])}")

    output_dir = os.path.join(work_dir, 'feature_selection_phase2')
    os.makedirs(output_dir, exist_ok=True)

    # Run same algorithm but cycle through elements
    print(f"\nPhase 2 starting feature counts:")
    for elem in elements_order:
        print(f"  {elem}: {len(symf_dict.get(elem, []))} features")

    # Baseline
    iter_dir = os.path.join(output_dir, 'iter_000_baseline')
    os.makedirs(iter_dir, exist_ok=True)
    setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

    metrics_0 = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                  symf_dict, elements_order,
                                  timeout_per_mode=timeout_per_mode)

    if metrics_0 is None or 'r2_test' not in metrics_0:
        print("ERROR: Phase 2 baseline failed!")
        return None

    baseline_r2 = metrics_0['r2_test']
    threshold_r2 = baseline_r2 * r2_threshold

    print(f"\n  Phase 2 Baseline R²_test = {baseline_r2:.6f}")
    print(f"  Threshold (95%)          = {threshold_r2:.6f}")

    results = {
        'phase': 2,
        'baseline_r2': baseline_r2,
        'threshold_r2': threshold_r2,
        'iterations': [{'iteration': 0,
                        'n_features': {e: len(symf_dict.get(e, [])) for e in elements_order},
                        'metrics': {k: v for k, v in metrics_0.items() if k != 'epoch_data'}}],
    }

    current_symf = dict(symf_dict)
    iteration = 0
    elem_cycle = 0  # Cycle through elements

    while True:
        iteration += 1
        target_elem = elements_order[elem_cycle % len(elements_order)]
        elem_cycle += 1

        n_current = len(current_symf.get(target_elem, []))
        if min_features > 0 and (n_current <= min_features or n_current - n_remove < min_features):
            print(f"  Skipping {target_elem}: {n_current} features (min={min_features})")
            if all(len(current_symf.get(e, [])) <= min_features for e in elements_order):
                print("  All elements reached minimum feature floor. Stopping.")
                break
            continue

        print(f"\n  Phase 2 Iteration {iteration}: removing from {target_elem} "
              f"({n_current} -> {n_current - n_remove})")

        # Rank and remove
        prev_iter_dir = iter_dir
        ranked = rank_features_by_variance(prev_iter_dir, target_elem)
        if ranked is None:
            ranked = list(range(n_current - 1, -1, -1))

        features_to_remove = ranked[:n_remove]
        current_features = list(current_symf[target_elem])
        removed = []
        for idx in sorted(features_to_remove, reverse=True):
            if idx < len(current_features):
                removed.append(current_features.pop(idx))
        current_symf[target_elem] = current_features

        # Run pipeline
        iter_dir = os.path.join(output_dir, f'iter_{iteration:03d}_{target_elem}')
        os.makedirs(iter_dir, exist_ok=True)
        setup_work_dir(work_dir, iter_dir, input_data_path=input_data_path)

        metrics_i = run_full_pipeline(iter_dir, runner_exe, header_lines,
                                      current_symf, elements_order,
                                      timeout_per_mode=timeout_per_mode)

        if metrics_i is None:
            print("  Pipeline failed!")
            break

        # BUG FIX: Never fall back to 0.0 — that always triggers threshold.
        if 'r2_test' not in metrics_i:
            print("  WARNING: R²_test could not be computed. Skipping stopping check.")
            r2_current = None
        else:
            r2_current = metrics_i['r2_test']

        if r2_current is not None:
            print(f"  R²_test = {r2_current:.6f} (ratio: {r2_current/baseline_r2:.4f})")
        else:
            print(f"  R²_test = N/A")

        results['iterations'].append({
            'iteration': iteration,
            'element_reduced': target_elem,
            'n_features': {e: len(current_symf.get(e, [])) for e in elements_order},
            'removed_features': removed,
            'metrics': {k: v for k, v in metrics_i.items() if k != 'epoch_data'},
            'r2_ratio': (r2_current / baseline_r2) if (r2_current is not None and baseline_r2 != 0) else None,
        })

        if r2_current is not None and r2_current < threshold_r2:
            print(f"\n  Phase 2 STOPPING: R² below threshold")
            break

    # Save results
    results['final_features'] = {e: len(current_symf.get(e, [])) for e in elements_order}
    results_file = os.path.join(output_dir, 'phase2_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Phase 2 results saved to {results_file}")
    return results


# ============================================================================
# 7) MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated Symmetry Function Feature Selection for RuNNer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Select features for La (one at a time)
  python symf_feature_selection.py --work_dir . --element La --n_remove 1

    # Joint mode: remove features from La/Zr/O in the same iteration
  python symf_feature_selection.py --work_dir . --element all --n_remove 1

  # Phase 2: Simultaneous reduction
  python symf_feature_selection.py --work_dir . --phase2 --n_remove 1

  # Custom settings
  python symf_feature_selection.py --work_dir . --element La --n_remove 1 \\
      --threshold 0.95 --ranking variance --max_iter 50 --timeout 7200
        """
    )
    
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Working directory with input.data and RuNNer.serial.x')
    parser.add_argument('--element', type=str, default=None,
                        help='Element to select features for: La, Zr, O, or "all"')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2 (simultaneous reduction)')
    parser.add_argument('--n_remove', type=int, default=1,
                        help='Number of features to remove per iteration (default: 1)')
    parser.add_argument('--min_features', type=int, default=0,
                        help='Minimum features to keep per element (default: 0 = disabled)')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='R² threshold as fraction of baseline (default: 0.95)')
    parser.add_argument('--ranking', choices=['variance', 'sensitivity'], default='variance',
                        help='Feature ranking method (default: variance)')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Maximum number of iterations (default: unlimited)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout per RuNNer mode in seconds (default: unlimited)')
    parser.add_argument('--runner_exe', type=str, default='./RuNNer.serial.x',
                        help='Path to RuNNer executable (default: ./RuNNer.serial.x)')
    parser.add_argument('--template', type=str, default=None,
                        help='Path to input.nn template (default: work_dir/input.nn-1)')
    parser.add_argument('--input_data', type=str, default=None,
                        help='Path to input.data file (default: work_dir/input.data)')
    
    args = parser.parse_args()
    
    work_dir = os.path.abspath(args.work_dir)
    runner_exe = args.runner_exe
    template_file = args.template or os.path.join(work_dir, 'input.nn-1')
    
    # Validate
    if not os.path.exists(template_file):
        print(f"ERROR: Template file not found: {template_file}")
        sys.exit(1)
    
    runner_path = os.path.join(work_dir, runner_exe.lstrip('./'))
    if not os.path.exists(runner_path) and not shutil.which(runner_exe):
        print(f"WARNING: RuNNer executable not found: {runner_exe}")
        print(f"Searched: {runner_path}")
    
    data_file = args.input_data or os.path.join(work_dir, 'input.data')
    data_file = os.path.abspath(data_file)
    if not os.path.exists(data_file):
        print(f"ERROR: input.data not found: {data_file}")
        sys.exit(1)

    print(f"\n Configuration:")
    print(f"   Work dir:     {work_dir}")
    print(f"   Template:     {template_file}")
    print(f"   Data file:    {data_file}")
    print(f"   Runner:       {runner_exe}")
    print(f"   Threshold:    {args.threshold}")
    print(f"   n_remove:     {args.n_remove}")
    print(f"   min_features: {args.min_features}")
    print(f"   Ranking:      {args.ranking}")
    
    start_time = time.time()
    
    if args.phase2:
        # Phase 2: Load phase 1 results if available
        phase1_results = {}
        for elem in ['La', 'Zr', 'O']:
            p1_file = os.path.join(work_dir, f'feature_selection_{elem}', 'selection_results.json')
            if os.path.exists(p1_file):
                with open(p1_file, 'r') as f:
                    phase1_results[elem] = json.load(f)
                print(f"  Loaded Phase 1 results for {elem}")
        
        run_phase2(work_dir, runner_exe, phase1_results,
                   n_remove=args.n_remove, r2_threshold=args.threshold,
                   timeout_per_mode=args.timeout, template_file=template_file,
                   input_data_path=data_file, min_features=args.min_features)
    
    elif args.element:
        if args.element.lower() == 'all':
            # Joint mode: remove from La/Zr/O in the same iteration, then evaluate once
            run_all_joint_selection(
                work_dir, runner_exe,
                n_remove=args.n_remove, r2_threshold=args.threshold,
                ranking_method=args.ranking, max_iterations=args.max_iter,
                timeout_per_mode=args.timeout, template_file=template_file,
                input_data_path=data_file,
                min_features=args.min_features
            )
        else:
            run_feature_selection(
                work_dir, args.element, runner_exe,
                n_remove=args.n_remove, r2_threshold=args.threshold,
                ranking_method=args.ranking, max_iterations=args.max_iter,
                timeout_per_mode=args.timeout, template_file=template_file,
                input_data_path=data_file,
                min_features=args.min_features
            )
    else:
        print("ERROR: Specify --element or --phase2")
        parser.print_help()
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == '__main__':
    main()
