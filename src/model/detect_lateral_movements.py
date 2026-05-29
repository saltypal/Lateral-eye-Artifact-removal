"""
Inference Script: Detect Lateral Eye Movements
==============================================

Load a trained classifier and use it to detect lateral eye movements
in new data, with visualization of detection results.

Usage:
    python detect_lateral_movements.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent / 'preprocessing'))
sys.path.append(str(Path(__file__).parent / 'model'))

from preprocess import EEGPreprocessor
from lateral_eye_classifier import LateralEyeClassifier


def map_predictions_to_timeline(predictions, metadata, total_samples, srate):
    """
    Map window predictions back to continuous timeline.
    
    Args:
        predictions: Binary predictions for each window
        metadata: List of window metadata dicts
        total_samples: Total number of samples in recording
        srate: Sampling rate
        
    Returns:
        timeline: (total_samples,) binary array (1=lateral detected)
        confidence: (total_samples,) confidence scores
    """
    timeline = np.zeros(total_samples, dtype=int)
    confidence = np.zeros(total_samples, dtype=float)
    counts = np.zeros(total_samples, dtype=int)  # For overlapping windows
    
    for pred, meta in zip(predictions, metadata):
        start = meta['sample_start']
        end = meta['sample_end']
        epoch = meta['epoch']
        
        # Map to global timeline
        global_start = epoch * (total_samples // len(np.unique([m['epoch'] for m in metadata]))) + start
        global_end = epoch * (total_samples // len(np.unique([m['epoch'] for m in metadata]))) + end
        
        timeline[global_start:global_end] += pred
        counts[global_start:global_end] += 1
    
    # Average overlapping windows
    timeline = (timeline / np.maximum(counts, 1)) > 0.5
    confidence = timeline / np.maximum(counts, 1)
    
    return timeline.astype(int), confidence


def visualize_detections(ic_all, predictions, metadata, events, 
                        srate, pnts, ic_to_plot=2,
                        start_epoch=0, n_epochs=5):
    """
    Visualize lateral eye movement detections overlaid on IC activations.
    
    Args:
        ic_all: (epochs, ICs, samples) IC activations
        predictions: Binary predictions for each window
        metadata: Window metadata
        events: Event list
        srate: Sampling rate
        pnts: Samples per epoch
        ic_to_plot: Which IC to visualize (0-indexed)
        start_epoch: Starting epoch for visualization
        n_epochs: Number of epochs to display
    """
    # Concatenate IC data for display window
    ic_concat = np.concatenate([ic_all[i, ic_to_plot, :] 
                                for i in range(start_epoch, start_epoch + n_epochs)])
    
    # Time axis
    total_samples = len(ic_concat)
    time = np.arange(total_samples) / srate
    
    # Create detection overlay
    detection_mask = np.zeros(total_samples, dtype=bool)
    
    for pred, meta in zip(predictions, metadata):
        if meta['epoch'] < start_epoch or meta['epoch'] >= start_epoch + n_epochs:
            continue
        
        epoch_offset = (meta['epoch'] - start_epoch) * pnts
        start_idx = epoch_offset + meta['sample_start']
        end_idx = epoch_offset + meta['sample_end']
        
        if pred == 1:  # Lateral detected
            if start_idx < total_samples and end_idx <= total_samples:
                detection_mask[start_idx:end_idx] = True
    
    # Extract events in display window
    display_events = []
    for ev in events:
        if start_epoch < ev['epoch'] <= start_epoch + n_epochs:
            epoch_offset = (ev['epoch'] - 1 - start_epoch) * pnts
            sample_in_epoch = (int(ev['latency']) - 1) % pnts
            sample_idx = epoch_offset + sample_in_epoch
            
            if 0 <= sample_idx < total_samples:
                display_events.append({
                    'sample': sample_idx,
                    'time': sample_idx / srate,
                    'type': ev['type']
                })
    
    # Plot
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # IC waveform
    ax.plot(time, ic_concat, 'k-', linewidth=0.8, label=f'IC{ic_to_plot+1}', zorder=2)
    
    # Highlight detected lateral movements
    detection_segments = []
    in_detection = False
    segment_start = 0
    
    for i in range(len(detection_mask)):
        if detection_mask[i] and not in_detection:
            segment_start = i
            in_detection = True
        elif not detection_mask[i] and in_detection:
            detection_segments.append((segment_start, i))
            in_detection = False
    
    if in_detection:
        detection_segments.append((segment_start, len(detection_mask)))
    
    for start, end in detection_segments:
        ax.axvspan(time[start], time[end], color='red', alpha=0.3, zorder=1)
    
    # Event markers
    event_colors = {
        'eye-l': 'red', 'eye-r': 'purple', 'eye-u': 'green',
        'eye-d': 'blue', 'blink': 'orange', 'fixation': 'gray'
    }
    
    for ev in display_events:
        color = event_colors.get(ev['type'], 'black')
        ax.axvline(ev['time'], color=color, linestyle='--', 
                  linewidth=1, alpha=0.6, zorder=0)
    
    # Labels and legend
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('IC Amplitude (µV)', fontsize=12)
    ax.set_title(f'Lateral Eye Movement Detection - IC{ic_to_plot+1} - Epochs {start_epoch+1}-{start_epoch+n_epochs}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Detected Lateral'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='eye-l/eye-r (ground truth)'),
        plt.Line2D([0], [0], color='k', linewidth=0.8, label=f'IC{ic_to_plot+1} activation')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """Run inference on preprocessed data."""
    
    print("\n" + "="*70)
    print("LATERAL EYE MOVEMENT DETECTION - INFERENCE")
    print("="*70 + "\n")
    
    # =================================================================
    # STEP 1: LOAD TRAINED MODEL
    # =================================================================
    print("STEP 1: LOAD TRAINED MODEL")
    print("-" * 70)
    
    model_path = "../model/lateral_eye_classifier_trained.pkl"
    
    try:
        classifier = LateralEyeClassifier.load(model_path)
        print(f"  Features: {len(classifier.feature_names)}")
        print(f"  Window size: {classifier.window_size} samples")
        print(f"  Artifact ICs: {classifier.artifact_ics}")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_pipeline.py first to train the model.")
        return
    
    # =================================================================
    # STEP 2: PREPROCESS DATA
    # =================================================================
    print("\n" + "="*70)
    print("STEP 2: PREPROCESS DATA")
    print("-" * 70)
    
    preprocessor = EEGPreprocessor(
        data_dir="../../dataset",
        set_file="without_eog_channels.set",
        ica_file="without_eog_channels_Filter_ExtRunica_ICA.set"
    )
    
    results = preprocessor.run_full_pipeline()
    
    # =================================================================
    # STEP 3: CREATE FEATURES
    # =================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATE FEATURES")
    print("-" * 70)
    
    X, y, metadata = classifier.create_windowed_dataset(
        ic_all=results['ic_all'],
        events=results['events'],
        srate=results['srate'],
        pnts=results['pnts']
    )
    
    # =================================================================
    # STEP 4: PREDICT
    # =================================================================
    print("\n" + "="*70)
    print("STEP 4: PREDICT LATERAL MOVEMENTS")
    print("-" * 70)
    
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    
    n_detected = np.sum(predictions)
    n_total = len(predictions)
    
    print(f"  Total windows: {n_total}")
    print(f"  Lateral detected: {n_detected} ({n_detected/n_total*100:.1f}%)")
    print(f"  Ground truth lateral: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    # Compare with ground truth
    correct = np.sum(predictions == y)
    accuracy = correct / n_total
    print(f"\n  Accuracy: {accuracy:.3f}")
    
    # Analyze high-confidence detections
    conf_thresh = 0.8
    high_conf = probabilities[:, 1] > conf_thresh
    print(f"\n  High-confidence detections (>{conf_thresh}): {np.sum(high_conf)}")
    
    # =================================================================
    # STEP 5: VISUALIZE
    # =================================================================
    print("\n" + "="*70)
    print("STEP 5: VISUALIZE DETECTIONS")
    print("-" * 70)
    
    print("Creating visualization...")
    
    fig = visualize_detections(
        ic_all=results['ic_all'],
        predictions=predictions,
        metadata=metadata,
        events=results['events'],
        srate=results['srate'],
        pnts=results['pnts'],
        ic_to_plot=2,  # IC3 (0-indexed)
        start_epoch=0,
        n_epochs=5
    )
    
    # Save figure
    output_path = "../model/detection_visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    # =================================================================
    # STEP 6: DETECTION SUMMARY BY EVENT TYPE
    # =================================================================
    print("\n" + "="*70)
    print("DETECTION SUMMARY BY EVENT TYPE")
    print("-" * 70)
    
    # Group by label type
    from collections import defaultdict
    detections_by_type = defaultdict(list)
    
    for pred, meta in zip(predictions, metadata):
        detections_by_type[meta['label_type']].append(pred)
    
    print("\nDetection rate by actual event type:")
    for evt_type in sorted(detections_by_type.keys()):
        preds = detections_by_type[evt_type]
        detection_rate = np.mean(preds)
        print(f"  {evt_type:15s}: {detection_rate*100:5.1f}% ({np.sum(preds)}/{len(preds)} windows)")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
