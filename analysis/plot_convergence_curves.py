import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curves(csv_path, output_image_path, window_size=50):
    """
    Reads the Somasays training CSV telemetry and generates premium loss plots.
    """
    print("================================================")
    print("  Somasays Convergence Visualizer  ")
    print("================================================")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Log file not found at: {csv_path}")
        print("Make sure your sync daemon is running or manually pull the file!")
        return

    print(f"[*] Reading training logs from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("[WARNING] The CSV file contains no data rows yet. Let it train a bit more!")
        return
        
    print(f"[*] Successfully parsed {len(df)} training steps.")
    
    # Create an incremental global step index for clean x-axis plotting
    df['Global_Step'] = np.arange(len(df))
    
    # Calculate rolling averages for smooth trendlines
    print(f"[*] Calculating smooth trendlines (moving average window = {window_size} steps)...")
    df['Smooth_Total'] = df['Total_Loss'].rolling(window=window_size, min_periods=1).mean()
    df['Smooth_Seq'] = df['Seq_Loss'].rolling(window=window_size, min_periods=1).mean()
    df['Smooth_Struct'] = df['Struct_Loss'].rolling(window=window_size, min_periods=1).mean()

    # Apply premium matplotlib styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    c_total_raw = '#d3d3d3'
    c_total_smooth = '#34495e'
    c_seq = '#3498db'
    c_struct = '#e74c3c'
    
    # ========================================================
    # Plot 1: Total Multi-Objective Loss
    # ========================================================
    ax1.plot(df['Global_Step'], df['Total_Loss'], color=c_total_raw, alpha=0.4, label='Raw Loss')
    ax1.plot(df['Global_Step'], df['Smooth_Total'], color=c_total_smooth, linewidth=2.5, label=f'Smoothed Trend ({window_size} window)')
    
    ax1.set_title('Multi-Objective Joint Convergence (Total Loss)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Training Steps (Global)', fontsize=11, labelpad=10)
    ax1.set_ylabel('Cross-Entropy Loss Value', fontsize=11, labelpad=10)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ========================================================
    # Plot 2: Sequence vs. Structural Loss Comparison
    # ========================================================
    ax2.plot(df['Global_Step'], df['Smooth_Seq'], color=c_seq, linewidth=2.5, label='Sequence Loss (1D Tokens)')
    ax2.plot(df['Global_Step'], df['Smooth_Struct'], color=c_struct, linewidth=2.5, label='Structure Loss (3D Coordinate Quantization)')
    
    # Underlay faint raw data for depth
    ax2.plot(df['Global_Step'], df['Seq_Loss'], color=c_seq, alpha=0.15)
    ax2.plot(df['Global_Step'], df['Struct_Loss'], color=c_struct, alpha=0.15)

    ax2.set_title('Component Optimization: Sequence vs. 3D Geometry', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Training Steps (Global)', fontsize=11, labelpad=10)
    ax2.set_ylabel('Loss Value', fontsize=11, labelpad=10)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Super-title to capture context
    current_epoch = df['Epoch'].max()
    fig.suptitle(f'Somasays Multimodal ESM3 Fine-Tuning Analytics (Up to Epoch {current_epoch})', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the high-resolution visual
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Publication-quality plot saved to: {output_image_path}")
    print("================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Somasays Training Loss Curves")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="../Somasays_Backup/logs/loss_log.csv",
        help="Path to your synchronized loss_log.csv"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="outputs/loss_convergence_plot.png",
        help="Where to save the output image plot"
    )
    parser.add_argument(
        "--smooth", 
        type=int, 
        default=50,
        help="Rolling window size for smoothing"
    )
    
    args = parser.parse_args()
    
    plot_convergence_curves(
        csv_path=args.csv,
        output_image_path=args.out,
        window_size=args.smooth
    )
