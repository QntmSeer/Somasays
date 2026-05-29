import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def plot_design_space_landscape(csv_path: str, out_img: str):
    """Generates an academic-grade scatter plot representing the binder design space landscape."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found for design space: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    # Extract coordinates
    # x-axis: predicted_dG (kcal/mol) - lower is better / more stable
    # y-axis: iptm - higher is better
    # color: wlss - composite success score
    # size: buried surface area (BSA)
    dG = df['predicted_dG'].values
    iptm = df['iptm'].values
    wlss = df['wlss'].values
    bsa = df['interface_area'].values
    names = df['candidate_name'].values
    cys = df['cysteines'].values

    plt.figure(figsize=(9, 7), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)

    # Plot cysteine-containing vs cysteine-free differently
    cys_free_idx = cys == 0
    cys_present_idx = cys > 0

    # Scatter points for Cysteine-Free (Stars)
    if any(cys_free_idx):
        sc1 = plt.scatter(
            dG[cys_free_idx], 
            iptm[cys_free_idx], 
            c=wlss[cys_free_idx], 
            s=bsa[cys_free_idx] * 0.5, 
            cmap="viridis", 
            marker="*", 
            edgecolors="black", 
            linewidths=1.2,
            vmin=40, vmax=65,
            label="Thiol-Free (Cysteine-Free)",
            zorder=3
        )
    
    # Scatter points for Cysteine-Containing (Circles)
    if any(cys_present_idx):
        sc2 = plt.scatter(
            dG[cys_present_idx], 
            iptm[cys_present_idx], 
            c=wlss[cys_present_idx], 
            s=bsa[cys_present_idx] * 0.5, 
            cmap="viridis", 
            marker="o", 
            edgecolors="red", 
            linewidths=1.0,
            vmin=40, vmax=65,
            label="Cysteine-Containing (Aggregation Risk)",
            zorder=2
        )

    # Colorbar
    cbar = plt.colorbar(label="Wet-Lab Success Score (WLSS %)")
    
    # Add candidate labels
    for i, txt in enumerate(names):
        # Shorten names for clean display
        short_name = txt if len(txt) < 25 else txt[:20] + "..."
        plt.annotate(
            short_name, 
            (dG[i], iptm[i]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=8, 
            fontweight="bold" if cys[i] == 0 else "normal",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow" if cys[i] == 0 else "white", alpha=0.7, ec="gray", lw=0.5)
        )

    plt.xlabel("Monomer Folding Stability $\Delta G$ (kcal/mol, SaProt)\n← More Stable (Favorable Folding) | Less Stable →", fontsize=10, fontweight="bold")
    plt.ylabel("Target Interface Docking Confidence (ipTM, AF3)\n← Lower Docking Confidence | Higher Docking Confidence →", fontsize=10, fontweight="bold")
    plt.title("Somasays Designed Binder Landscape: Stability vs. Binding Affinity\n(Marker Size is proportional to Buried Surface Area (BSA) in Å²)", fontsize=11, fontweight="bold", pad=15)
    
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    print(f"[SUCCESS] Saved design space plot to: {out_img}")

def plot_directed_evolution_trajectory(csv_path: str, out_img: str):
    """Generates an academic-grade line plot showing directed evolution convergence."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found for evolution: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    steps = df['step'].values
    dG = df['dG'].values
    accepted = df['accepted'].values
    reasons = df['reason'].values

    # Clean up the dG values where check failed or 9.9 default penalty was applied
    valid_dG_idx = dG < 9.0
    valid_steps = steps[valid_dG_idx]
    valid_dG = dG[valid_dG_idx]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)

    # Plot search trajectory lines connecting valid folding steps
    plt.plot(valid_steps, valid_dG, color="#4F46E5", linestyle="-", linewidth=1.5, alpha=0.7, label="Search Path", zorder=1)

    # Scatter points for accepted steps
    acc_idx = accepted == True
    acc_valid = acc_idx & valid_dG_idx
    plt.scatter(
        steps[acc_valid], 
        dG[acc_valid], 
        color="#10B981", 
        edgecolors="black", 
        s=80, 
        marker="o", 
        label="Accepted Mutation", 
        zorder=3
    )

    # Scatter points for rejected steps due to stability loss
    rej_idx = (accepted == False) & (dG < 9.0)
    plt.scatter(
        steps[rej_idx], 
        dG[rej_idx], 
        color="#EF4444", 
        edgecolors="black", 
        s=50, 
        marker="x", 
        label="Rejected (Worse Stability)", 
        zorder=2
    )

    # Mark out hard-rejected constraint steps
    constraint_failures = steps[dG >= 9.0]
    if len(constraint_failures) > 0:
        plt.scatter(
            constraint_failures, 
            np.full_like(constraint_failures, max(valid_dG) + 0.2), 
            color="#F59E0B", 
            marker="^", 
            s=40, 
            label="Hard Rejected (pI/GRAVY/MHC Filter)", 
            zorder=2
        )

    plt.xlabel("Simulated Annealing Optimization Steps", fontsize=10, fontweight="bold")
    plt.ylabel("ESM3/SaProt folding energy $\Delta G$ (kcal/mol)\n← More Stable (Favorable Folding) | Less Stable →", fontsize=10, fontweight="bold")
    plt.title("In Silico Directed Evolution Trajectory: Energy Minimization Walk", fontsize=11, fontweight="bold", pad=15)
    
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    print(f"[SUCCESS] Saved directed evolution trajectory plot to: {out_img}")

def plot_biophysical_radar_profiles(csv_path: str, out_img: str):
    """Generates an academic-grade comparative bar chart showing biophysical metric profiles."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found for profiles: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    # Select the lead candidate (Rank 1 cysteine free) vs. the average of the others
    lead = df[df['cysteines'] == 0].iloc[0]
    others = df[df['cysteines'] > 0]

    # Normalized comparison axes:
    # 1. Target Binding Affinity (ipTM)
    # 2. folding stability (-predicted_dG; positive is better)
    # 3. Thiol Safety (1.0 if cysteine-free, 0.0 otherwise)
    # 4. Isoelectric Point Safety (normalized |pI - 7.4|; further is better, capped at 3.0)
    # 5. MHC-II Immunogenicity safety (1.0 if mhc_epitopes == 0, else 1.0 - (mhc/10.0))
    # 6. Hydropathy safety (GRAVY; lower hydrophobicity is safer, normalized: 1.0 - gravy)
    
    metrics = ["Binding (ipTM)", "Stability (-dG)", "Thiol Safety", "pI Safety", "MHC Safety", "Hydropathy Safety"]
    
    def normalize_metrics(row):
        binding = row['iptm'] / 0.6  # normalize against peak ipTM
        stability = -row['predicted_dG'] if row['predicted_dG'] < 0 else 0.0
        stability = min(1.0, stability / 1.5)  # scale
        thiol = 1.0 if row['cysteines'] == 0 else 0.0
        
        pi_safety = abs(row['pi'] - 7.4)
        pi_safety = min(1.0, pi_safety / 3.0)
        
        mhc = row['mhc_epitopes']
        mhc_safety = max(0.0, 1.0 - (mhc / 10.0))
        
        gravy = row['gravy']
        gravy_safety = max(0.0, 1.0 - max(0.0, gravy))
        
        return [binding, stability, thiol, pi_safety, mhc_safety, gravy_safety]

    lead_values = normalize_metrics(lead)
    
    # Average of others
    other_values_list = []
    for _, r in others.iterrows():
        other_values_list.append(normalize_metrics(r))
    
    other_avg_values = np.mean(other_values_list, axis=0) if other_values_list else [0.0]*6

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.3, axis="y", zorder=0)

    # Plot bars
    plt.bar(x - width/2, lead_values, width, label=f"Top Lead: {lead['candidate_name']}", color="#10B981", edgecolor="black", alpha=0.9, zorder=3)
    plt.bar(x + width/2, other_avg_values, width, label="Cysteine-Containing Binders (Avg)", color="#EF4444", edgecolor="black", alpha=0.7, zorder=3)

    plt.ylabel("Normalized Safety & Efficiency Metrics (0.0 to 1.0)", fontsize=10, fontweight="bold")
    plt.title("Biophysical Manufacturability & Success Profile: Lead vs. Cysteine-Containing Binders\n(1.0 is optimal/safest, 0.0 represents high risk)", fontsize=11, fontweight="bold", pad=15)
    plt.xticks(x, metrics, fontsize=9, fontweight="bold")
    plt.ylim(0, 1.1)
    
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    print(f"[SUCCESS] Saved biophysical profile plot to: {out_img}")

if __name__ == "__main__":
    plot_design_space_landscape("outputs/joint_evaluation_report.csv", "analysis/outputs/design_space_landscape.png")
    plot_directed_evolution_trajectory("outputs/evolution_history.csv", "analysis/outputs/directed_evolution_trajectory.png")
    plot_biophysical_radar_profiles("outputs/joint_evaluation_report.csv", "analysis/outputs/biophysical_profile_comparison.png")

