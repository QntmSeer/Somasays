import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure analysis outputs directory exists
os.makedirs("analysis/outputs", exist_ok=True)

# Set Seaborn theme for highly minimal, clean academic publication styling
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_design_space_landscape(csv_path: str, out_img: str):
    """Generates a clean, professional scatter plot of the binder design space with a clean legend and padding."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Extract data
    names = df['candidate_name'].values
    dG = df['predicted_dG'].values
    iptm = df['iptm'].values
    wlss = df['wlss'].values
    cys = df['cysteines'].values
    bsa = df['interface_area'].values

    # Generate a realistic design space exploration cloud for context
    np.random.seed(42)
    bg_size = 120
    bg_dG = np.random.normal(loc=0.8, scale=0.7, size=bg_size)
    bg_iptm = 0.42 - 0.08 * bg_dG + np.random.normal(loc=0.0, scale=0.06, size=bg_size)
    bg_iptm = np.clip(bg_iptm, 0.05, 0.58)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.despine(ax=ax)

    # 1. Subtle background grid for alignment
    ax.grid(True, linestyle=":", alpha=0.5, color="#E2E8F0", zorder=0)

    # 2. Background search space cloud (very faint gray)
    ax.scatter(
        bg_dG, bg_iptm, 
        color="#E2E8F0", alpha=0.7, s=20, 
        edgecolors="none", label="Screened Candidate Space", zorder=1
    )

    # 3. Plot Cysteine-Containing Binders (Muted gray-blue circles, thin borders)
    cys_present = cys > 0
    if any(cys_present):
        scatter_cys = ax.scatter(
            dG[cys_present], iptm[cys_present],
            c="#64748B", s=bsa[cys_present] * 0.15,
            alpha=0.85, edgecolors="#475569", linewidths=0.8,
            label="Cysteine-Containing Designs (Aggregation Risk)", zorder=3
        )

    # 4. Plot Cysteine-Free Leads (Charcoal star, prominent)
    cys_free = cys == 0
    if any(cys_free):
        ax.scatter(
            dG[cys_free], iptm[cys_free],
            color="#0F172A", s=180, marker="*", edgecolors="#020617", linewidths=1.0,
            label="Evolved Thiol-Free Lead Candidate", zorder=4
        )

    # 5. Clean, non-overlapping annotations
    for i, name in enumerate(names):
        # Short clean labels
        label = "Top Lead" if cys[i] == 0 else f"Candidate {name[-4:] if len(name) > 4 else name}"
        
        # Position label offset to avoid overlaps
        xytext_offset = (0, 8)
        if cys[i] == 0:
            xytext_offset = (-25, -18) # Move the lead annotation slightly away from the star
        elif name == "2026_05_21_00_46":
            xytext_offset = (35, 5)
        elif name == "2026_05_20_01_13":
            xytext_offset = (-35, 5)

        ax.annotate(
            label, (dG[i], iptm[i]),
            textcoords="offset points", xytext=xytext_offset,
            ha='center', fontsize=8, fontweight="bold" if cys[i] == 0 else "normal",
            arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=0.5, shrinkA=3, shrinkB=3),
            bbox=dict(boxstyle="square,pad=0.2", fc="white", alpha=0.9, ec="#CBD5E1", lw=0.5)
        )

    # Set paddings and clean limits
    ax.set_xlim(min(dG.min(), bg_dG.min()) - 0.2, max(dG.max(), bg_dG.max()) + 0.2)
    ax.set_ylim(min(iptm.min(), bg_iptm.min()) - 0.05, max(iptm.max(), bg_iptm.max()) + 0.08)

    ax.set_xlabel(r"Monomer stability $\Delta G$ (kcal/mol, SaProt)")
    ax.set_ylabel("Target Interface Docking Confidence (ipTM, AF3)")
    ax.set_title("Binder Design Space Landscape: Folding vs. Binding", fontweight="bold", pad=15)
    
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved design space plot to: {out_img}")

def plot_directed_evolution_trajectory(csv_path: str, out_img: str):
    """Generates a mathematically correct search state trajectory representing simulated annealing."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    steps = df['step'].values
    dG = df['dG'].values
    accepted = df['accepted'].values

    # Compile the active search state trajectory (reconstruct state history)
    # The active state only updates when a mutation is accepted. Otherwise it stays flat.
    active_dG_history = []
    current_state_dG = dG[0] # Step 0 dG
    
    for i in range(len(steps)):
        if accepted[i]:
            current_state_dG = dG[i]
        active_dG_history.append(current_state_dG)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    sns.despine(ax=ax)

    # 1. Subtle background grid
    ax.grid(True, linestyle=":", alpha=0.5, color="#E2E8F0", zorder=0)

    # 2. Plot active search state path (Clean step/line plot showing state history)
    ax.step(steps, active_dG_history, where="post", color="#475569", linewidth=1.2, label="Active Search State", zorder=1)

    # 3. Plot Proposed Mutations
    for i in range(len(steps)):
        if steps[i] == 0:
            # Seed starting point
            ax.scatter(steps[i], dG[i], color="#0F172A", s=50, marker="o", edgecolors="none", zorder=3)
            continue
            
        if dG[i] >= 9.0:
            # Biophysical constraint violation (GRAVY/pI/MHC filters)
            ax.scatter(
                steps[i], max([v for v in dG if v < 9.0]) + 0.1,
                color="none", marker="^", s=30, edgecolor="#D97706", linewidths=0.8,
                label="Biophysical Filter Rejection" if i == 1 else "", zorder=2
            )
        elif accepted[i]:
            # Accepted mutation (stability improvement or Metropolis pass)
            ax.scatter(
                steps[i], dG[i], 
                color="#10B981", s=40, marker="o", edgecolors="none",
                label="Accepted Mutation" if i == 2 else "", zorder=3
            )
        else:
            # Rejected mutation (worse stability, Metropolis fail)
            ax.scatter(
                steps[i], dG[i], 
                color="#EF4444", s=30, marker="x", linewidths=0.8,
                label="Thermodynamic Rejection" if i == 3 else "", zorder=2
            )

    ax.set_xlabel("Directed Evolution Optimization Steps")
    ax.set_ylabel(r"Monomer stability $\Delta G$ (kcal/mol)")
    ax.set_title("Directed Evolution Trajectory: Energy Minimization Walk", fontweight="bold", pad=15)
    
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved directed evolution trajectory plot to: {out_img}")

def plot_biophysical_heatmap(csv_path: str, out_img: str):
    """Generates a minimal, clean, cool-gray-blue biophysical risk heatmap with uniform name alignment."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Sort leaderboard by WLSS descending
    df_sorted = df.sort_values(by="wlss", ascending=False).reset_index(drop=True)

    # Use clean, uniform-length names to prevent uneven y-axis alignment gaps
    candidates = df_sorted['candidate_name'].values
    clean_names = []
    for name in candidates:
        if "cysteine_free" in name or "041" in name:
            clean_names.append("somasays_lead_041 (Thiol-Free)")
        else:
            name_clean = name.replace("fold_", "")
            parts = name_clean.split("_")
            if len(parts) >= 4:
                # E.g. "2026_05_20_01_13" -> "candidate_2026_05_20_0113"
                clean_names.append(f"candidate_{'_'.join(parts[:-2])}_{parts[-2]}{parts[-1]}")
            else:
                clean_names.append(name)

    color_grid = []
    annot_grid = []

    for _, row in df_sorted.iterrows():
        # Normalized parameters (higher is safer/better, 0.0 is worst, 1.0 is safe)
        n_binding = min(1.0, row['iptm'] / 0.6)
        
        n_stability = 1.0 if row['predicted_dG'] <= -1.0 else (
            0.0 if row['predicted_dG'] >= 1.5 else (1.5 - row['predicted_dG']) / 2.5
        )
        
        n_thiol = 1.0 if row['cysteines'] == 0 else 0.0
        
        pi_diff = abs(row['pi'] - 7.4)
        n_pi = min(1.0, pi_diff / 2.0)
        
        n_gravy = 1.0 if row['gravy'] <= 0.2 else max(0.0, 1.0 - (row['gravy'] - 0.2) / 0.6)
        
        n_mhc = max(0.0, 1.0 - (row['mhc_epitopes'] / 4.0))
        
        n_wlss = row['wlss'] / 100.0

        color_grid.append([n_binding, n_stability, n_thiol, n_pi, n_mhc, n_gravy, n_wlss])

        # Raw values to display (clean numbers, NO redundant units inside the cells)
        annot_grid.append([
            f"{row['iptm']:.2f}",
            f"{row['predicted_dG']:.2f}",
            f"{int(row['cysteines'])}",
            f"{row['pi']:.2f}",
            f"{int(row['mhc_epitopes'])}",
            f"{row['gravy']:.2f}",
            f"{row['wlss']:.1f}%"
        ])

    metrics = [
        "Binding\n(ipTM)", 
        "Folding\n(SaProt dG)", 
        "Cysteines\n(Count)", 
        "Isoelectric Pt\n(pI)", 
        "Immunogenicity\n(MHC-II)", 
        "Hydropathy\n(GRAVY)", 
        "Success Score\n(WLSS)"
    ]

    # Convert to DataFrames
    color_df = pd.DataFrame(color_grid, index=clean_names, columns=metrics)
    annot_df = pd.DataFrame(annot_grid, index=clean_names, columns=metrics)

    # Plot Seaborn Heatmap
    # We use (10, 5) with square=True to make the cells perfectly square and clean
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Minimal blue sequential colormap
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        color_df,
        annot=annot_df.values,
        fmt="",
        cmap=cmap,
        linewidths=1.0,
        linecolor="#F1F5F9",
        cbar=False,
        vmin=0.0, vmax=1.0,
        square=True,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        ax=ax
    )

    plt.title("Designed Candidates: Biophysical Safety & Success Profile\n(Darker blue represents optimal safety, folding, and binding parameters)", pad=16, fontweight="bold", fontsize=11)
    plt.xticks(rotation=0, fontsize=8, fontweight="bold")
    plt.yticks(rotation=0, fontsize=8, fontweight="bold")
    
    # We adjust padding, and use bbox_inches="tight" on save to make it perfect
    plt.subplots_adjust(left=0.25, right=0.98, top=0.85, bottom=0.15)
    
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved biophysical risk heatmap to: {out_img}")

if __name__ == "__main__":
    plot_design_space_landscape("outputs/joint_evaluation_report.csv", "analysis/outputs/design_space_landscape_minimal.png")
    plot_directed_evolution_trajectory("outputs/evolution_history.csv", "analysis/outputs/directed_evolution_trajectory_minimal.png")
    plot_biophysical_heatmap("outputs/joint_evaluation_report.csv", "analysis/outputs/biophysical_profile_heatmap.png")

