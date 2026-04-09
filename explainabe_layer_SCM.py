"""
SCM CAUSAL ANALYSIS MODULE - MODIFIED VERSION
==============================================

This module performs causal explanation using three SCM models:
1. Acyclic SCM: Define causal structure (DAG)
2. Parameterized SCM: Learn structural equations
3. Stochastic SCM: Compute causal contributions and counterfactuals

MODIFICATIONS:
- Ranked horizontal bar chart: Shows TOTAL values only
- Pareto chart: Uses ONLY late deliveries
- New pie chart: Delivery status distribution
- Counterfactual analysis: CSV + before/after visualization

OUTPUTS:
- scm17_causal_results_complete.csv (ALL deliveries with status)
- scm17_counterfactual_analysis.csv (Counterfactual results)
- scm17_pareto_chart.png (Late deliveries only)
- scm17_ranked_horizontal_barchart.png (Total values)
- scm17_delivery_status_pie.png (New)
- scm17_counterfactual_comparison.png (New)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import networkx as nx
from tqdm import tqdm
import pickle
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


# ============================================================================
# MODEL 1: ACYCLIC SCM
# ============================================================================

class AcyclicSCM:
    """
    MODEL 1: Acyclic SCM
    
    Defines and validates the causal structure (DAG) among variables
    """
    
    def __init__(self):
        self.dag = None
        self.mechanism_names = [
            'Supplier Efficiency',
            'Demand Complexity', 
            'Pricing Pressure',
            'Logistics Constraints',
            'Geographic Friction',
            'Execution Delay',
            'Late Delivery Risk'
        ]
        self.node_mapping = {
            'E': 'Supplier Efficiency',
            'Q': 'Demand Complexity',
            'P': 'Pricing Pressure',
            'L': 'Logistics Constraints',
            'G': 'Geographic Friction',
            'D': 'Execution Delay',
            'Y': 'Late Delivery Risk'
        }
    
    def build_dag(self):
        """Build the causal DAG structure"""
        print("\n" + "="*70)
        print("MODEL 1: ACYCLIC SCM - BUILDING DAG")
        print("="*70)
        
        self.dag = nx.DiGraph()
        nodes = ['E', 'Q', 'P', 'L', 'G', 'D', 'Y']
        self.dag.add_nodes_from(nodes)
        
        print("\n✓ Nodes added:")
        for node, name in self.node_mapping.items():
            print(f"  {node}: {name}")
        
        edges = [
            ('E', 'D'), ('Q', 'D'), ('L', 'D'), ('G', 'D'), ('D', 'Y'),
            ('E', 'Y'), ('Q', 'Y'), ('P', 'Y'), ('L', 'Y'), ('G', 'Y'),
        ]
        
        self.dag.add_edges_from(edges)
        print("\n✓ Edges added (causal relationships established)")
        
        return self.dag
    
    def verify_acyclicity(self):
        """Verify that the DAG is acyclic"""
        print("\n" + "-"*70)
        print("VERIFYING ACYCLICITY")
        print("-"*70)
        
        is_acyclic = nx.is_directed_acyclic_graph(self.dag)
        
        if is_acyclic:
            print("✓ DAG is acyclic (no cycles detected)")
        else:
            print("❌ ERROR: DAG contains cycles!")
            cycles = list(nx.simple_cycles(self.dag))
            print(f"  Cycles found: {cycles}")
        
        return is_acyclic
    
    def save_dag(self, filepath='scm17_causal_dag.gml'):
        """Save DAG to file"""
        nx.write_gml(self.dag, filepath)
        print(f"\n✓ DAG saved to: {filepath}")
    
    def visualize_dag(self, filepath='scm17_causal_dag.png'):
        """Create visualization of DAG"""
        plt.figure(figsize=(12, 8))
        
        pos = {
            'E': (0, 2), 'Q': (1, 2), 'P': (2, 2),
            'L': (3, 2), 'G': (4, 2), 'D': (2, 1), 'Y': (2, 0)
        }
        
        nx.draw_networkx_nodes(self.dag, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(self.dag, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, width=2)
        
        labels = {node: f"{node}\n{self.node_mapping[node]}" for node in self.dag.nodes()}
        nx.draw_networkx_labels(self.dag, pos, labels, font_size=10, font_weight='bold')
        
        plt.title("Causal DAG: Supply Chain Late Delivery", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ DAG visualization saved to: {filepath}")


# ============================================================================
# MODEL 2: PARAMETERIZED SCM
# ============================================================================

class ParameterizedSCM:
    """
    MODEL 2: Parameterized SCM
    
    Learns structural equations using linear regression
    """
    
    def __init__(self):
        self.model_D = None
        self.model_Y = None
        self.params_D = None
        self.params_Y = None
    
    def fit_mediator_equation(self, E, Q, L, G, D):
        """Fit mediator equation: D = f_D(E, Q, L, G) + U_D"""
        print("\n" + "="*70)
        print("MODEL 2: PARAMETERIZED SCM - MEDIATOR EQUATION")
        print("="*70)
        
        X_delay = np.column_stack([E, Q, L, G])
        self.model_D = LinearRegression()
        self.model_D.fit(X_delay, D)
        
        self.params_D = {
            'intercept': self.model_D.intercept_,
            'beta_E': self.model_D.coef_[0],
            'beta_Q': self.model_D.coef_[1],
            'beta_L': self.model_D.coef_[2],
            'beta_G': self.model_D.coef_[3]
        }
        
        r2 = self.model_D.score(X_delay, D)
        print(f"✓ Mediator equation fitted (R² = {r2:.4f})")
        
        return self.model_D, self.params_D, r2
    
    def fit_outcome_equation(self, D, E, Q, P, L, G, Y_hat):
        """Fit outcome equation: Ŷ = f_Y(D, E, Q, P, L, G) + U_Y"""
        print("\n" + "="*70)
        print("MODEL 2: PARAMETERIZED SCM - OUTCOME EQUATION")
        print("="*70)
        
        X_outcome = np.column_stack([D, E, Q, P, L, G])
        self.model_Y = LinearRegression()
        self.model_Y.fit(X_outcome, Y_hat)
        
        self.params_Y = {
            'intercept': self.model_Y.intercept_,
            'beta_D': self.model_Y.coef_[0],
            'beta_E': self.model_Y.coef_[1],
            'beta_Q': self.model_Y.coef_[2],
            'beta_P': self.model_Y.coef_[3],
            'beta_L': self.model_Y.coef_[4],
            'beta_G': self.model_Y.coef_[5]
        }
        
        r2 = self.model_Y.score(X_outcome, Y_hat)
        print(f"✓ Outcome equation fitted (R² = {r2:.4f})")
        
        return self.model_Y, self.params_Y, r2
    
    def save_parameters(self, filepath='scm17_parameters.json'):
        """Save learned parameters to JSON"""
        params = {
            'mediator_equation': self.params_D,
            'outcome_equation': self.params_Y,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✓ Parameters saved to: {filepath}")


# ============================================================================
# MODEL 3: STOCHASTIC SCM
# ============================================================================

class StochasticSCM:
    """
    MODEL 3: Stochastic SCM
    
    Computes noise terms, causal contributions, and counterfactuals
    """
    
    def __init__(self, model_D, model_Y):
        self.model_D = model_D
        self.model_Y = model_Y
        self.U_D = None
        self.U_Y = None
    
    def compute_noise_terms(self, E, Q, L, G, D, P, Y_hat):
        """Compute exogenous noise terms (Abduction)"""
        print("\n" + "="*70)
        print("MODEL 3: STOCHASTIC SCM - COMPUTING NOISE TERMS")
        print("="*70)
        
        X_delay = np.column_stack([E, Q, L, G])
        D_pred = self.model_D.predict(X_delay)
        self.U_D = D - D_pred
        
        X_outcome = np.column_stack([D, E, Q, P, L, G])
        Y_fitted = self.model_Y.predict(X_outcome)
        self.U_Y = Y_hat - Y_fitted
        
        print(f"✓ Noise terms computed (U_D std={self.U_D.std():.4f}, U_Y std={self.U_Y.std():.4f})")
        
        return self.U_D, self.U_Y
    
    def robust_normalize(self, values, current_value):
        """Robust normalization using IQR"""
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0
        median = np.median(values)
        return abs((current_value - median) / iqr)
    
    def compute_causal_contributions(self, i, E, D, Q, P, L, G, 
                                    D_raw, Q_raw, P_raw, L_raw, G_raw):
        """Compute causal contributions for order i"""
        beta = self.model_Y.coef_
        
        # Compute robust normalized magnitudes
        C_D = self.robust_normalize(D_raw, D_raw[i])
        C_E = abs(1.0 - E[i])
        C_Q = self.robust_normalize(Q_raw, Q_raw[i])
        C_P = self.robust_normalize(P_raw, P_raw[i])
        C_L = self.robust_normalize(L_raw, L_raw[i])
        C_G = self.robust_normalize(G_raw, G_raw[i])
        C_U = self.robust_normalize(self.U_Y, self.U_Y[i]) * 0.5
        
        raw_contributions = np.array([C_E, C_D, C_Q, C_P, C_L, C_G, C_U])
        
        beta_weights = np.abs(beta)
        beta_weights = beta_weights / (beta_weights.sum() + 1e-10)
        
        feature_contributions = raw_contributions / (raw_contributions.sum() + 1e-10)
        blended_contributions = 0.7 * feature_contributions + 0.3 * np.append(beta_weights, 0.1)
        
        contributions_norm = blended_contributions / (blended_contributions.sum() + 1e-10)
        
        mechanism_names = [
            'Supplier Efficiency',
            'Execution Delay',
            'Demand Complexity',
            'Pricing Pressure',
            'Logistics Constraints',
            'Geographic Friction',
            'Uncertainty'
        ]
        
        contributions_dict = dict(zip(mechanism_names, contributions_norm))
        dominant_cause = max(contributions_dict.items(), key=lambda x: x[1])[0]
        
        return contributions_norm, dominant_cause
    
    def compute_counterfactual(self, i, E, Q, P, L, G, intervention_type='improve_efficiency'):
        """
        Compute counterfactual: What if we intervene on a causal mechanism?
        
        Intervention types:
        - 'improve_efficiency': Set supplier efficiency to top 10%
        - 'reduce_delay': Reduce execution delay to median
        - 'optimize_logistics': Improve logistics constraints to top quartile
        """
        # Original predicted mediator
        X_delay_orig = np.column_stack([E, Q, L, G])
        D_orig = self.model_D.predict(X_delay_orig)[i]
        
        # Original predicted outcome
        X_outcome_orig = np.column_stack([
            D_orig, E[i], Q[i], P[i], L[i], G[i]
        ]).reshape(1, -1)
        Y_orig = self.model_Y.predict(X_outcome_orig)[0]
        
        # Apply intervention
        if intervention_type == 'improve_efficiency':
            # Set efficiency to 90th percentile (top 10%)
            E_cf = np.percentile(E, 90)
            X_delay_cf = np.column_stack([E_cf, Q[i], L[i], G[i]]).reshape(1, -1)
            D_cf = self.model_D.predict(X_delay_cf)[0] + self.U_D[i]
            X_outcome_cf = np.column_stack([D_cf, E_cf, Q[i], P[i], L[i], G[i]]).reshape(1, -1)
            
        elif intervention_type == 'reduce_delay':
            # Keep original E, but set D to median
            D_cf = np.median(self.model_D.predict(X_delay_orig))
            X_outcome_cf = np.column_stack([D_cf, E[i], Q[i], P[i], L[i], G[i]]).reshape(1, -1)
            
        elif intervention_type == 'optimize_logistics':
            # Set logistics to 25th percentile (lower is better)
            L_cf = np.percentile(L, 25)
            X_delay_cf = np.column_stack([E[i], Q[i], L_cf, G[i]]).reshape(1, -1)
            D_cf = self.model_D.predict(X_delay_cf)[0] + self.U_D[i]
            X_outcome_cf = np.column_stack([D_cf, E[i], Q[i], P[i], L_cf, G[i]]).reshape(1, -1)
        
        # Counterfactual outcome
        Y_cf = self.model_Y.predict(X_outcome_cf)[0]
        
        # Causal effect (reduction in risk)
        causal_effect = Y_orig - Y_cf
        
        return {
            'original_risk': Y_orig,
            'counterfactual_risk': Y_cf,
            'causal_effect': causal_effect,
            'intervention': intervention_type
        }


# ============================================================================
# MODIFIED VISUALIZATION FUNCTIONS
# ============================================================================

def create_pareto_chart(results_df, output_path='scm17_pareto_chart.png'):
    """
    MODIFIED: Create Pareto chart showing dominant causes - LATE DELIVERIES ONLY
    """
    print("\n" + "="*70)
    print("CREATING PARETO CHART (Late Deliveries Only)")
    print("="*70)
    
    # Filter ONLY late deliveries
    late_df = results_df[results_df['Delivery_Status'] == 'Late Delivery']
    
    # Get dominant cause counts
    cause_counts = late_df['Dominant_Cause'].value_counts()
    cause_counts_sorted = cause_counts.sort_values(ascending=False)
    
    # Calculate cumulative percentage
    cumulative_pct = cause_counts_sorted.cumsum() / cause_counts_sorted.sum() * 100
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar chart
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    bars = ax1.bar(range(len(cause_counts_sorted)), 
                   cause_counts_sorted.values,
                   color=colors[:len(cause_counts_sorted)],
                   alpha=0.85,
                   edgecolor='black',
                   linewidth=1.5)
    
    ax1.set_xlabel('Causal Mechanism', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency (Count)', fontsize=13, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(range(len(cause_counts_sorted)))
    ax1.set_xticklabels(cause_counts_sorted.index, rotation=45, ha='right', fontsize=11)
    
    # Add value labels on bars
    for bar, count in zip(bars, cause_counts_sorted.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(cumulative_pct)), cumulative_pct.values,
            color='darkred', marker='o', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=13, fontweight='bold', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='80% Threshold')
    
    # Title
    plt.title(f'Pareto Chart: Dominant Causes of Late Deliveries (n={len(late_df)})\n' +
              '(80/20 Rule - Focus on top contributors)',
              fontsize=14, fontweight='bold', pad=20)
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Pareto chart saved to: {output_path} (Late deliveries only)")


def create_ranked_horizontal_barchart(results_df, output_path='scm17_ranked_horizontal_barchart.png'):
    """
    MODIFIED: Create ranked horizontal bar chart showing TOTAL values only
    """
    print("\n" + "="*70)
    print("CREATING RANKED HORIZONTAL BAR CHART (Total Values)")
    print("="*70)
    
    contribution_cols = [
        'Supplier_Efficiency_Contribution',
        'Execution_Delay_Contribution',
        'Demand_Complexity_Contribution',
        'Pricing_Pressure_Contribution',
        'Logistics_Constraints_Contribution',
        'Geographic_Friction_Contribution',
        'Uncertainty_Contribution'
    ]
    
    # Calculate TOTAL means across ALL deliveries
    total_means = results_df[contribution_cols].mean()
    
    # Clean names
    mechanism_names = [col.replace('_Contribution', '').replace('_', ' ') for col in contribution_cols]
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        'Mechanism': mechanism_names,
        'Total_Contribution': total_means.values
    })
    
    # Sort by total contribution
    comparison_df = comparison_df.sort_values('Total_Contribution', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(comparison_df))
    
    # Plot bars - single color for total
    bars = ax.barh(y_pos, comparison_df['Total_Contribution'], 
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison_df['Mechanism'], fontsize=11)
    ax.set_xlabel('Average Total Contribution', fontsize=13, fontweight='bold')
    ax.set_title('Ranked Causal Mechanisms by Total Contribution\n' +
                 'Average contribution magnitude across all deliveries',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for bar in bars:
        width_val = bar.get_width()
        if width_val > 0.001:
            ax.text(width_val, bar.get_y() + bar.get_height()/2,
                   f'{width_val:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Ranked horizontal bar chart saved to: {output_path} (Total values only)")


def create_delivery_status_pie_chart(results_df, output_path='scm17_delivery_status_pie.png'):
    """
    NEW: Create pie chart showing delivery status distribution with percentages
    """
    print("\n" + "="*70)
    print("CREATING DELIVERY STATUS PIE CHART")
    print("="*70)
    
    # Count deliveries by status
    status_counts = results_df['Delivery_Status'].value_counts()
    
    # Calculate percentages
    total = status_counts.sum()
    percentages = (status_counts / total * 100).round(2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors
    colors = ['#e74c3c', '#2ecc71']  # Red for late, green for on-time
    explode = (0.05, 0)  # Explode the late delivery slice
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Enhance autotext
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    # Add title with counts
    plt.title(f'Delivery Status Distribution\n' +
              f'Total Deliveries: {total:,}\n' +
              f'Late: {status_counts.get("Late Delivery", 0):,} ({percentages.get("Late Delivery", 0):.1f}%) | ' +
              f'On-Time: {status_counts.get("On-Time Delivery", 0):,} ({percentages.get("On-Time Delivery", 0):.1f}%)',
              fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Delivery status pie chart saved to: {output_path}")
    print(f"  - Late Deliveries: {status_counts.get('Late Delivery', 0):,} ({percentages.get('Late Delivery', 0):.1f}%)")
    print(f"  - On-Time Deliveries: {status_counts.get('On-Time Delivery', 0):,} ({percentages.get('On-Time Delivery', 0):.1f}%)")


def create_counterfactual_comparison(counterfactual_df, output_path='scm17_counterfactual_comparison.png'):
    """
    NEW: Create visualization comparing before and after counterfactual interventions
    """
    print("\n" + "="*70)
    print("CREATING COUNTERFACTUAL COMPARISON VISUALIZATION")
    print("="*70)
    
    # Group by intervention type and calculate averages
    intervention_summary = counterfactual_df.groupby('Intervention_Type').agg({
        'Original_Risk': 'mean',
        'Counterfactual_Risk': 'mean',
        'Causal_Effect': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    
    intervention_summary.columns = ['Intervention', 'Before', 'After', 'Effect', 'Count']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== LEFT PLOT: Before vs After ==========
    x_pos = np.arange(len(intervention_summary))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, intervention_summary['Before'], 
                    width, label='Before Intervention (Original Risk)', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, intervention_summary['After'], 
                    width, label='After Intervention (Counterfactual Risk)', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Late Delivery Risk', fontsize=12, fontweight='bold')
    ax1.set_title('Counterfactual Analysis: Before vs After Intervention\n' +
                  'Average risk reduction by intervention type',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(intervention_summary['Intervention'], rotation=15, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========== RIGHT PLOT: Causal Effect ==========
    bars3 = ax2.bar(x_pos, intervention_summary['Effect'], 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Risk Reduction (Causal Effect)', fontsize=12, fontweight='bold')
    ax2.set_title('Causal Effect of Interventions\n' +
                  'Average reduction in late delivery risk',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(intervention_summary['Intervention'], rotation=15, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels and sample counts
    for i, (bar, count) in enumerate(zip(bars3, intervention_summary['Count'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n(n={int(count)})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Counterfactual comparison saved to: {output_path}")
    
    # Print summary
    print("\nIntervention Effectiveness Summary:")
    for _, row in intervention_summary.iterrows():
        reduction_pct = (row['Effect'] / row['Before'] * 100) if row['Before'] > 0 else 0
        print(f"  - {row['Intervention']}: {reduction_pct:.1f}% risk reduction (n={int(row['Count'])})")
# ======================================================================================================================================
def create_global_heatmap_and_matrix(results_df, 
                                     heatmap_path='scm17_global_heatmap.png',
                                     matrix_csv='scm17_global_contribution_matrix.csv'):
    """
    NEW: Create normalized global contribution heatmap + matrix CSV
    """
    print("\n" + "="*70)
    print("CREATING GLOBAL CONTRIBUTION HEATMAP + MATRIX")
    print("="*70)

    contribution_cols = [
        'Supplier_Efficiency_Contribution',
        'Execution_Delay_Contribution',
        'Demand_Complexity_Contribution',
        'Pricing_Pressure_Contribution',
        'Logistics_Constraints_Contribution',
        'Geographic_Friction_Contribution',
        'Uncertainty_Contribution'
    ]

    # Row-wise normalization
    df_norm = results_df.copy()
    df_norm[contribution_cols] = df_norm[contribution_cols].abs()
    df_norm[contribution_cols] = df_norm[contribution_cols].div(
        df_norm[contribution_cols].sum(axis=1), axis=0
    )

    # Global matrix
    global_importance = df_norm[contribution_cols].mean().sort_values(ascending=False)
    global_importance.to_csv(matrix_csv)

    print("\n✓ Global Contribution Ranking:")
    print(global_importance)

    # Heatmap (limit rows for readability)
    plt.figure(figsize=(12,6))
    sns.heatmap(
        df_norm[contribution_cols].head(50),
        cmap="viridis",
        cbar=True
    )
    plt.title("Global Causal Contribution Heatmap (First 50 Orders)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"✓ Heatmap saved to: {heatmap_path}")
    print(f"✓ Global matrix saved to: {matrix_csv}")
# -------------------------------------------------------------------------------------------------------------------------------
def create_risk_vs_efficiency_plot(results_df, 
                                   output_path='scm17_risk_vs_efficiency.png'):
    """
    NEW: Risk vs Supplier Efficiency scatter plot
    """
    print("\n" + "="*70)
    print("CREATING RISK vs SUPPLIER EFFICIENCY PLOT")
    print("="*70)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=results_df['Supplier_Efficiency_Contribution'],
        y=results_df['Predicted_Risk'],
        alpha=0.6
    )

    plt.xlabel("Supplier Efficiency Contribution", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted Late Delivery Risk", fontsize=12, fontweight='bold')
    plt.title("Risk vs Supplier Efficiency Contribution",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Risk vs Efficiency plot saved to: {output_path}")
# -----------------------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def compute_classification_metrics(results_df):
    """
    NEW: Compute model evaluation metrics
    """
    print("\n" + "="*70)
    print("COMPUTING CLASSIFICATION METRICS")
    print("="*70)

    results_df['Actual_Label'] = results_df['Delivery_Status'].apply(
        lambda x: 1 if x == 'Late Delivery' else 0
    )

    results_df['Predicted_Label'] = results_df['Predicted_Risk'].apply(
        lambda x: 1 if x >= 0.5 else 0
    )

    print("\nClassification Report:")
    print(classification_report(results_df['Actual_Label'],
                                results_df['Predicted_Label']))

    auc = roc_auc_score(results_df['Actual_Label'],
                        results_df['Predicted_Risk'])
    print(f"AUC Score: {auc:.4f}")

    cm = confusion_matrix(results_df['Actual_Label'],
                          results_df['Predicted_Label'])

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("scm17_confusion_matrix.png", dpi=300)
    plt.close()

    print("✓ Confusion matrix saved to scm17_confusion_matrix.png")

# ============================================================================
# MAIN SCM PIPELINE
# ============================================================================

def main(input_csv='deatcn11_dataset.csv', 
         output_csv='scm17_causal_results_complete.csv',
         counterfactual_csv='scm17_counterfactual_analysis.csv',
         checkpoint_file='scm17_checkpoint.pkl'):
    """
    Main SCM causal analysis pipeline - Modified version
    """
    print("\n" + "="*70)
    print("SCM CAUSAL ANALYSIS MODULE - MODIFIED")
    print("="*70)
    print(f"\nInput: {input_csv}")
    print(f"Outputs: {output_csv}, {counterfactual_csv}")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING ENRICHED DATASET")
    print("="*70)
    
    try:
        df = pd.read_csv(input_csv, encoding='latin-1')
        print(f"✓ Dataset loaded: {len(df)} records")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found at {input_csv}")
        return None
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {str(e)}")
        return None
    
    # ========================================================================
    # STEP 2: EXTRACT FEATURES
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING CAUSAL MECHANISM VARIABLES")
    print("="*70)
    
    E = df['Supplier_Efficiency'].values
    Y_hat = df['Late_delivery_risk_prediction'].values
    
    # Extract mechanism variables
    Dr_raw = (df['Days for shipping (real)'] - df['Days for shipment (scheduled)']).fillna(0).clip(lower=0).values
    Q_raw = (df['Order Item Quantity'].fillna(1) * np.log1p(df['Sales'].fillna(100))).values
    P_raw = (df['Order Item Discount Rate'].fillna(0) * (1 - df['Order Item Profit Ratio'].fillna(0.1))).values
    
    shipping_mode_map = {'Standard Class': 4, 'Second Class': 3, 'First Class': 2, 'Same Day': 1}
    L_raw = df['Shipping Mode'].map(shipping_mode_map).fillna(4).values + np.log1p(df['Order Item Quantity'].fillna(1)).values
    
    market_map = {'US': 1, 'LATAM': 2, 'Europe': 3, 'Pacific Asia': 4, 'Africa': 5}
    G_raw = df['Market'].map(market_map).fillna(1).values
    G_raw = G_raw + np.log1p(np.abs(df['Latitude'].fillna(0)) + np.abs(df['Longitude'].fillna(0))).values / 10
    
    print("✓ Features extracted")
    
    # Standardize
    scaler = StandardScaler()
    Dr = scaler.fit_transform(Dr_raw.reshape(-1, 1)).flatten()
    Q = scaler.fit_transform(Q_raw.reshape(-1, 1)).flatten()
    P = scaler.fit_transform(P_raw.reshape(-1, 1)).flatten()
    L = scaler.fit_transform(L_raw.reshape(-1, 1)).flatten()
    G = scaler.fit_transform(G_raw.reshape(-1, 1)).flatten()
    
    # ========================================================================
    # STEP 3-5: BUILD SCM MODELS
    # ========================================================================
    print("\n" + "="*70)
    print("BUILDING THREE SCM MODELS")
    print("="*70)
    
    # Model 1: Acyclic
    acyclic_scm = AcyclicSCM()
    dag = acyclic_scm.build_dag()
    is_acyclic = acyclic_scm.verify_acyclicity()
    
    if not is_acyclic:
        print("\n❌ ERROR: DAG contains cycles")
        return None
    
    acyclic_scm.save_dag()
    acyclic_scm.visualize_dag()
    
    # Model 2: Parameterized
    param_scm = ParameterizedSCM()
    model_D, params_D, r2_D = param_scm.fit_mediator_equation(E, Q, L, G, Dr)
    model_Y, params_Y, r2_Y = param_scm.fit_outcome_equation(Dr, E, Q, P, L, G, Y_hat)
    param_scm.save_parameters()
    
    # Model 3: Stochastic
    stoch_scm = StochasticSCM(model_D, model_Y)
    U_D, U_Y = stoch_scm.compute_noise_terms(E, Q, L, G, Dr, P, Y_hat)
    
    # ========================================================================
    # STEP 6: COMPUTE CAUSAL EXPLANATIONS (ALL ORDERS)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: COMPUTING CAUSAL EXPLANATIONS (ALL ORDERS)")
    print("="*70)
    
    # Check checkpoint
    start_idx = 0
    results = []
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                results = checkpoint['results']
                start_idx = checkpoint['last_completed_idx'] + 1
                print(f"✓ Checkpoint loaded: Resuming from record {start_idx}")
        except:
            start_idx = 0
    
    # Process ALL orders
    print(f"\nProcessing ALL {len(df)} orders...")
    
    checkpoint_interval = 500
    
    for i in tqdm(range(start_idx, len(df)), desc="Computing Explanations", ncols=100):
        # Compute contributions
        contributions, dominant = stoch_scm.compute_causal_contributions(
            i, E, Dr, Q, P, L, G,
            Dr_raw, Q_raw, P_raw, L_raw, G_raw
        )
        
        # Determine delivery status
        delivery_status = 'Late Delivery' if Y_hat[i] > 0.5 else 'On-Time Delivery'
        
        # Store result
        result = {
            'Order_ID': df.iloc[i]['Order Id'],
            'Predicted_Risk': Y_hat[i],
            'Delivery_Status': delivery_status,
            'Supplier_Efficiency_Contribution': contributions[0],
            'Execution_Delay_Contribution': contributions[1],
            'Demand_Complexity_Contribution': contributions[2],
            'Pricing_Pressure_Contribution': contributions[3],
            'Logistics_Constraints_Contribution': contributions[4],
            'Geographic_Friction_Contribution': contributions[5],
            'Uncertainty_Contribution': contributions[6],
            'Dominant_Cause': dominant
        }
        results.append(result)
        
        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            checkpoint = {
                'results': results,
                'last_completed_idx': i,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
    
    # Remove checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # ========================================================================
    # STEP 7: SAVE CAUSAL RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING CAUSAL RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Causal results saved to: {output_csv}")
    print(f"  - Total orders: {len(results_df)}")
    print(f"  - Late deliveries: {len(results_df[results_df['Delivery_Status'] == 'Late Delivery'])}")
    print(f"  - On-time deliveries: {len(results_df[results_df['Delivery_Status'] == 'On-Time Delivery'])}")
    
    # ========================================================================
    # STEP 8: COMPUTE COUNTERFACTUAL ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: COMPUTING COUNTERFACTUAL ANALYSIS")
    print("="*70)
    
    # Sample late deliveries for counterfactual analysis
    late_indices = np.where(Y_hat > 0.5)[0]
    
    # Sample (e.g., 500 late deliveries for each intervention)
    sample_size = min(500, len(late_indices))
    sampled_indices = np.random.choice(late_indices, size=sample_size, replace=False)
    
    interventions = ['improve_efficiency', 'reduce_delay', 'optimize_logistics']
    
    counterfactual_results = []
    
    print(f"\nComputing counterfactuals for {len(sampled_indices)} late deliveries × {len(interventions)} interventions...")
    
    for intervention_type in interventions:
        print(f"\n  Processing intervention: {intervention_type}")
        for i in tqdm(sampled_indices, desc=f"  {intervention_type}", ncols=80):
            cf = stoch_scm.compute_counterfactual(i, E, Q, P, L, G, intervention_type)
            
            counterfactual_results.append({
                'Order_ID': df.iloc[i]['Order Id'],
                'Intervention_Type': intervention_type,
                'Original_Risk': cf['original_risk'],
                'Counterfactual_Risk': cf['counterfactual_risk'],
                'Causal_Effect': cf['causal_effect'],
                'Risk_Reduced': 'Yes' if cf['causal_effect'] > 0 else 'No'
            })
    
    # Save counterfactual results
    counterfactual_df = pd.DataFrame(counterfactual_results)
    counterfactual_df.to_csv(counterfactual_csv, index=False)
    
    print(f"\n✓ Counterfactual results saved to: {counterfactual_csv}")
    print(f"  - Total counterfactual scenarios: {len(counterfactual_df)}")
    
    # ========================================================================
    
    # ========================================================================
    # STEP 9: CREATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Pareto chart (Late deliveries only)
    create_pareto_chart(results_df, 'scm17_pareto_chart.png')
    
    # 2. Ranked horizontal bar chart (Total values)
    create_ranked_horizontal_barchart(results_df, 'scm17_ranked_horizontal_barchart.png')
    
    # 3. Delivery status pie chart (NEW)
    create_delivery_status_pie_chart(results_df, 'scm17_delivery_status_pie.png')
    
    # 4. Counterfactual comparison (NEW)
    create_counterfactual_comparison(counterfactual_df, 'scm17_counterfactual_comparison.png')
    
    # 5. Global heatmap + matrix
    create_global_heatmap_and_matrix(results_df)

    # 6. Risk vs efficiency transparency
    create_risk_vs_efficiency_plot(results_df)

    # 7. Classification metrics
    compute_classification_metrics(results_df)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    print("\n📊 Generated Files:")
    print("\n  CSV Files:")
    print(f"    1. {output_csv} - Complete causal explanations (ALL deliveries)")
    print(f"    2. {counterfactual_csv} - Counterfactual analysis results")
    
    print("\n  Visualization Files:")
    print(f"    1. scm17_pareto_chart.png - Pareto analysis (Late deliveries only)")
    print(f"    2. scm17_ranked_horizontal_barchart.png - Ranked comparison (Total values)")
    print(f"    3. scm17_delivery_status_pie.png - Status distribution pie chart")
    print(f"    4. scm17_counterfactual_comparison.png - Before/After comparison")
    
    print("\n" + "="*70)
    print("✅ MODIFIED SCM ANALYSIS COMPLETE")
    print("="*70)
    
    return results_df, counterfactual_df


if __name__ == "__main__":
    INPUT_CSV = 'deatcn11_dataset.csv'
    OUTPUT_CSV = 'scm17_causal_results_complete.csv'
    COUNTERFACTUAL_CSV = 'scm17_counterfactual_analysis.csv'
    
    print("\n" + "="*70)
    print("SCM CAUSAL ANALYSIS - MODIFIED VERSION")
    print("="*70)
    print("\nModifications:")
    print("  ✓ Pareto chart: Uses ONLY late deliveries")
    print("  ✓ Ranked bar chart: Shows TOTAL values only")
    print("  ✓ NEW: Delivery status pie chart with percentages")
    print("  ✓ NEW: Counterfactual analysis CSV + visualization")
    print("\nOutputs:")
    print("  - 2 CSV files (causal + counterfactual)")
    print("  - 4 visualization files (PNG)")
    print("\n" + "="*70 + "\n")
    
    result, cf_result = main(INPUT_CSV, OUTPUT_CSV, COUNTERFACTUAL_CSV)
    
    if result is not None:
        print("\n✅ All analysis complete!")
        print(f"\n📁 Check your directory for:")
        print(f"   - {OUTPUT_CSV}")
        print(f"   - {COUNTERFACTUAL_CSV}")
        print(f"   - scm17_pareto_chart.png")
        print(f"   - scm17_ranked_horizontal_barchart.png")
        print(f"   - scm17_delivery_status_pie.png")
        print(f"   - scm17_counterfactual_comparison.png")
    else:
        print("\n⚠️  Analysis failed.")