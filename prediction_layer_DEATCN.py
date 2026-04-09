"""
DEA-TCN PREDICTION MODULE - CORRECTED TO REPLICATE PAPER
=========================================================

CORRECTIONS APPLIED:
1. DMU = Supplier (not transaction) - aggregated supplier-level inputs/outputs
2. BIP-DEA formulation - efficiency from LP θ (not output/input ratio)
3. Supplier-wise time sequences - TCN learns per-supplier behavior patterns

This module performs:
1. Supplier-level DEA computation (aggregated DMUs)
2. TCN training on supplier-wise sequences
3. Outputs enriched CSV with predictions

OUTPUT: deatcn11_dataset.csv with columns:
- All original columns (except Late_delivery_risk)
- Supplier_Efficiency (from supplier-level DEA)
- Late_delivery_risk_prediction (from TCN on supplier sequences)

Features:
- Checkpoint support for DEA computation
- Checkpoint support for TCN training
- Progress tracking
- Automatic resume on interruption
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import linprog
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# CORRECTION 1: SUPPLIER-LEVEL DEA (DMU = Supplier, not Transaction)
# ============================================================================

class SupplierDEAModel:
    """
    Data Envelopment Analysis using BIP-DEA model
    CORRECTION: DMU = Supplier (aggregated), not individual transactions
    """
    
    def __init__(self):
        self.efficiency_scores = None
        self.supplier_efficiency_map = {}
    
    def aggregate_supplier_data(self, df):
        """
        Aggregate transaction-level data to supplier-level DMUs
        
        Args:
            df: Transaction-level dataframe
            
        Returns:
            supplier_df: Aggregated supplier-level dataframe
        """
        print("\n" + "="*70)
        print("AGGREGATING TRANSACTIONS TO SUPPLIER-LEVEL DMUs")
        print("="*70)
        
        # Use Product Name as supplier identifier (paper uses supplier ID)
        # In real implementation, use actual Supplier ID column
        supplier_groups = df.groupby('Product Name')
        
        # Aggregate inputs (resources consumed by supplier)
        print("\nAggregating inputs (resources consumed):")
        print("  - Total shipping days (sum of real shipping days)")
        print("  - Total scheduled days (sum of scheduled days)")
        print("  - Average cost proxy (mean of benefit per order)")
        
        supplier_data = []
        
        for supplier, group in tqdm(supplier_groups, desc="Processing suppliers"):
            # INPUTS (to minimize)
            total_shipping_days = group['Days for shipping (real)'].sum()
            total_scheduled_days = group['Days for shipment (scheduled)'].sum()
            avg_cost_proxy = -group['Benefit per order'].mean()  # Negative = cost
            
            # OUTPUTS (to maximize)
            on_time_rate = 1 - group['Late_delivery_risk'].mean()
            total_sales = group['Sales per customer'].sum()
            num_orders = len(group)
            
            supplier_data.append({
                'Supplier': supplier,
                'Input_ShippingDays': total_shipping_days,
                'Input_ScheduledDays': total_scheduled_days,
                'Input_Cost': avg_cost_proxy,
                'Output_OnTimeRate': on_time_rate,
                'Output_Sales': total_sales,
                'NumOrders': num_orders
            })
        
        supplier_df = pd.DataFrame(supplier_data)
        
        print(f"\n✓ Aggregated {len(df)} transactions to {len(supplier_df)} suppliers")
        print(f"  - Mean orders per supplier: {supplier_df['NumOrders'].mean():.1f}")
        print(f"  - Median orders per supplier: {supplier_df['NumOrders'].median():.1f}")
        
        return supplier_df
    
    def calculate_efficiency_bip(self, inputs, outputs, checkpoint_file='deatcn11_checkpoint.pkl', 
                                  checkpoint_interval=100):
        """
        CORRECTION 2: Calculate DEA efficiency using BIP-DEA formulation
        Paper uses: minimize θ subject to constraints
        (Not the output/input ratio approach)
        
        BIP-DEA Model (Input-oriented):
        minimize θ
        subject to:
          Σ λ_j * x_ij ≤ θ * x_i0  (for all inputs i)
          Σ λ_j * y_rj ≥ y_r0       (for all outputs r)
          Σ λ_j = 1                  (VRS constraint)
          λ_j ≥ 0, θ free
        
        Args:
            inputs: Input variables (n_suppliers x n_inputs)
            outputs: Output variables (n_suppliers x n_outputs)
            checkpoint_file: Path to checkpoint file
            checkpoint_interval: Save checkpoint every N suppliers
            
        Returns:
            efficiency_scores: Array of efficiency scores (θ values, 0 to 1)
        """
        n_suppliers = inputs.shape[0]
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        
        efficiency_scores = np.zeros(n_suppliers)
        start_idx = 0
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    if checkpoint['n_suppliers'] == n_suppliers:
                        efficiency_scores = checkpoint['efficiency_scores']
                        start_idx = checkpoint['last_completed_idx'] + 1
                        print(f"\n✓ Supplier DEA Checkpoint loaded: Resuming from supplier {start_idx}/{n_suppliers}")
                    else:
                        print(f"\n⚠️  Checkpoint size mismatch. Starting from scratch...")
                        start_idx = 0
            except Exception as e:
                print(f"\n⚠️  Could not load checkpoint: {e}")
                print("Starting from scratch...")
                start_idx = 0
        
        # Progress tracking
        if start_idx == 0:
            print(f"\nComputing BIP-DEA efficiency for {n_suppliers} suppliers...")
        else:
            print(f"\nResuming BIP-DEA computation for remaining {n_suppliers - start_idx} suppliers...")
        
        # Main BIP-DEA computation loop
        for i in tqdm(range(start_idx, n_suppliers), 
                     desc="BIP-DEA Calculation", 
                     ncols=100,
                     initial=start_idx, 
                     total=n_suppliers):
            try:
                # BIP-DEA formulation: minimize θ
                # Variables: [λ_1, ..., λ_n, θ]
                # Objective: minimize θ
                c = np.concatenate([
                    np.zeros(n_suppliers),  # λ weights (don't minimize)
                    [1]                     # θ (minimize this)
                ])
                
                # Input constraints: Σ λ_j * x_ij ≤ θ * x_i0
                # Rearranged: Σ λ_j * x_ij - θ * x_i0 ≤ 0
                A_ub_inputs = np.hstack([
                    inputs.T,                           # λ coefficients
                    -inputs[i].reshape(-1, 1)          # θ coefficient
                ])
                b_ub_inputs = np.zeros(n_inputs)
                
                # Output constraints: Σ λ_j * y_rj ≥ y_r0
                # Rearranged: -Σ λ_j * y_rj ≤ -y_r0
                A_ub_outputs = np.hstack([
                    -outputs.T,                         # λ coefficients
                    np.zeros((n_outputs, 1))           # θ doesn't appear
                ])
                b_ub_outputs = -outputs[i]
                
                A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
                b_ub = np.concatenate([b_ub_inputs, b_ub_outputs])
                
                # Convexity constraint (VRS): Σ λ_j = 1
                A_eq = np.zeros((1, c.shape[0]))
                A_eq[0, :n_suppliers] = 1
                b_eq = np.array([1])
                
                # Bounds: λ_j ≥ 0, θ free (can be < 1)
                bounds = [(0, None)] * n_suppliers + [(None, None)]
                
                # Solve linear program
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                               A_eq=A_eq, b_eq=b_eq, 
                               bounds=bounds, method='highs')
                
                if result.success:
                    # Extract θ (efficiency score)
                    theta = result.x[-1]
                    efficiency_scores[i] = theta
                else:
                    # If optimization fails, assign default score
                    efficiency_scores[i] = 0.5
                    
            except Exception as e:
                # If any error occurs, assign default score
                efficiency_scores[i] = 0.5
            
            # Save checkpoint at intervals
            if (i + 1) % checkpoint_interval == 0:
                checkpoint = {
                    'efficiency_scores': efficiency_scores,
                    'last_completed_idx': i,
                    'n_suppliers': n_suppliers,
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
        
        # Save final checkpoint
        checkpoint = {
            'efficiency_scores': efficiency_scores,
            'last_completed_idx': n_suppliers - 1,
            'n_suppliers': n_suppliers,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Remove checkpoint file after successful completion
        try:
            os.remove(checkpoint_file)
            print(f"\n✓ Supplier DEA computation complete. Checkpoint file removed.")
        except:
            pass
        
        # Clip efficiency scores to valid range [0, 1]
        self.efficiency_scores = np.clip(efficiency_scores, 0, 1)
        return self.efficiency_scores
    
    def map_efficiency_to_transactions(self, df, supplier_df):
        """
        Map supplier-level efficiency back to individual transactions
        
        Args:
            df: Original transaction-level dataframe
            supplier_df: Supplier-level dataframe with efficiency scores
            
        Returns:
            efficiency_array: Efficiency score for each transaction
        """
        print("\nMapping supplier efficiency to transactions...")
        
        # Create supplier -> efficiency mapping
        supplier_eff_map = dict(zip(supplier_df['Supplier'], 
                                    supplier_df['Efficiency']))
        
        # Map to each transaction
        efficiency_array = df['Product Name'].map(supplier_eff_map).fillna(0.5).values
        
        print(f"✓ Mapped efficiency for {len(efficiency_array)} transactions")
        
        return efficiency_array


# ============================================================================
# TCN ARCHITECTURE (Same as before)
# ============================================================================

class Chomp1d(nn.Module):
    """Removes padding from the end of sequences"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with:
    - Dilated causal convolutions
    - Residual connections
    - Weight normalization
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, 
                     stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacked Temporal Convolutional Network"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """TCN for Late Delivery Risk Prediction"""
    def __init__(self, input_size=6, num_channels=[32, 64, 64, 32], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]  # Take last timestep
        y = self.fc(y)
        y = self.sigmoid(y)
        return y.squeeze()


# ============================================================================
# CORRECTION 3: SUPPLIER-WISE DATASET (Not Global Stream)
# ============================================================================

class SupplierWiseDataset(Dataset):
    """
    CORRECTION 3: Create sequences per supplier, not global sequences
    Paper's approach: Learn supplier-specific behavior patterns
    
    Each sequence belongs to ONE supplier's transaction history
    """
    def __init__(self, df, features, labels, sequence_length=10, min_supplier_orders=10):
        """
        Args:
            df: Dataframe with 'Product Name' (supplier identifier)
            features: Feature array
            labels: Label array
            sequence_length: Length of each sequence
            min_supplier_orders: Minimum orders for supplier to be included
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        # Group by supplier
        supplier_groups = df.groupby('Product Name')
        
        print(f"\nCreating supplier-wise sequences (min {min_supplier_orders} orders)...")
        suppliers_used = 0
        sequences_created = 0
        
        for supplier, group_indices in tqdm(supplier_groups.groups.items(), 
                                            desc="Building supplier sequences"):
            indices = group_indices.tolist()
            
            # Skip suppliers with too few orders
            if len(indices) < sequence_length:
                continue
            
            suppliers_used += 1
            
            # Create sequences from this supplier's history
            for i in range(len(indices) - sequence_length + 1):
                seq_indices = indices[i:i+sequence_length]
                seq_features = features[seq_indices]
                seq_label = labels[seq_indices[-1]]  # Label of last transaction
                
                self.sequences.append(seq_features)
                self.labels.append(seq_label)
                sequences_created += 1
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
        print(f"✓ Created {sequences_created} sequences from {suppliers_used} suppliers")
        print(f"  - Avg sequences per supplier: {sequences_created/suppliers_used:.1f}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]
        
        x_tensor = torch.FloatTensor(x).T  # Transpose for TCN format
        y_tensor = torch.FloatTensor([y])
        
        return x_tensor, y_tensor


# ============================================================================
# TCN TRAINING WITH CHECKPOINTING (Same as before)
# ============================================================================

def train_tcn_model(model, train_loader, val_loader, epochs=50, lr=0.001, 
                   checkpoint_file='supplier_tcn_checkpoint.pth'):
    """Train TCN model with checkpoint support"""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    start_epoch = 0
    
    train_losses = []
    val_losses = []
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            patience_counter = checkpoint['patience_counter']
            print(f"\n✓ TCN Checkpoint loaded: Resuming from epoch {start_epoch}/{epochs}")
        except Exception as e:
            print(f"\n⚠️  Could not load checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
    
    print("\n" + "="*70)
    print("TRAINING TCN MODEL ON SUPPLIER SEQUENCES")
    print("="*70)
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(output.detach().cpu().numpy())
            train_labels.extend(target.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.squeeze())
                
                val_loss += loss.item()
                val_preds.extend(output.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate accuracies
        train_acc = accuracy_score(
            np.array(train_labels).flatten(), 
            (np.array(train_preds) > 0.5).astype(int)
        )
        val_acc = accuracy_score(
            np.array(val_labels).flatten(), 
            (np.array(val_preds) > 0.5).astype(int)
        )
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), 'deatcn11_supplier_tcn_best_model.pth')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'patience_counter': patience_counter,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, checkpoint_file)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('supplier_tcn_best_model.pth'))
    
    # Remove checkpoint file after successful training
    try:
        os.remove(checkpoint_file)
        print(f"\n✓ Training complete. Checkpoint file removed.")
    except:
        pass
    
    print("="*70)
    
    return model, train_losses, val_losses


def predict_with_supplier_tcn(model, df, features, sequence_length=10):
    """
    Generate predictions maintaining supplier-wise structure
    
    Args:
        model: Trained TCN model
        df: Original dataframe with supplier info
        features: Feature array
        sequence_length: Sequence length
        
    Returns:
        predictions: Prediction for each transaction
    """
    model.eval()
    predictions = np.full(len(df), 0.5)  # Default prediction
    
    # Group by supplier
    supplier_groups = df.groupby('Product Name')
    
    print("\nGenerating supplier-wise predictions...")
    
    with torch.no_grad():
        for supplier, group_indices in tqdm(supplier_groups.groups.items(),
                                            desc="Predicting per supplier"):
            indices = group_indices.tolist()
            
            if len(indices) < sequence_length:
                # Too few orders, use mean prediction
                continue
            
            # Generate predictions for this supplier
            for i in range(len(indices)):
                if i < sequence_length - 1:
                    # Not enough history, use default
                    continue
                
                # Get sequence
                seq_start = max(0, i - sequence_length + 1)
                seq_indices = indices[seq_start:i+1]
                
                if len(seq_indices) < sequence_length:
                    continue
                
                seq_features = features[seq_indices[-sequence_length:]]
                x_tensor = torch.FloatTensor(seq_features).T.unsqueeze(0).to(device)
                pred = model(x_tensor).cpu().numpy()
                
                predictions[indices[i]] = pred
    
    return predictions


# ============================================================================
# MAIN PIPELINE - CORRECTED
# ============================================================================

def main(dataset_path='DataCoSupplyChainDataset.csv', max_records=180510, 
         output_csv='deatcn11_dataset.csv'):
    """
    Main DEA-TCN prediction pipeline - CORRECTED VERSION
    
    CORRECTIONS:
    1. DMU = Supplier (aggregated from transactions)
    2. BIP-DEA formulation (minimize θ)
    3. Supplier-wise sequences (not global stream)
    """
    print("\n" + "="*70)
    print("DEA-TCN PREDICTION MODULE - CORRECTED VERSION")
    print("="*70)
    print("\nCORRECTIONS APPLIED:")
    print("  1. DMU = Supplier (not transaction)")
    print("  2. BIP-DEA formulation (efficiency from LP θ)")
    print("  3. Supplier-wise time sequences")
    print("="*70)
    print(f"\nDataset: {dataset_path}")
    print(f"Max Records: {max_records if max_records else 'All'}")
    print(f"Output: {output_csv}")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    try:
        df = pd.read_csv(dataset_path, encoding='latin-1')
        print(f"✓ Dataset loaded: {len(df)} total records")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {str(e)}")
        return None
    
    if max_records:
        df = df.head(max_records).copy()
        print(f"✓ Using first {len(df)} records")
    
    # Sort by supplier and date for proper sequencing
    if 'order date (DateOrders)' in df.columns:
        df = df.sort_values(['Product Name', 'order date (DateOrders)']).reset_index(drop=True)
        print("✓ Sorted by supplier and order date")
    else:
        df = df.sort_values('Product Name').reset_index(drop=True)
        print("✓ Sorted by supplier")
    
    # ========================================================================
    # STEP 2: SUPPLIER-LEVEL DEA (CORRECTION 1 & 2)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: SUPPLIER-LEVEL DEA COMPUTATION")
    print("="*70)
    
    dea_model = SupplierDEAModel()
    
    # Aggregate to supplier level
    supplier_df = dea_model.aggregate_supplier_data(df)
    
    # Prepare supplier-level inputs and outputs
    inputs = supplier_df[['Input_ShippingDays', 'Input_ScheduledDays', 'Input_Cost']].values
    outputs = supplier_df[['Output_OnTimeRate', 'Output_Sales']].values
    
    # Normalize inputs and outputs
    inputs = (inputs - inputs.min(axis=0)) / (inputs.max(axis=0) - inputs.min(axis=0) + 1e-10)
    outputs = (outputs - outputs.min(axis=0)) / (outputs.max(axis=0) - outputs.min(axis=0) + 1e-10)
    
    print(f"\n✓ Supplier-level features prepared:")
    print(f"  - Suppliers (DMUs): {len(supplier_df)}")
    print(f"  - Inputs: {inputs.shape[1]}")
    print(f"  - Outputs: {outputs.shape[1]}")
    
    # Compute BIP-DEA efficiency
    supplier_efficiency = dea_model.calculate_efficiency_bip(inputs, outputs)
    supplier_df['Efficiency'] = supplier_efficiency
    
    print(f"\n✓ Supplier Efficiency Statistics:")
    print(f"  - Mean: {supplier_efficiency.mean():.4f}")
    print(f"  - Std: {supplier_efficiency.std():.4f}")
    print(f"  - Min: {supplier_efficiency.min():.4f}")
    print(f"  - Max: {supplier_efficiency.max():.4f}")
    print(f"  - Efficient suppliers (≥0.95): {np.sum(supplier_efficiency >= 0.95)} "
          f"({np.sum(supplier_efficiency >= 0.95)/len(supplier_efficiency)*100:.1f}%)")
    
    # Map back to transactions
    E = dea_model.map_efficiency_to_transactions(df, supplier_df)
    
    # ========================================================================
    # STEP 3: EXTRACT TRANSACTION-LEVEL FEATURES FOR TCN
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING TRANSACTION FEATURES FOR TCN")
    print("="*70)
    
    print("\nExtracting causal mechanism variables:")
    
    # D_r: Execution Delay
    Dr = (df['Days for shipping (real)'] - df['Days for shipment (scheduled)']).fillna(0).clip(lower=0).values
    print(f"  ✓ Execution Delay (D_r): mean={Dr.mean():.2f} days")
    
    # Q: Demand Complexity
    Q = (df['Order Item Quantity'].fillna(1) * np.log1p(df['Sales'].fillna(100))).values
    print(f"  ✓ Demand Complexity (Q): mean={Q.mean():.2f}")
    
    # P: Pricing Pressure
    P = (df['Order Item Discount Rate'].fillna(0) * (1 - df['Order Item Profit Ratio'].fillna(0.1))).values
    print(f"  ✓ Pricing Pressure (P): mean={P.mean():.4f}")
    
    # L: Logistics Constraints
    shipping_mode_map = {'Standard Class': 4, 'Second Class': 3, 'First Class': 2, 'Same Day': 1}
    L = df['Shipping Mode'].map(shipping_mode_map).fillna(4).values + np.log1p(df['Order Item Quantity'].fillna(1)).values
    print(f"  ✓ Logistics Constraints (L): mean={L.mean():.2f}")
    
    # G: Geographic Friction
    market_map = {'US': 1, 'LATAM': 2, 'Europe': 3, 'Pacific Asia': 4, 'Africa': 5}
    G = df['Market'].map(market_map).fillna(1).values
    G = G + np.log1p(np.abs(df['Latitude'].fillna(0)) + np.abs(df['Longitude'].fillna(0))).values / 10
    print(f"  ✓ Geographic Friction (G): mean={G.mean():.2f}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    Dr_scaled = scaler.fit_transform(Dr.reshape(-1, 1)).flatten()
    Q_scaled = scaler.fit_transform(Q.reshape(-1, 1)).flatten()
    P_scaled = scaler.fit_transform(P.reshape(-1, 1)).flatten()
    L_scaled = scaler.fit_transform(L.reshape(-1, 1)).flatten()
    G_scaled = scaler.fit_transform(G.reshape(-1, 1)).flatten()
    
    print("✓ Features standardized")
    
    # Ground truth
    y_true = df['Late_delivery_risk'].fillna(0).astype(int).values
    print(f"\n✓ Ground truth loaded:")
    print(f"  - Late deliveries: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.1f}%)")
    
    # Combine features
    features = np.column_stack([E, Dr_scaled, Q_scaled, P_scaled, L_scaled, G_scaled])
    
    # ========================================================================
    # STEP 4: CREATE SUPPLIER-WISE SEQUENCES (CORRECTION 3)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: CREATING SUPPLIER-WISE SEQUENCES")
    print("="*70)
    
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 32
    
    # Split by suppliers (not random split)
    unique_suppliers = df['Product Name'].unique()
    n_train_suppliers = int(0.8 * len(unique_suppliers))
    train_suppliers = set(unique_suppliers[:n_train_suppliers])
    
    train_mask = df['Product Name'].isin(train_suppliers)
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)
    
    features_train = features[train_mask]
    features_test = features[~train_mask]
    y_train = y_true[train_mask]
    y_test = y_true[~train_mask]
    
    print(f"\n✓ Data split by suppliers:")
    print(f"  - Train: {len(df_train)} transactions from {len(train_suppliers)} suppliers")
    print(f"  - Test: {len(df_test)} transactions from {len(unique_suppliers) - len(train_suppliers)} suppliers")
    
    # Create supplier-wise datasets
    train_dataset = SupplierWiseDataset(df_train, features_train, y_train, SEQUENCE_LENGTH)
    test_dataset = SupplierWiseDataset(df_test, features_test, y_test, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n✓ Supplier-wise datasets created:")
    print(f"  - Train sequences: {len(train_dataset)}")
    print(f"  - Test sequences: {len(test_dataset)}")
    
    # ========================================================================
    # STEP 5: TRAIN TCN MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: TRAINING TCN MODEL")
    print("="*70)
    
    # Initialize TCN
    tcn_model = TCNModel(
        input_size=6,
        num_channels=[32, 64, 64, 32],
        kernel_size=3,
        dropout=0.2
    ).to(device)
    
    print(f"\n✓ TCN model initialized on {device}")
    
    # Train model
    tcn_model, train_losses, val_losses = train_tcn_model(
        tcn_model, 
        train_loader, 
        test_loader,
        epochs=50,
        lr=0.001
    )
    
    # ========================================================================
    # STEP 6: GENERATE PREDICTIONS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: GENERATING PREDICTIONS")
    print("="*70)
    
    Y_hat = predict_with_supplier_tcn(tcn_model, df, features, SEQUENCE_LENGTH)
    
    print(f"\n✓ Predictions generated:")
    print(f"  - Mean risk: {Y_hat.mean():.4f}")
    print(f"  - Predicted late (>0.5): {np.sum(Y_hat > 0.5)} ({np.sum(Y_hat > 0.5)/len(Y_hat)*100:.1f}%)")
    
    # ========================================================================
    # STEP 7: EVALUATE MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    y_pred_binary = (Y_hat > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, Y_hat)
    except:
        auc = 0.5
    
    print("\n" + "-"*70)
    print("CORRECTED TCN MODEL PERFORMANCE")
    print("-"*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("-"*70)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'corrections_applied': {
            'dmu_level': 'supplier',
            'dea_formulation': 'BIP-DEA',
            'sequence_structure': 'supplier-wise'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('deatcn11_metrics_corrected.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✓ Metrics saved to: deatcn11_metrics_corrected.json")
    
    # ========================================================================
    # STEP 8: CREATE ENRICHED DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: CREATING ENRICHED DATASET")
    print("="*70)
    
    enriched_df = df.drop(columns=['Late_delivery_risk'], errors='ignore').copy()
    enriched_df['Supplier_Efficiency'] = E
    enriched_df['Late_delivery_risk_prediction'] = Y_hat
    
    enriched_df.to_csv(output_csv, index=False)
    print(f"\n✓ Enriched dataset saved to: {output_csv}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("CORRECTED PIPELINE COMPLETE")
    print("="*70)
    
    print("\nKey Differences from Original:")
    print("  1. ✓ DMU = Supplier (aggregated from transactions)")
    print("  2. ✓ BIP-DEA formulation (minimize θ, not ratio)")
    print("  3. ✓ Supplier-wise sequences (learn supplier patterns)")
    
    print("\nGenerated Files:")
    print(f"  1. {output_csv}")
    print(f"  2. supplier_tcn_best_model.pth")
    print(f"  3. deatcn11_metrics_corrected.json")
    
    print("\n" + "="*70)
    print("✅ CORRECTED DEA-TCN COMPLETE")
    print("="*70)
    
    return enriched_df


if __name__ == "__main__":
    DATASET_PATH = 'DataCoSupplyChainDataset.csv'
    MAX_RECORDS = 180510  # Use full dataset
    OUTPUT_CSV = 'deatcn11_dataset.csv'
    
    print("\n" + "="*70)
    print("DEA-TCN PREDICTION - CORRECTED CONFIGURATION")
    print("="*70)
    print("\nCORRECTIONS:")
    print("  1. DMU = Supplier (not transaction)")
    print("  2. BIP-DEA formulation (minimize θ)")
    print("  3. Supplier-wise sequences")
    print("\n" + "="*70 + "\n")
    
    result = main(DATASET_PATH, MAX_RECORDS, OUTPUT_CSV)
    
    if result is not None:
        print("\n✅ Corrected processing complete!")
    else:
        print("\n⚠️  Processing failed.")