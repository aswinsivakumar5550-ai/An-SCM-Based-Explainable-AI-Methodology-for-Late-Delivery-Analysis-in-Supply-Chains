# Supply Chain Late Delivery Risk Explainability using Structural Causal Models (SCM)

## Overview

This repository presents an interpretable framework for analyzing **late delivery risk** in supply chain systems using a combination of **machine learning prediction** and **causal inference modeling**. The primary objective is not only to predict late deliveries but to explain *why* delays occur and identify operational factors responsible for delivery risk.

Traditional machine learning models often produce accurate predictions but provide limited interpretability, making them less useful for operational decision-making. This project addresses this limitation by integrating a **Structural Causal Model (SCM)** with a **DEA–TCN prediction framework**, enabling decomposition of late delivery risk into interpretable causal components.

The proposed framework identifies dominant operational drivers of delivery delays and evaluates hypothetical operational improvements using **counterfactual analysis**.

---

## Key Contribution (Main Novelty)

The main contribution of this work is the **SCM-based causal explanation layer**, which:

* Quantifies the causal influence of supply chain factors on delivery risk
* Identifies dominant causes of late delivery for each order
* Enables counterfactual reasoning for operational decision support
* Converts black-box predictions into interpretable causal insights

---

## Methodology

The framework consists of two major components:

### 1. Prediction Layer (DEA–TCN)

The prediction layer produces continuous late delivery risk probabilities.

#### Data Envelopment Analysis (DEA)

DEA estimates supplier efficiency scores using operational inputs and outputs.

Inputs:

* Actual shipping duration
* Scheduled shipping duration
* Cost proxy (profit-related indicator)

Outputs:

* On-time delivery rate
* Sales performance

Output:
Supplier Efficiency score between 0 and 1

---

#### Temporal Convolutional Network (TCN)

TCN models sequential supply chain patterns to predict late delivery risk probability.

Model characteristics:

* Dilated causal convolutions
* Residual connections
* Sequential feature learning
* Sigmoid output layer producing probability values

Output:
Late Delivery Risk probability between 0 and 1

---

### 2. Explainability Layer (Structural Causal Model)

The Structural Causal Model explains predictions through causal relationships among supply chain mechanisms.

Seven causal variables are modeled:

E — Supplier Efficiency
Q — Demand Complexity
P — Pricing Pressure
L — Logistics Constraints
G — Geographic Friction
D — Execution Delay (Mediator)
Y — Late Delivery Risk (Outcome)

Execution Delay functions as a mediator linking operational conditions to delivery outcomes.

---

## Causal Graph Structure

Relationships between variables are represented using a Directed Acyclic Graph (DAG).

Causal paths include:

Supplier Efficiency → Execution Delay
Demand Complexity → Execution Delay
Logistics Constraints → Execution Delay
Geographic Friction → Execution Delay

Execution Delay → Late Delivery Risk

Direct effects on Late Delivery Risk:

Supplier Efficiency
Demand Complexity
Pricing Pressure
Logistics Constraints
Geographic Friction

---

## Counterfactual Analysis

Counterfactual reasoning evaluates how hypothetical operational improvements influence predicted delivery risk.

Example interventions:

* Improving supplier efficiency
* Reducing execution delay
* Optimizing logistics constraints

Causal effect is computed as the difference between original predicted risk and counterfactual predicted risk.

---

## Repository Structure

```bash
project/
│
├── SupplychainRiskPredictionDataset.csv     # Main dataset
│
├── prediction_layer_DEATCN.py               # DEA–TCN prediction model
├── explainabe_layer_SCM.py                  # Structural causal explanation model
├── acyclic_dag_graph.py                     # DAG construction
├── validation_metrics.py                    # evaluation metrics for explanations
│
├── counterfactual_analysis.csv              # counterfactual intervention results
├── Dominant_cause_result.csv                # dominant cause per order
│
├── chart_DAG_Graph.png                      # causal graph visualization
├── chart_confusion_matrix.png               # prediction performance visualization
├── chart_pareto.png                         # dominant cause distribution
├── chart_pie.png                            # contribution distribution
├── chart_ranked_horizontal_barchart.png     # ranked mechanism importance
│
└── STRUCTURAL_CAUSAL_MODEL.docx             # detailed methodological description
```

---

## Description of Files

### Dataset

SupplychainRiskPredictionDataset.csv
Primary dataset containing transactional supply chain information used for prediction and causal analysis.

---

### Python Scripts

prediction_layer_DEATCN.py
Implements DEA–TCN hybrid model for supplier efficiency estimation and late delivery risk prediction.

explainabe_layer_SCM.py
Implements Structural Causal Model to estimate causal relationships and compute contribution scores.

acyclic_dag_graph.py
Defines Directed Acyclic Graph representing causal dependencies among supply chain variables.

validation_metrics.py
Computes causal evaluation metrics such as faithfulness, sparsity, stability, and domain alignment.

---

### Output Files

Dominant_cause_result.csv
Contains dominant causal mechanism responsible for predicted delivery risk for each order.

counterfactual_analysis.csv
Contains results of counterfactual simulations evaluating potential operational improvements.

---

### Visual Outputs

chart_DAG_Graph.png
Graphical representation of causal structure.

chart_confusion_matrix.png
Prediction performance summary.

chart_pareto.png
Pareto distribution of dominant causes.

chart_pie.png
Contribution share of causal mechanisms.

chart_ranked_horizontal_barchart.png
Ranked importance of causal factors.

---

## Workflow

1. Load supply chain dataset
2. Compute supplier efficiency using DEA
3. Predict late delivery probability using TCN
4. Construct Structural Causal Model
5. Estimate causal contributions of mechanisms
6. Identify dominant causes of delivery delays
7. Perform counterfactual intervention analysis
8. Evaluate explanation quality metrics

---

## System Requirements

Recommended hardware configuration for efficient execution:

Processor:
AMD Ryzen 7000 Series (Ryzen 7 or higher recommended)

Graphics:
NVIDIA RTX 4050 GPU (CUDA supported)

Memory:
Minimum 24 GB RAM
Recommended 46 GB RAM for faster training

Storage:
At least 5 GB free disk space

Operating System:
Windows 10 / Windows 11 / Linux / macOS

Python Version:
Python 3.9 or later

GPU acceleration significantly reduces training time for the Temporal Convolutional Network.

---

## Software Requirements

Required Python libraries:

numpy
pandas
scikit-learn
pytorch
matplotlib
networkx
scipy

Install dependencies:

```bash
pip install numpy pandas scikit-learn torch matplotlib networkx scipy
```

---

## Results

The framework identifies execution delay as a primary mediator linking operational factors to delivery risk.

Causal contribution analysis shows that logistics constraints, demand complexity, and supplier efficiency influence predicted late deliveries through both direct and mediated pathways.

Counterfactual simulations demonstrate how targeted improvements in logistics and supplier performance can reduce predicted delay probability.

---

## Applications

Supply chain risk management
Logistics optimization
Supplier performance evaluation
Decision support systems
Explainable artificial intelligence

---

## Future Work

Extension to nonlinear structural causal models
Real-time supply chain monitoring systems
Multi-class delivery risk prediction
Integration with SHAP-based explainability
Application to additional logistics datasets

---

If needed, I can also provide:

• requirements.txt
• architecture diagram description
• GitHub short description (160 characters)
• citation BibTeX
• keywords for repository tags
• project badges
• raw dataset
• portfolio-ready description
