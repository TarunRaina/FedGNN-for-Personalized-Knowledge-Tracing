# FedGKT

## Federated Graph-based Knowledge Tracing with Personalised Prerequisite-aware Graph Attention Networks

---

## 📌 The Problem

Knowledge Tracing — predicting a student's future performance from their interaction history — is the engine behind every adaptive learning platform. Yet every widely deployed approach shares the same fundamental flaws:

- **Flat Sequence Modelling:** DKT, SAKT, AKT, and transformers treat concepts as interchangeable tokens in a stream. They have no understanding that fractions are a prerequisite for algebra, or that linear equations precede quadratic equations. The curriculum structure — already annotated by experts — is completely invisible to these models.

- **Opaque Knowledge States:** These models compress a student's entire knowledge into a single hidden vector. You cannot inspect the model and determine which specific concepts a student has mastered, which are partially understood, or which unmastered prerequisites are blocking their progress.

- **Privacy Violation:** Every existing system trains centrally — all student interaction logs are pooled on a server. This directly conflicts with FERPA, GDPR, and India's DPDP Act. Schools routinely transmit complete learning histories to third-party EdTech companies, creating significant regulatory and trust risks.

---

## 🔍 The Gap

| Approach | Prerequisite Structure? | Personalised per Student? | Federated (Privacy-Preserving)? |
|----------|------------------------|---------------------------|--------------------------------|
| DKT / SAKT / AKT | ❌ No | ❌ No | ❌ No |
| GKT (Nakagawa et al., 2019) | ✅ Yes (global graph) | ⚠️ Partial (shared states) | ❌ No |
| FDKT (Wu et al., 2021) | ❌ No | ❌ No | ✅ Yes |
| FedStar (Tan et al., AAAI 2023) | ✅ Yes (general graphs) | ✅ Yes | ✅ Yes |
| **FedGKT (Ours)** | **✅ Yes** | **✅ Yes** | **✅ Yes** |

**The missing piece:** No existing system combines:
1. Explicit prerequisite-aware modelling
2. Personalised per-student knowledge state representation
3. Federated training that preserves student privacy

FedStar (AAAI 2023) proves the split architecture principle works, but it addresses cross-domain graph classification — not educational knowledge tracing with prerequisite graphs. FedGKT applies the same principle specifically to the knowledge tracing domain.

---

## 💡 Our Solution — FedGKT

We replace the sequence representation with a **Personal Knowledge Graph (PKG)** for every student. Each student owns a graph `G_s = (V, E, X_s)` where:

- `V` = shared concept vocabulary (722 concepts from Junyi Academy)
- `E` = shared prerequisite edges (1,401 expert-annotated dependencies)
- `X_s` = student-specific node features (7 mastery indicators per concept)

A **Graph Attention Network (GAT)** processes this personal graph to predict the probability of correctly answering the next question on any concept. The attention mechanism learns which prerequisite concepts matter most for each target concept — providing interpretable prerequisite importance weights as a byproduct.

**Training is federated.** Each student is a client. Their PKG and interaction history never leave their local machine. Only trained GNN weights are shared with the aggregation server. The server applies FedAvg (or FedProx for non-IID students) and broadcasts updated weights for the next round.

**Personalisation is achieved via FedPer.** The GAT base layers are shared and aggregated (learning general prerequisite reasoning), while the output MLP is personal per student (adapting to their specific learning patterns). Only the GAT base layers are transmitted each round.

---

## 🎯 Key Innovations

**1. Personal Knowledge Graphs (PKGs)**
- 7 node features per concept: mastery_score, attempt_count, recent_accuracy_5, recent_accuracy_10, time_decay, streak, first_attempt
- Updated in real-time after every interaction
- Captures both long-term learning trajectory and short-term recent performance

**2. Prerequisite-aware GAT**
- 3-layer multi-head GAT
- Attention-weighted aggregation over prerequisite neighbours
- Learns which prerequisites actually matter for predicting performance on each concept

**3. Federated Training Protocol**
- 50 simulated clients (picked from Junyi dataset)
- Flower framework with virtual client simulation
- FedAvg weighted by student interaction count
- 100 communication rounds, 5 local epochs per round

**4. FedProx for Non-IID Robustness**
- Proximal regularisation term `(μ/2) * ||θ_local - θ_global||²`
- Prevents local model drift when student populations are highly heterogeneous
- Tested under Dirichlet partitions (α = 1000, 1.0, 0.1)

**5. FedPer Personalisation**
- Shared base: GAT Layers 1–2 (global prerequisite reasoning)
- Personal head: GAT Layer 3 + Output MLP (student-specific prediction calibration)
- Personal head weights never transmitted or aggregated

**6. Interpretability & Privacy**
- Attention heatmaps reveal prerequisite importance per concept
- Membership Inference Attack (MIA) evaluation quantifies privacy leakage
- Federation empirically reduces MIA AUC compared to centralised training

---

## 🔬 Technical Approach (High-Level)

### Dataset

**Junyi Academy (Primary)**
- 722 concepts, 1,401 expert-annotated prerequisite edges
- 247,000 students, 25 million interactions
- Downloaded via `EduData` or from the USTC mirror

### Model Architecture

- **Input:** PKG `G_s` — 722 nodes × 7 features
- **Layer 1:** GAT, 4 heads, output dim 128 per node
- **Layer 2:** GAT, 4 heads, output dim 128 per node
- **Layer 3:** GAT, 1 head (averaging), output dim 64 per node
- **Output MLP (personal):** Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1) → Sigmoid
- **Total parameters:** ~85,000 (lightweight for FL communication)

### Federated Protocol

- **Clients:** 50 simulated students
- **Rounds:** 100
- **Clients selected per round:** 10 (random subset)
- **Local epochs per round:** E=5
- **Aggregation:** FedAvg (weighted by interaction count per student)
- **Communication:** ~340 KB per round

### Ablation Studies

| # | What is tested | Why |
|---|----------------|-----|
| 1 | Expert graph vs random vs no edges | Proves prerequisite graph matters |
| 2 | All 7 features vs subset | Identifies contribution of each feature |
| 3 | GAT vs GCN vs GraphSAGE | Justifies attention mechanism |
| 4 | FedPer vs FedAvg | Shows personalisation improves individual AUC |
| 5 | FedProx vs FedAvg under non-IID | Demonstrates robustness to heterogeneity |
| 6 | Expert vs co-occurrence learned vs random | Answers: is human curation better? |
| 7 | Remove 10%/30%/50% of edges | Tests graph completeness dependency |

### Baselines

**Centralised:** DKT, SAKT, AKT, GKT (predecessor)
**Federated:** FDKT, FedAvg-DKT, FedAvg-GNN (no personalisation)
**Full system:** FedGKT (PKG + GAT + FedAvg/FedProx + FedPer)

### Evaluation Metrics

- **AUC-ROC** (primary) — macro-averaged across test students
- **Per-concept AUC** — identifies which concepts benefit most from graph structure
- **FL Convergence Rounds** — communication efficiency
- **Communication Cost (MB/round)** — compare FedPer vs FedAvg
- **MIA AUC** — privacy leakage (centralised vs federated)

---

## 📊 Expected Outcomes

- FedGKT AUC beats centralised sequence models (DKT, SAKT, AKT) on Junyi
- FedProx outperforms FedAvg under severe non-IID (α=0.1)
- FedPer improves per-student AUC over vanilla FL
- MIA shows federation reduces privacy leakage (target: 0.65 → 0.55)
- Attention heatmaps provide interpretable prerequisite importance
- Interactive dashboard visualises personal knowledge graphs and learning gaps

---

## 📚 Literature Foundation

FedGKT is built on 20 core papers across:
- **Knowledge Tracing Core:** DKT, DKVMN, SAKT, AKT, SAINT+
- **Graph-based KT:** GKT, HGKT, PSI-KT
- **KT Surveys & Tools:** Liu et al. 2021, pyKT 2022
- **GNN Architecture:** GAT (Velickovic et al., ICLR 2018)
- **Federated Learning:** FedAvg, FedProx, Kairouz survey, Flower
- **FL in Education:** Liu et al. 2025, Chu et al. 2022 (CIKM, arXiv)
- **FL + GNN:** FedStar (Tan et al., AAAI 2023)

---

