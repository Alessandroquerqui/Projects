# Federated Learning on Heterogeneous Data: FedAvg vs FedProx

**Signal Processing for Machine Learning – exam project, Sapienza University of Rome (September 2026)**

**Question:** how much does FedAvg degrade when client data is not identically distributed (Non-IID), and is a known mitigation (FedProx) enough to fix it?

In federated learning many devices train a model locally and share only the updates with a central server, which averages them (FedAvg). This implicitly assumes that client data is fairly similar (IID). In reality each device generates different data, and this statistical heterogeneity is what the project stress-tests.

## Setup

- **Dataset:** MNIST Digit Dataset (10 classes, 28×28 images), 10,000 samples, stratified 80/20 train/test split, per-pixel z-score standardization.
- **Model:** softmax regression (multinomial logistic regression), trained with full-batch gradient descent.
- **Federation:** 5 clients, 100 communication rounds, 20 local epochs per round, local learning rate 0.5.
- **FedProx:** adds a proximal term `(μ/2)·‖w − w_global‖²` to each client's local objective, penalising local weights that drift away from the global model. `μ = 0` recovers plain FedAvg; here `μ = 0.3`.

Everything is implemented from scratch in MATLAB, including the softmax training loop and the Dirichlet/Gamma sampling (Marsaglia–Tsang), with no dependency on the Statistics or Machine Learning toolboxes.

## Three degrees of heterogeneity

| Scenario | How clients see the data |
|---|---|
| **IID** | Each client gets a random, representative sample of all 10 digits |
| **Extreme Non-IID** | Each client sees only 2 of the 10 digits: zero information about the other 8 |
| **Moderate Non-IID (Dirichlet, α=0.5)** | Each client sees all 10 digits, but in highly unbalanced proportions |

## Results

| Scenario | FedAvg | FedProx |
|---|---|---|
| IID | 0.754 | – |
| Extreme Non-IID | 0.695 | 0.691 |
| Moderate Non-IID (Dirichlet) | 0.739 | 0.744 |

(Centralized training on all data pooled together is used as an upper-bound reference.)

## Key finding

The interesting result is **not** that Non-IID data hurts, but that **FedProx only helps against the right kind of heterogeneity**:

- **Extreme case:** FedProx (0.691) is practically identical to FedAvg (0.695), and their confusion matrices are indistinguishable. The proximal term limits how far a client drifts from the global model, but it cannot inject information a client never had: 8 of 10 digits are simply missing locally.
- **Moderate (Dirichlet) case:** here the failure mode of FedAvg genuinely is client drift. FedProx gives a small but consistent accuracy gain (+0.005), and a clearer signal in the loss: FedAvg's test loss starts rising again after 20–30 rounds as drift accumulates, while FedProx stays monotonically decreasing.

So the severity of the degradation depends on the *type* of heterogeneity, not just its presence, and a mitigation has to match the actual cause of the problem.

## Limitations

Honest about scope: single seed (no averaging across runs), a single Dirichlet α, `μ` set heuristically, and a linear model, so the findings may not transfer directly to deep networks. Natural next steps are multiple seeds for variance, an α sweep to trace a heterogeneity → accuracy curve, a systematic search for `μ`, and a non-linear model.

## Files

- [MATLAB script](federated_learning.m)
- [Project Presentation](Federated_Learning_Presentation.pdf)

## Tools

MATLAB (implemented from scratch, no toolboxes for the ML or the sampling)
