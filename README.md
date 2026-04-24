# Self-Pruning Neural Network

## Overview

This project implements a self-pruning neural network that learns to remove unnecessary weights during training. Instead of pruning after training, the model dynamically identifies and suppresses weak connections using learnable gate parameters.

---

## Key Idea

Each weight in the network is associated with a **learnable gate**:

* Gate value ∈ (0,1) using sigmoid
* Output = weight × gate
* If gate → 0 → weight is effectively removed (pruned)

To encourage pruning, an **L1 regularization term** is applied on the gate values.

---

## Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

* Classification Loss: Cross Entropy
* Sparsity Loss: Sum / Mean of gate values
* λ (lambda): Controls pruning strength

---

## Dataset

* CIFAR-10 dataset (10-class image classification)
* Loaded using torchvision

---

## Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 0.01   | XX%           | XX%          |
| 0.02   | XX%           | XX%          |
| 0.05   | XX%           | XX%          |

---

## Observations

* Lower λ → higher accuracy, lower sparsity
* Higher λ → higher sparsity, lower accuracy
* Demonstrates a clear **sparsity–accuracy trade-off**

---

## Gate Distribution

The histogram of gate values shows:

* Many values near 0 → pruned connections
* Some values away from 0 → important weights retained

---

## How to Run

1. Open the Colab notebook or run the Python script
2. Train the model for different λ values
3. Observe accuracy and sparsity
4. View the gate distribution plot

---

## Files Included

* `self_pruning_nn.py` → Model + training code
* `notebook.ipynb` → Colab implementation
* `report.md` → Explanation
* `gate_distribution.png` → Visualization

---

## Conclusion

The model successfully learns to prune itself during training using gate-based regularization. The experiment demonstrates how increasing λ increases sparsity while affecting model accuracy.

---

## Colab Notebook

https://colab.research.google.com/drive/1-wv6WxA2_I0y-OeMsPzoPNbxwniA_0f_?usp=sharing
---
