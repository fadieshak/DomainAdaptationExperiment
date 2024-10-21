# **Domain Adaptation Experiments**

## **Overview**

This project aims to explore different domain adaptation techniques by leveraging clustering models, knowledge distillation, and adversarial training. The core idea is to bridge the domain gap between a source and target domain by using pre-trained clustering models or knowledge distillation techniques to guide the model’s learning on the target domain.

## **Key Experiments**

1. **Clustering-Based Class-Cluster Projection**:  
   A clustering teacher model (e.g., DEC or N2D) is pre-trained on the target dataset. The output of the classification student model trained on the source domain is then projected to predict the clusters of target samples, effectively learning a mapping from source classes to target clusters.

2. **Adversarial Training**:  
   Additional experiments introduced Maximum Mean Discrepancy (MMD) loss and gradient reversal layers to align feature distributions from the source and target domains. These techniques, in combination with the previous technique, aim to further reduce domain divergence and improve model generalization on target data.

## **Results**

### **SVHN to MNIST Domain Adaptation**:

- N2D model as teacher: **89.53% accuracy**.
- N2D model as teacher + adversarial training with gradient reversal: **92.12% accuracy**.

### **USPS to MNIST Domain Adaptation**:

- DEC model as teacher: achieved **87.54% accuracy**.
- N2D model as teacher: achieved **95.34% accuracy**.
- Further boosted with adversarial training: **97.2% accuracy**.

The clustering teacher method showed improvements, particularly for tasks with a smaller domain gap (USPS to MNIST). However, results on more challenging tasks (SVHN to MNIST) suggest there is still room for improvement, especially when compared to state-of-the-art methods.

## **Requirements**

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## **Training Models**

You can use the provided training scripts to run different experiments:

- `train_project_DEC.py`: Train using DEC clustering teacher model.
- `train_project_N2D_mmd.py`: Train using N2D clustering teacher model with MMD loss.
- `train_project_N2D_revgrad.py`: Train using N2D clustering teacher model with adversarial training.

## **Evaluate Performance**

Use `calculate_acc.py` to evaluate the model’s performance on the target dataset.

## **KNN Experiments**

Run `knn_acc.py` to fit a KNN classifier on extracted features and evaluate domain adaptation effectiveness.

## **Acknowledgements**

This project drew inspiration from various works, including:

- **[ADDA (Adversarial Discriminative Domain Adaptation)](https://github.com/ayushtues/ADDA_pytorch)**.
- **[DEC (Unsupervised Deep Embedding for Clustering Analysis)](https://github.com/vlukiyanov/pt-dec)**.
- **[N2D ((Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding)](https://arxiv.org/pdf/1908.05968)**.
- **[CAT (Cluster Alignment with a Teacher for Unsupervised Domain Adaptation)](https://github.com/thudzj/CAT)**.

