# Explainable ML Course Materials & Projects
This repository contains my course materials and projects for the Explainable Machine Learning course at UW-Madison taught by Prof. Kris Sankaran. It is forked from the [official course repository](https://github.com/krisrs1128/stat992_s25)


## Projects: Explainability of LightGCN-based Recommender Systems

- [**`LightGCN+Integrated_Gradients`**](https://github.com/LangtianM/stat992_s25/tree/main/LightGCN%2BIntegrated_Gradients):  
  This project applies Integrated Gradients to a feature-augmented LightGCN trained on the MovieLens 1M dataset to attribute recommendation scores to user demographics and movie genres. It demonstrates how gradient-based attribution can explain individual recommendations while also revealing limitations related to feature correlation and non-causal interpretations.  
  - Report: [**`stat992hw1.pdf`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BIntegrated_Gradients/stat992hw1.pdf)
  - Experiments: [**`Explain_Recommender.ipynb`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BIntegrated_Gradients/Explain_Recommender.ipynb)

- [**`LightGCN+CAV`**](https://github.com/LangtianM/stat992_s25/tree/main/LightGCN%2BCAV):  
  This project explores Concept Activation Vectors (CAVs) for concept-level interpretability in embedding-based LightGCN recommenders. Movie genres are treated as semantic concepts, enabling efficient analysis of genre geometry in the latent space and user-specific conceptual sensitivity, and further supporting concept-driven customization of recommendations through interpretable embedding manipulation.  
  - Reports: [**`stat992hw2.pdf`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/stat992hw2.pdf), [**`stat992hw3.pdf`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/stat992_hw3.pdf)
  - Experiments: [**`Explain_LightGCN.ipynb`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/Explain_LightGCN.ipynb), [**`hw3_explore.ipynb`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/hw3_explore.ipynb)
  - Model&Method: [**`LightGCN.py`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/LightGCN.py), [**`RecommenderCAV.py`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/RecommenderCAV.py), [**`MovieDataProcessor.py`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/MovieDataProcessor.py), [**`RecommenderTrainer.py`**](https://github.com/LangtianM/stat992_s25/blob/main/LightGCN%2BCAV/RecommenderTrainer.py)

## Course Materials
- **`Mynotes`**: Personal weekly notes and reflections on course topics
- **`notes`**, **`demos`**, **`exercises`**, **`logistics`**: Official course materials including lecture notes, code demonstrations, assignments, and syllabus




