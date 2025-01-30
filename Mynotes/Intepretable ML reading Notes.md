# Convext and Motivation
[Towards A Rigorous Science of Interpretable Machine Learning](zotero://select/items/@doshi-velezRigorousScienceInterpretable2017)
2017; Finale Doshi-Velez, Been Kim
- Fairness
- Safety
- Misinformation

What makes model interpretable?
- Sparsity
- Simulatability: Can you make the prediction "by hand"?
- Modularity: Can the model be broken down?

# Saliency Maps
## Adventurous Begining
Consider a classifier
$$
f: X \to m \in\mathbb{R}^{c}
$$
where $X$ is an image and $f$ is the model, $m$ is the output while $c$ represents the number of classes.
General Salency Methods:
Gradient
$$
\frac{ \partial f }{ \partial x_{i} } 
$$
Drawbacks: large in put to a logistic function gives a nearly 0 gradient

## Reflections + Progress
Qualititive to quantitive
- Model Randomization
- Data Randomization


## Axiomatic attribution
[Axiomatic Attribution for Deep Networks](zotero://select/items/@sundararajanAxiomaticAttributionDeep2017)
2017; PMLR; Mukund Sundararajan, Ankur Taly, Qiqi Yan
### Two Fundamental Axioms
- Sensitivity
	For every input and baseline that differ in one feature but have different predictions then the differing feature, then the differing feature should be given a non-zero attribution
- Implementation Invariance
	The attributions are always identical for two functionally equivalent networks (functionally equivalent networks may have different implementation)

### Integrated Gradients
Let $F:\mathbb{R}^{n}\to [0, 1]$ represent a deep NN, $x \in\mathbb{R}^{n}$ be the input and $x'\in\mathbb{R}^{n}$ be the baseline input. 
**Integrated Gradient**: path integral of the gradients along the straightline path from the baseline $x'$ to the input $x$.
$$
\\text{IG}_{i}(x):=(x-x_{i}')\int _{\alpha=0}^{1} \frac{ \partial F(x'+\alpha(x-x')) }{ \partial x_{i} }  \, d\alpha 
$$
Property: **Completeness**: the attributions add up to the difference between the $F(x)$ and $F(x')$.
### Uniqueness of Integrated Gradients
- Path methods are the only methods to satisfy the axioms.
- Integrated gradients is symmetry-preserving 
## Sanity Checks for Saliency Maps
[Sanity Checks for Saliency Maps](zotero://select/items/@adebayoSanityChecksSaliency2020)
2020; Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, Been Kim

###  Model parametxer randomization test.
Does the saliency method depend on the model?
	Trained model v.s. randomly initialized model

> Why some methods are invariant under randomized reparameterization?
#### Cascading Randomization
  Randomize the weights of a model starting from the top layer all the way to the bottom layers.
  ![[attachments/Pasted image 20250129184516.png]]
> Guided BP is just insensitive to the top layers in this exmaple. Does this **really** tells us it's a bad method ?
>Can the model do the same task using the frist few layers only ?

![[attachments/Pasted image 20250129185247.png]]

  
#### Data randomization test
the saliency method depends on the trianing data?
	Training model with labeled data v.s.  with randomly labeled data
![[attachments/Pasted image 20250129190932.png]]
> What's the hypothesis being test ?

[Interpretable Explanations of Black Boxes by Meaningful Perturbation](zotero://select/items/@fongInterpretableExplanationsBlack2017)
2017; Ruth Fong, Andrea Vedaldi
Mask a part of image to see the influence on the classification results.
$$
\begin{aligned}
\min _{m \in[0,1]^{\Lambda}} \lambda_1\|\mathbf{1}-m\|_1 & +\lambda_2 \sum_{u \in \Lambda}\|\nabla m(u)\|_\beta^\beta \\
& +\mathbb{E}_\tau\left[f_c\left(\Phi\left(x_0(\cdot-\tau), m\right)\right)\right]
\end{aligned}
$$
-  Use as small mask as one can.
- Prefer simple mask.
- Predict on "deleted" image.