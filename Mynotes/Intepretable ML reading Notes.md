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

### Section1