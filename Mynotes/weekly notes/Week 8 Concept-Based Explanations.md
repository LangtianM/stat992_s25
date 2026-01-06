> [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](zotero://select/items/@kimInterpretabilityFeatureAttribution2018)
> 2018; PMLR; Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres

**Concept activation vectors**: In the latent space of a deep neural network, train a classifer to distinguish positive samples and negative samples w.r.t. a concept, then the normal vector of the decison boundary becomes the CAV

**Conceptual sensitivity** of a model's predictions:
$$
S_{C, k, l}(x) = \nabla h_{k}(f_{l}(x))^{T}v_{C}^{l}
$$
where:
- $S_{C, k, l}(x)$ is the sensitivity of class $k$ to concept $C$ at layer $l$.
- $h_k\left(f_l(x)\right)$ is the logit output for class $k$.
- $v_C^l$ is the CAV vector.

A TCAV score is then calculated:

$$
T C A V_{Q, C, k, l}=\frac{\left|\left\{x \in X_k: S_{C, k, l}(x)>0\right\}\right|}{\left|X_k\right|}
$$

which represents the fraction of class $k$ examples influenced by concept $C$.


> [Understanding intermediate layers using linear classifier probes](zotero://select/items/@alainUnderstandingIntermediateLayers2017)
> 2017; Guillaume Alain, Yoshua Bengio

![[attachments/Pasted image 20250313144300.png]]
For any hidden layer $H_{k}$ 





