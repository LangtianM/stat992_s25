[Week9Notes](file:///Users/malangtian/Documents/StudyMaterials/Courses/stat992_s25/notes/week9.pdf)

> [Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation](zotero://select/items/@tanDistillandCompareAuditingBlackBox2018)
> 2018; Association for Computing Machinery; Sarah Tan, Rich Caruana, Giles Hooker, Yin Lou

- **Distillation:** Treat the black-box model as a teacher and train a transparent mimic model (student) to reproduce its risk scores.
- **Comparison:** Train an outcome model using the same model class and features to predict the ground-truth outcome.
- **Compare:** Identify where these two models behave differently to reveal potential biases or flaws in the black-box model.
![[attachments/Pasted image 20250402171708.png]]

> [InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders](zotero://select/items/@simonInterPLMDiscoveringInterpretable2025)
> 2025; Elana Simon, James Zou

![[attachments/Pasted image 20250402190324.png]]
- Train a sparse autoencoder in the embeddng space to get higher-dimensional sparse embeddings.
- There are only few dimensions are activated in the high-dimensional embeddings, which makes it easier to be connected with certain biology features.
- The model may "discover" some new biology patterns.


> [Adaptive wavelet distillation from neural networks through interpretations](zotero://select/items/@haAdaptiveWaveletDistillation)
> Wooseok Ha, Chandan Singh, Francois Lanusse, Srigokul Upadhyayula, Bin Yu

![[attachments/Pasted image 20250402212952.png]]
Afttribute to the wavelet transformed data
