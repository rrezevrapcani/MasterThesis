# Synthetic Biomedical Text Generation for Relation Extraction

This repository contains the code for my master's thesis on **synthetic biomedical text generation for relation extraction** using the BioRED dataset.

The main focus is on whether adding generated examples to the real training data improves performance on real documents. 

The study is guided by two research questions:
**RQ1**: Can generated biomedical texts improve relation extraction when added to an existing annotated training dataset?
**RQ2**: Which of the investigated augmentation approaches provides the greatest benefit?

## Methods

Three approaches are explored:

- **Prompting:** Generate synthetic twins, reword existing abstracts, and use back-translation.
- **Activation Steering:** Modify language model activations during generation using steering vectors.
- **Supervised Fine-Tuning (SFT):** Fine-tune a language model using LoRA to generate biomedical texts from entity and relation information.

## Dataset

The project uses [BioRED](https://doi.org/10.1093/bib/bbac282), which contains 600 PubMed abstracts annotated with biomedical entities and relations.

The generated texts are processed into entity-pair examples using the original relation annotations.

## Evaluation

Generated datasets are evaluated through:

- **Data analysis:** Text statistics, label distributions, and embedding visualizations using PubMedBERT and PCA.
- **Downstream evaluation:** A hierarchical PubMedBERT classifier that first detects relations and then predicts their types.

The classifier is trained using different combinations of real and synthetic data and evaluated on the original BioRED test set.

## References

- [BioRED](https://doi.org/10.1093/bib/bbac282)
- [PubMedBERT](https://aclanthology.org/2021.acl-long.331/)
- [LoRA](https://arxiv.org/abs/2106.09685/)
