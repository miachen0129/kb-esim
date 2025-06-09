# kb-esim

This repository contains the code for my graduate thesis:  
**_Research on False Claim Identification Based on Improved Textual Entailment Recognition Model_**


## Project Overview

This research aims to improve the accuracy of false scientific claim detection based on Recognizing Textual Entailment (RTE) models, enhanced with knowledge graph integration.

### Key Components

The proposed model consists of two main components:

- **K-BERT**  
  A variant of BERT that integrates external knowledge graphs to enhance contextual understanding.  
  → *Reference: Liu, Weijie, et al. "K-BERT: Enabling language representation with knowledge graph." AAAI 2020.*

- **ESIM Model**  
  A classical architecture for textual entailment recognition.  
  → *Reference: Chen, Q. et al. "Enhanced LSTM for natural language inference." arXiv:1609.06038 (2016).*


## Research Contributions

This work focuses on:

- **Combining K-BERT and ESIM** to enhance performance on NLI tasks for scientific corpus.
- **Improving knowledge injection in BERT** by introducing a *knowledge selection layer* before integrating knowledge into BERT, aiming to boost semantic understanding in knowledge-intensive fields.
- **Applying the model to claim verification** following the pipeline, mainly improving the rationale selection and entailment inference steps.

