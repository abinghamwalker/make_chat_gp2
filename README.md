# Building an LLM from Scratch: A GPT-2 Implementation

This repository contains a step-by-step implementation of a Large Language Model based on GPT-2 architecture, following Sebastian Raschka's book "Build a Large Language Model (From Scratch)" and foundational papers like "Attention Is All You Need" (2017) by Vaswani et al.

## Project Overview

This project aims to:

- Implement a GPT-2 model from the ground up with detailed explanation of each step required to do this.
- Load publicly available weights and biases for GPT-2
- I originally intended to make a model that was small enough to run on a standard Macbook Air, this did not turn out to be the case.
- Allow for reasonable parameter variations from the original OpenAI implementation

## Implementation Details

The repository contains extensively documented Jupyter notebooks that:

1. Explain the fundamental concepts of transformer-based language models
2. Break down the key components (attention mechanisms, feed-forward networks, etc.)
3. Demonstrate how these components integrate to form a complete model
4. Show how to load pre-trained weights from OpenAI's GPT-2 release

## Documentation Approach

The code is accompanied by comprehensive markdown explanations that:

- Connect implementation details to theoretical concepts
- Highlight critical design decisions and their rationales
- Provide mathematical context for the algorithms
- Compare implementation choices with those in the original papers
- Discuss computational optimizations and their tradeoffs

## Learning Focus

While this repository draws heavily on others, I hope to be able to include personal modifications and additional explanations where they enhance understanding. The code and explanations serve as a personal reference but are structured to be accessible to others interested in understanding LLM architecture.

## Acknowledgements

This work stands on the shoulders of:

- Sebastian Raschka's "Build a Large Language Model (From Scratch)"
- Building Large Language Models from Scratch a Vizuara YouTube series
- The foundational paper "Attention Is All You Need" by Vaswani et al.
- OpenAI's GPT-2 implementation and released model weights

## Disclaimer

This repository is primarily a personal learning project. While others may find it useful as a reference or educational resource, it is not intended as a production-ready implementation.

---

_Note: This README will be updated as the implementation progresses._
