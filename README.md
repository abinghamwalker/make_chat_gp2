# Building an LLM from Scratch: A GPT-2 Implementation

This repository contains a step-by-step implementation of a Large Language Model based on GPT-2 architecture, following Sebastian Raschka's book "Build a Large Language Model (From Scratch)" and foundational papers like "Attention Is All You Need" (2017) by Vaswani et al.

## Files to note in order are: -

- My_GPT.py : This is the python files of my GPT structure which draws heavy links with the credited source material, this was written later so I could easily import funcitions for other uses including running on Colab.
- My_GPT_construction: This is the first stages of constructing an LLM from tokenization through to the end of the transformer block.
- My_GPT_download : This is a download of the GPT-2 actual weights and then run through the same generation function written for my model.
- My_GPT_testing : This is an area that I have run tests of different attention heads, temperatures and top k values. I had wanted to test changing the entire structure to see how this effects the foundational model, such as the number attention heads, learning rate, decay or context lengths but with my available infrastucture this would be a massively time consuming task.
- My_GPT_fine_tuning: This is the area where I turn the foundational model into an acutally useable LLM. I have shown two use cases where it is used for classification where you would normally believe that to be a tasks of machine learninig and secondly into an LLM agent.

## Project Overview

This project aims to:

- Implement a GPT-2 model from the ground up with detailed explanation of each step required to do this.
- Load publicly available weights and biases for GPT-2
- Allow for reasonable parameter variations from the original OpenAI implementation and experimention where possible.

## Implementation Details

The repository contains Jupyter notebooks that:

1. Explain the fundamental concepts of transformer-based language models
2. Break down the key components (attention mechanisms, feed-forward networks, etc.)
3. Demonstrate how these components integrate to form a complete model
4. Show how to load pre-trained weights from OpenAI's GPT-2 release

## Documentation Approach

The code is accompanied by markdown explanations that:

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
