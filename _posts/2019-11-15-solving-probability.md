---
layout: post
title:  "Teaching a neural network to use a calculator"
image: 
  path: "images/rnst/banner.jpg"
  thumbnail: "images/rnst/banner.jpg"
  hide: true
date:   2019-11-01
excerpt: Teaching a neural network to solve simple probability problems step by step with an external symbolic solver.
---

A few months ago, DeepMind released [Mathematics Dataset][mathematics_dataset], a codebase for procedurally generating pairs of mathematics questions and answers, to serve as a benchmark for the ability of modern neural architectures to learn mathematical reasoning.

The data consists of a wide variety of categories, ranging from basic arithmetic to probability. Here's an example question-answer pair from the paper:

Question:
```
What is g(h(f(x))), where f(x) = 2x + 3, g(x) = 7x − 4, and h(x) = −5x − 8?
```

Answer:
```
−70x − 165
```

Both questions and answers are in the form of free-form text, making [seq2seq][seq2seq_paper] models a natural first step for solving this dataset. In fact, the [paper][mathematics_dataset_paper] released alongside the dataset includes baseline performance metrics for today's [state-of-the-art seq2seq][attention_paper] models, applied naively to the dataset.

For more details, I highly recommend reading the [accompanying paper][mathematics_dataset_paper].

### Intermediate steps

In this article, we focus on the dataset categories relating to probability: `swr_level_set` and `swr_p_sequence`.

`swr_level_set` contains questions for calculating the probability of obtaining certain counts of letters.

```
QUESTION: Two letters picked without replacement from {v: 3, q: 1, x: 1, o: 1}. What is prob of picking 1 x and 1 v?
ANSWER: 1/5
```

`swr_p_sequence` contains questions for calculating the probability of a particular sequence of letters.
```
QUESTION: Calculate_prob_of_sequence_ko_when_two_letters_picked_without_replacement_from_yyyykkoykkyoyyokyyy.
ANSWER: 5/114
```

With the baseline approach used in the paper, 

### The model/Using an external symbolic solver


### Results


### Analysis of Results

success and failure cases, especially extrapolation failure (broken clock right twice a day, no "understanding")

### Equivalent solutions

found using beam search

### Visualizing attention


### Examining the data distribution

almost feels like an imbalanced classification problem.
with external calc, it is reduced to a counting and copying problem.
this is all assuming that the only training data used is from the probability set. there might be external knowledge from using the other numerous categories that make these observations not really apply.

### Related Literature


### Conclusions and future work


### Acknowledgments


### Citation

If you found this work useful, please cite it as:

```
```

[^1]: Adversarial examples are inputs that are specially crafted by an attacker to trick a classifier into producing an incorrect label for that input. There is an entire field of research dedicated to adversarial attacks and defenses in deep learning literature.


[google_colab]: https://colab.research.google.com/
[mathematics_dataset]: https://github.com/deepmind/mathematics_dataset/
[mathematics_dataset_paper]: https://arxiv.org/abs/1904.01557
[seq2seq_paper]: https://arxiv.org/abs/1409.3215
[attention_paper]: https://arxiv.org/abs/1706.03762
