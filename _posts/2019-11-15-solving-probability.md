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

### Solving simple probability problems

In this article, we focus on the dataset categories relating to probability: `swr_level_set` and `swr_p_sequence`.

`swr_level_set` contains questions for calculating the probability of obtaining certain counts of letters.

```
QUESTION: Two letters picked without replacement from {v: 3, q: 1, x: 1, o: 1}. What is prob of picking 1 x and 1 v?
ANSWER: 1/5
```

`swr_p_sequence` contains questions for calculating the probability of a particular sequence of letters.
```
QUESTION: Calculate prob of sequence ko when two letters picked without replacement from yyyykkoykkyoyyokyyy.
ANSWER: 5/114
```

With the baseline approach used in DeepMind's paper, the model takes in the question as a *sequence* of characters, and tries to directly map that to another *sequence* of characters, representing the correct probability. A vanilla  [transformer][attention_paper] architecture does surprisingly well, with accuracies of ~0.77 and ~0.73 on the `swr_level_set` and `swr_p_sequence` test sets, respectively.

### Humans use intermediate steps to solve math problems

To solve the same problems, a human does not just take a look at the question and immediately spit out an answer. One must go through a series of reasoning and intermediate steps, similar to the following:

```
QUESTION: Calculate prob of sequence ko when two letters picked without replacement from yyyykkoykkyoyyokyyy.
*Count the total number of letters in yyyykkoykkyoyyokyyy* -> 19
*Count the number of specific letters needed* -> k: 5 counts, o: 3 counts
*Set up the equation for solving probability of the sequence ko* -> 5/19 * 3/18
*Solve the equation (manually or using a calculator)* -> 5/19 * 3/18 = 5/114
ANSWER: 5/114
```

This insight leads to an obvious question. Instead of training the network on question-answer pairs, can we use intermediate steps to provide a better signal for the model to learn from? Presumably, a network will find it easier to capture the structure between intermediate steps, rather than the more complex structure between a question and its answer.

### Humans can use calculators for tedious work

The final intermediate step above `5/19 * 3/18 = 5/114` involves a multiplication between two fractions. While this specific equation is fairly simple and can be solved manually by a human in a few seconds, we can easily imagine questions about probability to involve more complicated fractions e.g. `(23/543 * 34/2551 * 673/12043) * (25!) / (20! * 19!)`. In probability (and other fields), most of human "intelligence" is used for setting up the equations, not evaluating them. Calculators were invented for a reason.

While we can try forcing our neural networks to figure out how to work through tedious intermediate calculations on their own, we can make their task much simpler by instead giving them access to an external symbolic calculator. This way, the network can shift its focus on learning *how* to solve a problem, outsourcing tedious calculations elsewhere.

### Putting it all together

We modify the [generation code][mathematics_dataset] to procedurally generate question-intermediate step (IS) pairs, instead of the original question-answer pairs. A question-IS pair looks like the following:

Question:
```
Calculate prob of sequence jppx when four letters picked without replacement from {x: 3, j: 1, p: 5}. 
```

Intermediate steps:
```
x:3 j:1 p:5
3+1+5=9 
(1/9)*(5/8)*(4/7)*(3/6)=5/252
5/252
```

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

make no mistake, this is a toy problem

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
