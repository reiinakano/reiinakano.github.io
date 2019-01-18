---
layout: post
title:  "Teaching agents to paint inside their own dreams"
image: 
  path: "images/world_painter.gif"
  thumbnail: "images/world_painter.gif"
date:   2019-01-16
---

In this post, I talk about using [World Models] to train agents to paint with [real painting software][MyPaint] - including my thought process, approach, failures, and future work in this direction I'm excited about.

# Introduction

[Style transfer][1] has always been my favorite deep learning algorithm, and after completing my [previous project][style-transfer-browser] on porting arbitrary style transfer to the browser, I started thinking more deeply about what constitutes *style* in art. (Please note that I have 0 experience in art theory or history and cannot tell you the difference between a Monet or Manet. I just find it fascinating to see neural networks generate pretty things.)

The style loss function introduced by [Gatys, et. al.][1], determines style similarity by comparing the correlations between intermediate convolutional layer activations of a trained VGG image classifier. Simply put, style is defined by how often (or how seldom) a particular image feature occurs with another image feature.

\\[ \frac{1}{n^{2}} \\]

This has always struck me as texture transfer rather than true style transfer, and sure enough, doesn't hold up to certain human-defined "styles".

<blockquote class="twitter-tweet" data-conversation="none" data-lang="en"><p lang="en" dir="ltr">Learning that cubist paintings don&#39;t translate well using the statistical approach of <a href="https://twitter.com/hashtag/NeuralStyle?src=hash&amp;ref_src=twsrc%5Etfw">#NeuralStyle</a>. As features of the image are jumbled around when re-assembling them, there&#39;s no guarantee long straight lines emerge—and they don&#39;t.<br><br>Two examples, loss is low but quality average: <a href="https://t.co/pwSSEiwNAx">pic.twitter.com/pwSSEiwNAx</a></p>&mdash; Alex J. Champandard (@alexjc) <a href="https://twitter.com/alexjc/status/1055515329965801472?ref_src=twsrc%5Etfw">October 25, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>





[style-transfer-browser]: /2018/12/20/porting-arbitrary-style-transfer-to-the-browser.html
[World Models]: https://worldmodels.github.io
[MyPaint]: http://mypaint.org

[1]: https://arxiv.org/abs/1508.06576
