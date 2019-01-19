---
layout: post
title:  "Teaching agents to paint inside their own dreams"
image: 
  path: "images/world_painter.gif"
  thumbnail: "images/world_painter.gif"
date:   2019-01-16
---

In this post, I talk about using [World Models] to train agents to paint with [real painting software][MyPaint]. This includes my thought process, approach, failures, and some future work in this direction I'm excited about.

# Introduction

[Style transfer][1] has always been my favorite deep learning algorithm, and after completing my [previous project][style-transfer-browser] on porting arbitrary style transfer to the browser, I started thinking more deeply about what constitutes *style* in art. (Please note that I have 0 experience in art theory or history and cannot tell you the difference between a Monet or Manet. I just find it fascinating to see neural networks generate pretty things.)

The style loss function introduced by [Gatys, et. al.][1] for neural style transfer determines style by measuring the correlations between intermediate convolutional layer activations of a trained VGG image classifier. Simply put, style is defined statistically by how often (or how seldom) a particular image feature occurs with another image feature.

\\[ \frac{1}{n^{2}} \\]

This has always struck me as texture transfer rather than true style transfer, and sure enough, the approach doesn't hold up well to certain human-defined "styles".

<blockquote class="twitter-tweet" data-conversation="none" data-lang="en"><p lang="en" dir="ltr">Learning that cubist paintings don&#39;t translate well using the statistical approach of <a href="https://twitter.com/hashtag/NeuralStyle?src=hash&amp;ref_src=twsrc%5Etfw">#NeuralStyle</a>. As features of the image are jumbled around when re-assembling them, there&#39;s no guarantee long straight lines emerge—and they don&#39;t.<br><br>Two examples, loss is low but quality average: <a href="https://t.co/pwSSEiwNAx">pic.twitter.com/pwSSEiwNAx</a></p>&mdash; Alex J. Champandard (@alexjc) <a href="https://twitter.com/alexjc/status/1055515329965801472?ref_src=twsrc%5Etfw">October 25, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

So what defines style?

I recently visited the [Hakone Open-Air Museum] and found myself fascinated by some "paintings" that used glass shards in lieu of paint.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/shard-painting.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Not exactly what I saw, but like this.</figcaption>
</figure>

The resulting portraits were beautiful, unique, and could never be replicated with brushstrokes alone. And all it took to achieve this new, unique style was one thing: changing the medium.

At this point, I started to get an inkling that perhaps the artistic medium had a large part in defining style. It wasn't until I saw a random Reddit post a few weeks later (on r/gaming of all places) that the idea solidified itself in my mind.

[The post][reddit-gaming] was quite amusing in itself. Somebody had (very painstakingly) created custom characters in Super Smash Bros. that looked like the characters from Family Guy.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/ssb-fg.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Pretty standard mildly amusing Reddit content</figcaption>
</figure>

However, there was a particular comment in the thread that caught my attention (and thousands of others).

<figure class="align-center">
  <a href="#"><img src="{{ '/images/reddit-comment.png' | absolute_url }}" alt=""></a>
</figure>

"All great art is a response to the limitations of the medium."

This comment summarized in the best way possible what I had been thinking of for the last few weeks. Style can be seen as something that emerges from constraints. The Family Guy characters created above have a distinct style from the actual animated characters, simply because they were constrained to the options the game's character creation process provides. The glass shard paintings have a distinct style from regular paintings since they were constrained to use glass shards stuck to a portrait. We can extend this idea to regular 2D brushstroke paintings. Impressionist paintings like the famous "Starry Night" are constrained to use small and thin brush strokes.

**Note**: I am not declaring this is all there is to artistic style, nor that it is better than the statistical style loss mentioned above. This is obviously untrue. I am merely suggesting that, hey, this *could* be one way of looking at it.
{: .notice--warning}

[style-transfer-browser]: /2018/12/20/porting-arbitrary-style-transfer-to-the-browser.html
[World Models]: https://worldmodels.github.io
[MyPaint]: http://mypaint.org
[Hakone Open-Air Museum]: https://www.japan-guide.com/e/e5208.html
[reddit-gaming]: https://www.reddit.com/r/gaming/comments/a5zwbs/was_this_worth_my_time_probably_not/

[1]: https://arxiv.org/abs/1508.06576
