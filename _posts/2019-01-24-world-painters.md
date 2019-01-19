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

At this point, I started to think that perhaps the artistic medium had a large part in defining style. It wasn't until I saw a random Reddit post a few weeks later (on r/gaming of all places) that the idea solidified itself in my mind.

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

This comment summarized in the best way possible what I had been thinking of. Style can be seen as something that emerges from constraints. The Family Guy characters created above have a distinct style from the actual animated characters, simply because they were constrained to the options the game's character creation process provides. The glass shard paintings have a distinct style from regular paintings since they were constrained to use glass shards stuck to a portrait. We can extend this idea to regular 2D portraits. Oil paintings have different constraints from pencil sketches. Impressionist paintings, like the famous "Starry Night", have been constrained to use small and thin brush strokes.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/starry.jpeg' | absolute_url }}" alt=""></a>
  <figcaption>The Starry Night by Vincent van Gogh</figcaption>
</figure>

This idea of constraints being beneficial to creativity turns out to be one that has been explored before. Here is [an article][constraints] showing some nice examples of constraints resulting in unique works of art. One of my favorites is the story of [Phil Hansen], an artist who suffered a debilitating injury and ended up being unable to keep his hand from shaking. This prevented him from ever drawing again with his usual pointillist style. He eventually incorporated the squiggly lines into his artwork, resulting in his own unique style.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/bruce.jpeg' | absolute_url }}" alt=""></a>
  <figcaption>Painting of Bruce Lee by Phil Hansen. This was painted by <a href="https://www.youtube.com/watch?v=CbvSms-1yj4">dipping his forearms in paint and "striking" the canvas</a> - an appropriate constraint for a portrait of the martial artist.</figcaption>
</figure>

To be clear, I am not declaring this is all there is to style, nor am I saying it is better than the statistical style loss mentioned above. Artistic style is obviously much, much deeper than this. I am merely suggesting that this point of view *could* help us build neural networks with a more human-like understanding of style.

It is with this mindset that I read [Ha, et. al.][2]'s work on [World Models]. It was not a big logical leap from here to the next step. To produce art in the way we described, an agent would need to interact with a particular constrained medium. This constrained medium could be represented by a world model.

To see if the approach has promise, I tried to see if an agent could learn to work with the constraints of using a paintbrush, purely by interacting with a world model of a [real painting program][MyPaint]. I chose this task as I was aware of [Ganin, et. al.][3]'s work with [SPIRAL], that had indeed shown a neural network learning to work with a paintbrush. It would be a good goal to replicate their experiment results using a world model approach.

Note that the rest of this blog post assumes you have read and understood the excellent [World Models] article. I reuse most of the terminology from that article in this blog post.
{: .notice--info}

[style-transfer-browser]: /2018/12/20/porting-arbitrary-style-transfer-to-the-browser.html
[World Models]: https://worldmodels.github.io
[MyPaint]: http://mypaint.org
[Hakone Open-Air Museum]: https://www.japan-guide.com/e/e5208.html
[reddit-gaming]: https://www.reddit.com/r/gaming/comments/a5zwbs/was_this_worth_my_time_probably_not/
[constraints]: https://www.fastcompany.com/3027379/the-psychology-of-limitations-how-and-why-constraints-can-make-you-more-creative
[Phil Hansen]: http://www.philinthecircle.com/
[SPIRAL]: https://deepmind.com/blog/learning-to-generate-images/

[1]: https://arxiv.org/abs/1508.06576
[2]: https://arxiv.org/abs/1803.10122
[3]: https://arxiv.org/abs/1804.01118
