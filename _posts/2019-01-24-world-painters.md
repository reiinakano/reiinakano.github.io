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

**Lately, I've been thinking about the role that constraints play in artistic style.**

A few months ago, I visited the [Hakone Open-Air Museum] and found myself fascinated by some "paintings" that used glass shards in lieu of paint.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/shard-painting.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Not exactly what I saw, but like this.</figcaption>
</figure>

The resulting portraits were beautiful, unique, and something that could never be replicated with brushstrokes alone. All it took to achieve this new, unique style was one thing: changing the medium.

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

This comment summarized in the best way possible what I had been thinking of. Style can be seen as something that emerges from constraints. The Family Guy characters created above have a distinct style from the actual animated characters, simply because they were constrained to the options the game's character creation process provides. The glass shard paintings have a distinct style from regular paintings since they were constrained to use glass shards stuck to a portrait. We can extend this idea to regular 2D portraits. Oil paintings are created with different constraints from pencil sketches. Impressionist paintings, like the famous "Starry Night", have been constrained to use small and thin brush strokes.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/starry.jpeg' | absolute_url }}" alt=""></a>
  <figcaption>The Starry Night by Vincent van Gogh</figcaption>
</figure>

This idea of constraints being beneficial to creativity turns out to be one that has been thoroughly explored. Here is [an article][constraints] showing some nice examples of constraints resulting in unique works of art. One of my favorites is the story of [Phil Hansen], an artist who suffered a debilitating injury and ended up being unable to keep his hand from shaking. This prevented him from ever drawing again with his usual pointillist style. He eventually incorporated the squiggly lines into his artwork, resulting in his own unique style.

<figure class="align-center">
  <a href="#"><img src="{{ '/images/bruce.jpeg' | absolute_url }}" alt=""></a>
  <figcaption>Painting of Bruce Lee by Phil Hansen. This was painted by <a href="https://www.youtube.com/watch?v=CbvSms-1yj4">dipping his forearms in paint and "striking" the canvas</a> - an appropriate constraint for a portrait of the martial artist.</figcaption>
</figure>

Today, neural networks have been used by artists, with great success, to generate 2D images that look like paintings (e.g. neural style transfer, GANs that generate portraits sold in auctions for thousands of dollars). Most of these networks are set up to directly generate each pixel of the output. While the results are generally good, this strikes me as odd, because artists don't create paintings by calculating pixels one by one, they create paintings by *painting*. 

While per-pixel calculation is in itself a constraint (in fact, I'd say the artifacts present in GAN outputs give them their own distinct *style*), if we wanted to have an agent replicate a piece of artwork (a painting), we'd get more realistic outputs by providing it with the same medium (a paintbrush).

It is with this mindset that I read [Ha, et. al.][2]'s work on [World Models]. It was not a big logical leap from here to the next step. To produce art in the way we described, an agent would need to interact with a particular constrained medium. This constrained medium could be represented by a world model.

As a proof of concept, I tried to see if an agent could learn to work with the constraints of using a paintbrush, purely by interacting with a world model of a [real painting program][MyPaint]. I chose this task as I was aware of [Ganin, et. al.][3]'s work with [SPIRAL], that had indeed shown a neural network learning to work with a paintbrush. It would be a good goal to replicate their experiment results using a world model approach.

Note that the rest of this blog post assumes you have read and understood the excellent [World Models] article. I reuse most of the terminology from that article in this blog post.
{: .notice--info}

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
