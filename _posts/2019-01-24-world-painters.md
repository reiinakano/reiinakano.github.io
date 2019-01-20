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

Today, neural networks have been used by artists, with great success, to generate 2D images that look like paintings (e.g. neural style transfer, GANs that [generate portraits sold in auctions for thousands of dollars][christies-sell]). Most of these networks are set up to directly generate each pixel of the output. While the results are generally good, the approach strikes me as odd, because artists don't create paintings by calculating pixels one by one, they create paintings by *painting*. 

While per-pixel calculation is in itself a constraint (in fact, I'd say the artifacts present in GAN outputs give them their own distinct *style*), if we wanted to have an agent replicate a piece of artwork (a painting), I imagine we'd get more realistic outputs by providing it with the same medium (a paintbrush).

It is with this mindset that I read [Ha, et. al.][2]'s work on [World Models]. It was not a big logical leap from here to the next step. To produce art in the way we described, an agent would need to interact with a particular constrained medium. This constrained medium could be represented by a world model.

As a proof of concept, I tried to see if an agent could learn to work with the constraints of using a paintbrush, purely by interacting with a world model of a [real painting program][MyPaint]. I chose this task as I was aware of [Ganin, et. al.][3]'s work with [SPIRAL], that had indeed shown a neural network learning to work with a paintbrush. It would be a good goal to replicate their experiment results using a world model approach.

<small>Note that the rest of this blog post assumes you have read and understood the excellent [World Models] article. I reuse most of the terminology from that article in this blog post.</small>
{: .notice--info}

# Naively throwing the World Models code at the task

The full code for World Models is available at [this repository][world-models-code]. The first thing I did was apply the code with the bare minimum modifications to run on my task. There were two options: I could train a world model on the environment then train the agent purely on the world model (as in the Doom task), or I could train a world model but still use the outputs from the real environment during agent training (as in the CarRacing task). Since I do all my training on a single free GPU from [Google Colaboratory], I opted to go for the Doom approach, as running the paint program during training would considerably slow things down. It turns out this choice would be crucial, as I would eventually need full gradients from the world model during agent training, something I would not have in the CarRacing approach.

For the paint program environment, I reused code from the [SPIRAL implementation by Taehoon Kim][SPIRAL-code]. The implementation provides a Gym environment wrapping [MyPaint] and maps actions to brushstrokes and applies them to a 64x64 canvas. The following table shows the environment's action space.

| Action Parameter | Description |
|------------------|-------------|
| Pressure | Two options: 0.5 or 0.8. Determines the pressure applied to the brush. |
| Size | Two options: 0.2 or 0.7. Determines the size of the brush. |
| Jump | Binary choice 0 or 1 to determine whether or not to lift the brush for a certain stroke. |
| Color | 3D integer vector from 0-255 determining the RGB color of the brush stroke. |
| Endpoint | 2D point determining where to end the brush stroke. |
| Control point | 2D point determining the trajectory of the brush stroke. View the [SPIRAL paper][3] for more details on how the stroke trajectory is calculated. |

<small>This action space simply mirrors that described in the [SPIRAL paper][3]. The notable exceptions are the two rather specific choices for Pressure (0.5 or 0.8) and Size (0.2 or 0.7). There is no special reason for this other than that the [SPIRAL implementation][SPIRAL-code] I used had the environment set this way.</small>
{: .notice}

As a standard sanity check, my first goal was to reproduce MNIST characters with my agent. For this purpose, I disregard any Color input into the painting environment and fix everything to black.

[World Models]: https://worldmodels.github.io
[MyPaint]: http://mypaint.org
[Hakone Open-Air Museum]: https://www.japan-guide.com/e/e5208.html
[reddit-gaming]: https://www.reddit.com/r/gaming/comments/a5zwbs/was_this_worth_my_time_probably_not/
[constraints]: https://www.fastcompany.com/3027379/the-psychology-of-limitations-how-and-why-constraints-can-make-you-more-creative
[Phil Hansen]: http://www.philinthecircle.com/
[christies-sell]: https://www.theverge.com/2018/10/25/18023266/ai-art-portrait-christies-obvious-sold
[SPIRAL]: https://deepmind.com/blog/learning-to-generate-images/
[world-models-code]: https://github.com/worldmodels/worldmodels.github.io
[Google Colaboratory]: https://colab.research.google.com/
[SPIRAL-code]: https://github.com/carpedm20/SPIRAL-tensorflow

[1]: https://arxiv.org/abs/1508.06576
[2]: https://arxiv.org/abs/1803.10122
[3]: https://arxiv.org/abs/1804.01118
