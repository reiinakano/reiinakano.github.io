---
layout: post
title:  "Neural Style Transfer with Adversarially Robust Classifiers"
image: 
  path: "images/rnst/banner.jpg"
  thumbnail: "images/rnst/banner.jpg"
  hide: true
date:   2019-06-15
---

I recently read an intriguing paper by Engstrom, et. al. about a radically different way to view adversarial examples [^1], titled "[Adversarial Examples Are Not Bugs, They Are Features][not_bugs_features_arxiv]". 

The authors propose the existence of so-called "robust" and "non-robust" features in the images used for training image classifiers. 
*Robust features* can be thought of as features that humans naturally use for classification e.g. flappy ears are indicative of certain breeds of dogs, while black and white stripes are indicative of zebras.
*Non-robust features*, on the other hand, are features that humans aren't sensitive to [^2], but are truly indicative of a particular class (i.e. they correlate with the class over the *whole* (train and test) dataset).
The authors argue that adversarial examples are produced by replacing the non-robust features in an image with non-robust features of another class.

I highly recommend reading the [paper][not_bugs_features_arxiv] or at the very least, the accompanying [blog post][not_bugs_features_blog].

One figure in the paper that particularly struck me as interesting was the following graph showing the correlation between transferability of adversarial examples to the ability to learn similar non-robust features.

<figure class="align-center">
  <img src="{{ '/images/rnst/transferability.png' | absolute_url }}" alt="">
  <figcaption><a href="http://gradientscience.org/adv/">Source</a></figcaption>
</figure>

One way to interpret this graph is that it shows how well a particular architecture is able to capture non-robust features in an image. [^5]

Notice how far back VGG is compared to the other models.

In the unrelated field of neural style transfer, VGG is also quite special since non-VGG architectures are known [to not work very well][vggtables] [^3] without some sort of [parameterization trick][diff_img_params_style_transfer].
The above interpretation of the graph provides an alternative explanation for this phenomenon.
**Since VGG is unable to capture non-robust features as well as other architectures, the outputs for style transfer actually look more correct to humans!** [^4]

Before proceeding, let's quickly discuss the results obtained by Mordvintsev, et. al. in [Differentiable Image Parameterizations][diff_img_params], where they show that non-VGG architectures can be used for style transfer with a simple technique. 
In their experiment, instead of optimizing the output image in RGB space, they optimize it in Fourier space, and run the image through a series of transformations (e.g jitter, rotation, scaling) before passing it through the neural network. 

<figure class="align-center">
  <img src="{{ '/images/rnst/diff_image_params_style_transfer.png' | absolute_url }}">
  <figcaption>Style transfer on non-VGG architectures via decorrelated parameterization and transformation robustness. From <a href="https://distill.pub/2018/differentiable-parameterizations/">Differentiable Image Parameterizations</a> by Mordvintsev, et. al.</figcaption>
</figure>

Can we reconcile this result with our hypothesis linking neural style transfer and non-robust features?

One possible theory is that all of these image transformations *weaken* or even *destroy* non-robust features.
Since the optimization can no longer reliably manipulate non-robust features to bring down the loss, it is forced to use robust features instead, which are presumably more robust to the applied image transformations (a rotated and jittered flappy ear still looks like a flappy ear).

Testing this hypothesis is fairly straightforward: 
Use an adversarially robust classifier for (regular) neural style transfer and see what happens.

Fortunately, Engstrom, et. al. open-sourced their [code and model weights][robust_github] for a robust ResNet-50, saving me the trouble of having to train my own. 
I compared a regularly trained (non-robust) ResNet-50 with a robustly trained ResNet-50 on their performance on Gatys, et. al.'s original [neural style transfer][neural_style_transfer_arxiv] algorithm. 
For comparison, I also performed the style transfer with a regular VGG-19.

My experiments can be fully reproduced inside this [Colab notebook][colab_link]. 
To ensure a fair comparison despite different networks having different optimal hyperparameters, I performed a small grid search for each image and manually picked the best output per network. 
Further details can be read in a footnote [^6]. 

The results of the experiment can be explored in the diagram below.

<script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
<link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
<script src="{{ '/assets/image-picker/image-picker.min.js' | absolute_url }}"></script>
<link rel="stylesheet" href="{{ '/assets/image-picker/image-picker.css' | absolute_url }}">
<style>
#style-transfer-slider.juxtapose {
  max-height: 512px;
  max-width: 512px;
}
</style>

<div style="margin-bottom: 30px;">
<b>Content image</b>
<select id="content-select" class="image-picker">
    <option data-img-src="{{ '/images/rnst/thumbnails/ben.jpg' | absolute_url }}" value="ben"></option>
    <option data-img-src="{{ '/images/rnst/thumbnails/tubingen.jpg' | absolute_url }}" value="tubingen"></option>
</select>
<b>Style image</b>
<select id="style-select" class="image-picker">
    <option data-img-src="{{ '/images/rnst/thumbnails/scream.jpg' | absolute_url }}" value="scream"></option>
    <option data-img-src="{{ '/images/rnst/thumbnails/woman.jpg' | absolute_url }}" value="woman"></option>
    <option data-img-src="{{ '/images/rnst/thumbnails/picasso.jpg' | absolute_url }}" value="picasso"></option>
</select>
<input id="check-compare-vgg" type="checkbox"><small>&nbsp; Compare VGG <> Robust ResNet</small>
<div id="style-transfer-slider" class="align-center"></div>
</div>
<script src="{{ '/assets/rnst/js/style-transfer-slider.js' | absolute_url }}"></script>

Success! 
The robust ResNet shows drastic improvement over the regular ResNet. 
Remember, all we did was switch the ResNet's weights, the rest of the code for performing style transfer is exactly the same!

A more interesting comparison can be done between VGG-19 and the robust ResNet. 
At first glance, the robust ResNet's outputs seem on par with VGG-19. 
Looking closer, however, we see that the ResNet outputs are slightly noisier and exhibit some artifacts [^7].

Diagram here

It is currently unclear exactly what causes these artifacts. 
One theory is that they are [checkerboard artifacts][checkerboard_artifacts] (Odena, et. al.) caused by non-divisible kernel size and stride in the convolution layers.
They could also be artifacts caused by the presence of max pooling layers ([Henaff, et. al.][max_pool_artifacts_arxiv]). 
Whatever the case, it seems like evidence that these checkerboard artifacts, while problematic, are distinct from the problem that adversarial robustness solves in neural style transfer.

[^1]: Adversarial examples are inputs that are specially crafted by an attacker to trick a classifier into producing an incorrect label for that input. There is an entire field of research dedicated to adversarial attacks and defenses in deep learning literature.
[^2]: This is usually defined as being in some pre-defined perturbation set such as an L2 ball. Humans don't notice individual pixels changing within some pre-defined epsilon, so any perturbations within this set can be used to create an adversarial example.  
[^3]: This phenomenon is discussed at length in [this Reddit thread][vggtables].
[^4]: To follow this argument, note that the perceptual losses used in neural style transfer are dependent on matching features learned by a separately trained image classifier. If these learned features don't make sense to humans (non-robust features), the outputs for neural style transfer won't make sense either.
[^5]: Since the non-robust features are defined by the non-robust features ResNet-50 captures, $$NRF_{resnet}$$, what this graph really shows is how well an architecture captures $$NRF_{resnet}$$.
[^6]: L-BFGS was used for optimization as it showed faster convergence over Adam. For ResNet-50, the style layers used were the ReLu outputs after each of the 4 residual blocks, $$[relu2\_x, relu3\_x, relu4\_x, relu5\_x]$$ while the content layer used was $$relu4\_x$$. For VGG-19, style layers $$[relu1\_1,relu2\_1,relu3\_1,relu4\_1,relu5\_1]$$ were used with a content layer $$relu4\_2$$. In VGG-19, max pooling layers were replaced with avg pooling layers, as is standard practice.
[^7]: This is more obvious when the output image is initialized not with the content image, but with Gaussian noise. 

[not_bugs_features_arxiv]: https://arxiv.org/abs/1905.02175
[not_bugs_features_blog]: http://gradientscience.org/adv/
[diff_img_params]: https://distill.pub/2018/differentiable-parameterizations/
[diff_img_params_style_transfer]: https://distill.pub/2018/differentiable-parameterizations/#section-styletransfer
[vggtables]: https://www.reddit.com/r/MachineLearning/comments/7rrrk3/d_eat_your_vggtables_or_why_does_neural_style/
[robust_github]: https://github.com/MadryLab/robust_representations
[neural_style_transfer_arxiv]: https://arxiv.org/abs/1508.06576
[colab_link]: https://google.com
[checkerboard_artifacts]: https://distill.pub/2016/deconv-checkerboard/
[max_pool_artifacts_arxiv]: https://arxiv.org/abs/1511.06394
