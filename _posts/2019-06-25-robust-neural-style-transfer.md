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

The authors propose the existence of so-called "robust" and "non-robust" features in the dataset used for training e.g. CIFAR10, ImageNet. 
<i>Robust features</i> can be thought of as features that humans naturally use for classification e.g. flappy ears are indicative of certain breeds of dogs, while black and white stripes are indicative of zebras.
<i>Non-robust features</i>, on the other hand, are features that humans aren't sensitive to [^2], but truly correlate with a particular class (i.e. exist in the train and test set).
The authors argue that adversarial examples are produced by replacing the non-robust features in an image with non-robust features of another class.

I highly recommend reading the [paper][not_bugs_features_arxiv] or at the very least, the accompanying [blog post][not_bugs_features_blog].

One figure in the paper that particularly struck me as interesting was the following graph showing the correlation between transferability of adversarial examples to the ability to learn similar non-robust features.

<figure class="align-center">
  <img src="{{ '/images/rnst/transferability.png' | absolute_url }}" alt="">
  <figcaption><a href="http://gradientscience.org/adv/">Source</a></figcaption>
</figure>

One way to interpret this graph is that it shows how well a particular architecture is able to capture non-robust features in an image. 

Notice how far back VGG is to the other models.
In the unrelated field of neural style transfer, VGG is quite special since non-VGG architectures are known [not to work very well][vggtables] [^3] without some sort of [parameterization trick][diff_img_params_style_transfer].
The above interpretation of the graph provides an alternative explanation for this phenomenon.
Since VGG is unable to capture non-robust features as well as other architectures, the outputs for style transfer actually look more correct to humans! [^4]

Testing this hypothesis is very straightforward: Use an adversarially robust classifier for neural style transfer and see what happens.

<script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
<link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
<button id='switch-style-transfer'>Compare VGG \<\> Robust ResNet</button>
<div id="style-transfer-slider"></div>
<script>
var currentContent = 'tubingen';
var currentStyle = 'woman';
var currentLeft = 'nonrobust';
function refreshSlider(content, style, left) {
  const imgPath1 = '/images/rnst/style-transfer/' + currentContent + '_' + currentStyle + '_' + left + '.jpg';
  const imgPath2 = '/images/rnst/style-transfer/' + currentContent + '_' + currentStyle + '_robust.jpg';
  new juxtapose.JXSlider('#style-transfer-slider',
      [
          {
              src: imgPath1, // TODO: Might need to use absolute_url?
              label: left === 'nonrobust' ? 'Non-robust ResNet50' : 'VGG'
          },
          {
              src: imgPath2,
              label: 'Robust ResNet50'
          }
      ],
      {
          animate: true,
          showLabels: true,
          showCredits: false,
          startingPosition: "50%",
          makeResponsive: true
  });
}
refreshSlider(currentContent, currentStyle, currentLeft);
var switchStyleTransferBtn = document.getElementById("switch-style-transfer");
var styleTransferSliderDiv = document.getElementById("style-transfer-slider");
switchStyleTransferBtn.onclick = function() {
  currentLeft = currentLeft === 'nonrobust' ? 'vgg' : 'nonrobust';
  switchStyleTransferBtn.textContent = currentLeft === 'nonrobust' ? 
      'Compare VGG <> Robust ResNet' : 
      'Compare Non-robust ResNet <> Robust ResNet';
  styleTransferSliderDiv.removeChild(styleTransferSliderDiv.lastElementChild);
  refreshSlider(currentContent, currentStyle, currentLeft);
}
</script>


[^1]: Adversarial examples are inputs that are specially crafted by an attacker to trick a classifier into producing an incorrect label for that input. There is an entire field of research dedicated to adversarial attacks and defenses in deep learning literature.
[^2]: This is usually defined as being in some pre-defined perturbation set such as an L2 ball. Humans don't notice individual pixels changing within some pre-defined epsilon, so any perturbations within this set can be used to create an adversarial example.  
[^3]: This phenomenon is discussed at length in [this Reddit thread][vggtables].
[^4]: To follow this argument, note that the perceptual losses used in neural style transfer are dependent on matching features learned by a separately trained image classifier. If these learned features don't make sense to humans (non-robust features), the outputs for neural style transfer won't make sense either.

[not_bugs_features_arxiv]: https://arxiv.org/abs/1905.02175
[not_bugs_features_blog]: http://gradientscience.org/adv/
[diff_img_params_style_transfer]: https://distill.pub/2018/differentiable-parameterizations/#section-styletransfer
[vggtables]: https://www.reddit.com/r/MachineLearning/comments/7rrrk3/d_eat_your_vggtables_or_why_does_neural_style/
