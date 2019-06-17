// I don't know how to write JavaScript without a bundler. Please someone save me.

// Initialize slider
var currentContent = 'ben';
var currentStyle = 'scream';
var currentLeft = 'nonrobust';

const switchStyleTransferBtn = document.getElementById("switch-style-transfer");
const styleTransferSliderDiv = document.getElementById("style-transfer-slider");

function refreshSlider() {
  while (styleTransferSliderDiv.firstChild) {
      styleTransferSliderDiv.removeChild(styleTransferSliderDiv.firstChild);
  }
  const imgPath1 = '/images/rnst/style-transfer/' + currentContent + '_' + currentStyle + '_' + currentLeft + '.jpg';
  const imgPath2 = '/images/rnst/style-transfer/' + currentContent + '_' + currentStyle + '_robust.jpg';
  new juxtapose.JXSlider('#style-transfer-slider',
      [
          {
              src: imgPath1, // TODO: Might need to use absolute_url?
              label: currentLeft === 'nonrobust' ? 'Non-robust ResNet50' : 'VGG'
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

refreshSlider();

switchStyleTransferBtn.onclick = function() {
  currentLeft = currentLeft === 'nonrobust' ? 'vgg' : 'nonrobust';
  switchStyleTransferBtn.textContent = currentLeft === 'nonrobust' ? 
      'Compare VGG <> Robust ResNet' : 
      'Compare Non-robust ResNet <> Robust ResNet';
  styleTransferSliderDiv.removeChild(styleTransferSliderDiv.lastElementChild);
  refreshSlider();
}

// Initialize selector
$("#content-select").imagepicker({
  changed: function(oldVal, newVal, event) {
    currentContent = newVal;
    refreshSlider();
  }
});
$("#style-select").imagepicker({
  changed: function(oldVal, newVal, event) {
    currentStyle = newVal;
    refreshSlider();
  }
});
