var currentContent = 'ben';
var currentStyle = 'scream';
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

const switchStyleTransferBtn = document.getElementById("switch-style-transfer");
const styleTransferSliderDiv = document.getElementById("style-transfer-slider");

switchStyleTransferBtn.onclick = function() {
  currentLeft = currentLeft === 'nonrobust' ? 'vgg' : 'nonrobust';
  switchStyleTransferBtn.textContent = currentLeft === 'nonrobust' ? 
      'Compare VGG <> Robust ResNet' : 
      'Compare Non-robust ResNet <> Robust ResNet';
  styleTransferSliderDiv.removeChild(styleTransferSliderDiv.lastElementChild);
  refreshSlider(currentContent, currentStyle, currentLeft);
}
