(function() {

// Initialize slider
var currentStyle = 'woman';

const styleTransferSliderDiv = document.getElementById("banner-slider");

function refreshSlider() {
  while (styleTransferSliderDiv.firstChild) {
      styleTransferSliderDiv.removeChild(styleTransferSliderDiv.firstChild);
  }
  const imgPath = '/images/rnst/style-transfer/' + 'banner_' + currentStyle + '_robust.jpg';
  const imgPathNonRobust = '/images/rnst/style-transfer/' + 'banner_' + currentStyle + '_nonrobust.jpg';
  new juxtapose.JXSlider('#banner-slider',
      [
          {
              src: imgPathNonRobust,
              label: 'Non-robust ResNet50'
          },
          {
              src: imgPath,
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

$("#banner-style-select").imagepicker({
  changed: function(oldVal, newVal, event) {
    currentStyle = newVal;
    refreshSlider();
  }
});

})()
