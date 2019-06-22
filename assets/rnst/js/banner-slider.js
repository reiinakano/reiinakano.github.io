(function() {

// Initialize slider
var currentStyle = 'starry';

const styleTransferSliderDiv = document.getElementById("banner-slider");

function refreshSlider() {
  while (styleTransferSliderDiv.firstChild) {
      styleTransferSliderDiv.removeChild(styleTransferSliderDiv.firstChild);
  }
  const imgPath = '/images/rnst/style-transfer/' + 'banner_' + currentStyle + '_robust.jpg';
  new juxtapose.JXSlider('#banner-slider',
      [
          {
              src: '/images/rnst/style-transfer/banner_scream_nonrobust.jpg',
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
