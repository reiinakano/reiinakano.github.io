(function() {

$("#banner-style-select").imagepicker({
  changed: function(oldVal, newVal, event) {
    currentStyle = newVal;
    refreshSlider();
  }
});
new juxtapose.JXSlider('#banner-slider',
    [
        {
            src: '/images/rnst/style-transfer/banner_scream_nonrobust.jpg', // TODO: Might need to use absolute_url?
            label: 'Non-robust ResNet50'
        },
        {
            src: '/images/rnst/style-transfer/banner_woman_robust.jpg',
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

})()
