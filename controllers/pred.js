const resizeOptimizeImages = require("resize-optimize-images");
const getImage = require("get-image-data");
const tf = require("@tensorflow/tfjs-node");

exports.preprocess = () => {
    //public\uploads\test-image.jpg
    return new Promise((resolve) => {
        getImage("./public/uploads/test-image.jpg", (err, imageData) => {
            // pre-process image
            const numChannels = 3;
            const numPixels = imageData.width * imageData.height;
            const values = new Int32Array(numPixels * numChannels);
            const pixels = imageData.data;

            for (let i = 0; i < numPixels; i++) {
                for (let channel = 0; channel < numChannels; ++channel) {
                    values[i * numChannels + channel] = pixels[i * 4 + channel];
                }
            }
            const outShape = [imageData.height, imageData.width, numChannels];
            const input = tf.tensor3d(values, outShape, "float32");
            // console.log(input.expandDims(0));
            resolve(input.expandDims(0));
        });
    });
};

exports.resize = async() => {
    const options = {
        images: ["./uploads/test-image.jpg"],
        width: 224,
        height: 224,
        quality: 90,
    };
    await resizeOptimizeImages(options);
    console.log("resized");
};