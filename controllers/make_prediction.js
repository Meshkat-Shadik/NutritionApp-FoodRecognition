/********************************************************************************************************/
/********************************************************************************************************/
/*********************************** This File is not used **********************************************/
/********************************* Only for testing purpose *********************************************/
/********************************************************************************************************/
/********************************************************************************************************/

const tfnode = require("@tensorflow/tfjs-node");
const tff = require("@tensorflow/tfjs");
const mobilenet = require("@tensorflow-models/mobilenet");
const { Image } = require("image-js");
const getImage = require("get-image-data");
const fs = require("fs");
const { createImageData } = require("canvas");
//const ImageData = require('Image');

exports.makePredictions = async(req, res, next) => {
    async function execute() {
        let image = await Image.load("./public/uploads/test-image.jpg");
        let grey = image.resize({ height: 224, width: 224 });
        return grey;
    }

    let buffer = await await execute();
    // console.log(buffer);

    // let model = await tff.loadLayersModel(
    //     "http://localhost:3001/tfjsModels/mobilenet/model.json"
    // );
    // if (model) {
    //     console.log("Loaded!!!!!!!!");
    //     // async function classify(imageBuffer, topk = 3) {
    //     //     const tfimage = tf.node.decodeImage(imageBuffer);
    //     //     return classify(tfimage, topk);
    //     // }
    // }

    const ui8ca = new Uint8Array(buffer);

    const readImage = (imagePath) => {
        //reads the entire contents of a file.
        //readFileSync() is synchronous and blocks execution until finished.
        const imageBuffer = fs.readFileSync(imagePath);
        //Given the encoded bytes of an image,
        //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
        const tfimage = tfnode.node.decodeImage(imageBuffer);
        return tfimage;
    };

    const imageClassification = async(imagePath) => {
        let image = readImage(imagePath);
        return new Promise(async(resolve, reject) => {
            try {
                const model = await tff.loadLayersModel(
                    "http://localhost:3001/tfjsModels/mobilenet/model.json"
                );
                // const y = tfnode.tensor3d(image.expandDims());
                //console.log(y);
                image = await image
                    .resizeNearestNeighbor([224, 224])
                    .sub(tff.scalar(255))
                    .div(tff.scalar(255));
                const predictions = await model.predict(image.expandDims(0));
                let x = predictions.argMax((axis = -1));

                console.log(x);
                for (let n = 0; n < x.length; n++) {
                    console.log(x[n]);
                }
                x.print();
                console.log(x.arraySync());
            } catch (err) {
                return reject(err);
            }
        });
    };

    const result = await imageClassification("./uploads/test-image.jpg");

    res.status(200).json({
        data: result,
    });

    // const imageBuffer = await fetch(
    //     "http://localhost:3001/uploads/test-image.jpg"
    // );
    // getImage("./uploads/test-image.jpg", async(err, imgData) => {
    //     if (err) {
    //         console.log(err);
    //     } else {
    //         const numChannels = 3;
    //         const numPixels = 224 * 224;
    //         let values = new Uint32Array(numPixels * numChannels);
    //         const pixels = imgData.data;
    //         for (let i = 0; i < numPixels; i++) {
    //             for (let channel = 0; channel < numChannels; ++channel) {
    //                 values[i * numChannels + channel] = pixels[i * 4 + channel];
    //             }
    //         }
    //         // values = values / 255;
    //         // const outShape = [224, 224, numChannels];
    //         //   const input = tf.tensor3d(values, outShape, "int32");
    //         //   console.log(values);
    //         //let ans = await preProcessImage(imgData.data);
    //         // const img = imgData.data_url.replace(
    //         //     /^data:image\/(png|jpeg|jpg);base64,/,
    //         //     ""
    //         // );
    //         //  const b = Buffer.from(img, "base64");
    //         const dat = await tf.node.decodeImage(values);
    //         //   let ans = await preProcessImage(dat);
    //         console.log(dat);
    //     }
    // });
    async function preProcessImage(image) {
        let tensorImg = (await tff.browser.fromPixelsAsync(image))
            .resizeNearestNeighbor([224, 224])
            .toFloat();
        let offset = tff.scalar(255);
        return tensorImg.sub(offset).div(offset).expandDims();
    }
    // console.log(imageBuffer);
    //const tensor = tf.node.decodeImage(imageBuffer);
    // const decoded = tf.node.decodeImage(buffer);
    //const prediction = await model.console.log(decoded);
    //console.log(await preProcessImage(decoded));
    //const imageData = new ImageData(ui8ca, 224, 224);
    // let tensor = tff.tensor(ui8ca);
    //console.log("tensor" + tensor);

    // res.status(200).json({
    //     data: "hello",
    // });

    // const imagePath = "./images/test-image.jpg";
    // console.log(req.file);
    // try {
    //     const loadModel = async(img) => {
    //         const output = {};
    //         //load model
    //         console.log("Loading.......");
    //         const model = await mobilenet.load();
    //         //classify
    //         output.predictions = await model.classify(img);
    //         console.log(output);
    //         res.status(200).json({
    //             data: output,
    //         });
    //         await image(imagePath, async(err, imageData) => {
    //             //pre process image
    //             const numChannels = 3;
    //             const numPixels = imageData.width * imageData.height;
    //             const values = new Int32Array(numPixels * numChannels);
    //             const pixels = imageData.data;
    //             for (let i = 0; i < numPixels; i++) {
    //                 for (let channel = 0; channel < numChannels; ++channel) {
    //                     values[i * numChannels + channel] = pixels[i * 4 + channel];
    //                 }
    //             }
    //             const outShape = [imageData.height, imageData.width, numChannels];
    //             const input = tf.tensor3d(values, outShape, "int32");
    //             await loadModel(input);
    //             // delete image file
    //             fs.unlinkSync(imagePath, (error) => {
    //                 if (error) {
    //                     console.error(error);
    //                 }
    //             });
    //         });
    //     };
    // } catch (error) {
    //     console.log(error);
    // }
};