// const tf = require("@tensorflow/tfjs-node");
// const { IMAGE_CLASS } = require("./image_classes");

// async function loadModel() {
//     console.log("Loading Model.................");
//     model = await tf.loadLayersModel("./tfjsModels/model.json", false);
//     console.log("model load successfull!");
// }

// const items = tf.range(0, 35);
// const itemLen = 35;

// exports.prediction = async function prediction() {

// }

// $for("#image-selector").change(function() {
//     let reader = new FileReader();
//     reader.onload = function() {
//         let dataURL = reader.result;
//         $("#selected-image").attr("src", dataURL);
//         $("#prediction-list").empty();
//     };
//     let file = $("#image-selector").prop("files")[0];
//     reader.readAsDataURL(file);
// });

// let model;
// (async function() {
//     model = await tf.loadModel('http://localhost:3001/tfjsModels/model.json');
//     $('.progress-bar').hide();
// })();

//