const express = require("express");
const { diskStorage } = require("multer");
const multer = require("multer");
const path = require("path");
const cors = require("cors");
const controller = require("./controllers/pred");
const tf = require("@tensorflow/tfjs-node");
const app = express();
const corsOptions = {
    origin: "*",
};
app.use(express.json());
app.use(cors(corsOptions));
app.use(express.urlencoded({ extended: false }));
const port = 3001;

const UPLOADS_FOLDER = "./public/uploads/";

console.log(path.join(__dirname, "public"));
app.use("/", express.static(path.join(__dirname, "public")));

const storage = diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOADS_FOLDER);
    },
    filename: (req, file, cb) => {
        // const fileExt = path.extname(file.originalname);
        // const fileName =
        //     file.originalname
        //     .replace(fileExt, "")
        //     .toLowerCase()
        //     .split(" ")
        //     .join("-") +
        //     "-" +
        //     Date.now();

        cb(null, "test-image.jpg");
    },
});
var upload = multer({
    storage: storage,
    dest: UPLOADS_FOLDER,
    fileFilter: (req, file, cb) => {
        console.log(file.mimetype);

        if (
            file.mimetype == "image/png" ||
            file.mimetype == "image/jpg" ||
            file.mimetype == "image/jpeg"
        ) {
            cb(null, true);
        } else {
            cb(new Error("Only .jpg, .png or .jpeg format allowed!"));
        }
    },
});

// app.post("/", upload.single("avatar"), (req, res) => {
//     res.send("Hello, World!");
// });

const IMAGE_CLASS = {
    0: "apple",
    1: "banana",
    2: "beetroot",
    3: "bell pepper",
    4: "cabbage",
    5: "capsicum",
    6: "carrot",
    7: "cauliflower",
    8: "chilli pepper",
    9: "corn",
    10: "cucumber",
    11: "eggplant",
    12: "garlic",
    13: "ginger",
    14: "grapes",
    15: "jalepeno",
    16: "kiwi",
    17: "lemon",
    18: "lettuce",
    19: "mango",
    20: "onion",
    21: "orange",
    22: "paprika",
    23: "pear",
    24: "peas",
    25: "pineapple",
    26: "pomegranate",
    27: "potato",
    28: "raddish",
    29: "soy beans",
    30: "spinach",
    31: "sweetcorn",
    32: "sweetpotato",
    33: "tomato",
    34: "turnip",
    35: "watermelon",
};

let model;
(async function() {
    model = await tf.loadLayersModel(
        //   "file://public/uploads/tfjsModels/mobilenet/model.json"
        "file://./public/tfjsModels/mobilenet/model.json"
    );
})();

app.post("/test", upload.single("image"), async(req, res) => {
    // await controller.resize();
    var input = await controller.preprocess();
    input.print();
    input = input.div(tf.scalar(255)).resizeNearestNeighbor([224, 224]);

    console.log("After dividing 255\n");
    input.print();
    let prediction = await model.predict(input);
    console.log("After Prediction\n");
    prediction.print();
    let resl = await prediction.array();
    console.log(resl);
    let tempArr = [];
    resl.map((e) => tempArr.push(e));
    let index = tempArr[0].findIndex((dt) => dt != 0);
    //console.log(tempArr[0][1]);
    console.log(index);
    console.log(IMAGE_CLASS[index]);
    //let temp = await prediction.argMax((axis = -1));
    //console.log("After Argmax\n");
    //console.log(temp);
    // let x = temp.map((e) => console.log(e));
    //let ans = await temp.array();
    // console.log(temp);
    //temp.print();
    //  console.log(x);
    res.status(200).json({
        data: IMAGE_CLASS[index],
    });
});

app.use(errorHandler);
//default error handler
function errorHandler(err, req, res, next) {
    if (res.headersSent) {
        return next(err);
    }
    if (err instanceof multer.MulterError) {
        res.status(500).json({ error: err });
    }
    res.status(500).json({ error: err, message: err.message });
}

app.listen(port, () => {
    console.log("App listening at port " + port);
});

/*
Tensor {
  kept: false,
  isDisposedInternal: false,
  shape: [ 1, 224, 224, 3 ],
  dtype: 'float32',
  size: 150528,
  strides: [ 150528, 672, 3 ],
  dataId: {},
  id: 788,
  rankType: '4',
  scopeId: 0



Float32Array(36) [
      0.9999982118606567, 0.000001821745740926417,
   6.536208719309757e-27,                       0,
                       0,  2.5083090113755755e-27,
   5.503100197756794e-10,                       0,
   5.937288128918567e-29,   5.635795163883017e-10,
   6.949857355746006e-18,                       0,
   5.868994433845196e-20,   8.274684756837125e-12,
                       0,   3.204104937230913e-26,
   1.798382229369478e-26,                       0,
  3.0836247393441815e-35,    5.832446969833427e-9,
  1.9650931279532427e-29,   9.301654193790473e-15,
   4.869099827710197e-18,   4.235976568806283e-21,
  3.0673404845940535e-20,     4.5523219848765e-32,
                       0,  1.2974180757518994e-11,
    7.337886298586227e-9,   4.657438189048556e-15,
                       0,   9.098314628592056e-24,
   3.458853083578857e-25,  3.1565410986641894e-38,
                       0,                       0
]















}*/