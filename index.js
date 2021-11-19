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
const port = process.env.PORT || 3001;

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
  0: "Apple",
  1: "Banana",
  2: "Broccoli",
  3: "Carrots",
  4: "Cauliflower",
  5: "Chili",
  6: "Coconut",
  7: "Cucumber",
  8: "Custard apple",
  9: "Dates",
  10: "Dragon",
  11: "Egg",
  12: "Garlic",
  13: "Grape",
  14: "Green Lemon",
  15: "Jackfruit",
  16: "Kiwi",
  17: "Mango",
  18: "Okra",
  19: "Onion",
  20: "Orange",
  21: "Papaya",
  22: "Peanut",
  23: "Pineapple",
  24: "Pomegranate",
  25: "Star Fruit",
  26: "Strawberry",
  27: "Sweet Potato",
  28: "Watermelon",
  29: "White Mushroom",
};

let model;
(async function () {
  model = await tf.loadLayersModel(
    //   "file://public/uploads/tfjsModels/mobilenet/model.json"
    "file://./public/tfjsModels/mobilenet/model.json"
  );
})();

app.post("/v1", upload.single("image"), async (req, res) => {
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

  console.log("this is resl" + resl);
  let tempArr = [];
  resl.map((e) => tempArr.push(e));
  console.log(resl[0][1]);

  let firstMax = 0;
  let firstPos = 0;

  console.log(typeof resl);
  for (let i = 0; i < 30; i++) {
    if (resl[0][i] > firstMax) {
      firstMax = resl[0][i];
      firstPos = i;
    }
    // console.log(resl[0][i]);
  }
  console.log("Position and Max");
  console.log(firstPos, firstMax);

  let secondMax = 0;
  let secondPos = firstPos;

  for (let i = 0; i < 30; i++) {
    if (resl[0][i] > secondMax) {
      if (resl[0][i] == firstMax) {
        continue;
      }
      secondMax = resl[0][i];
      secondPos = i;
    }
    // console.log(resl[0][i]);
  }
  console.log("Position and Max");
  console.log(secondPos, secondMax);

  let thirdMax = 0;
  let thirdPos = secondPos;

  for (let i = 0; i < 30; i++) {
    if (resl[0][i] > thirdMax) {
      if (resl[0][i] == firstMax || resl[0][i] == secondMax) {
        continue;
      }
      thirdMax = resl[0][i];
      thirdPos = i;
    }
    // console.log(resl[0][i]);
  }
  console.log("Position and Max");
  console.log(thirdPos, thirdMax);

  //let index = tempArr[0].findIndex((dt) => dt != 0);
  //console.log(tempArr[0][1]);
  // console.log(index);
  // console.log(IMAGE_CLASS[index]);
  // let temp = await prediction.argMax((axis = -1));
  // console.log("After Argmax\n");
  // temp.print();
  // let x = temp.map((e) => console.log(e));
  //let ans = await temp.array();
  // console.log(temp);
  //temp.print();
  //  console.log(x);
  res.status(200).json({
    //data: IMAGE_CLASS[index],
    data: [
      { name: IMAGE_CLASS[firstPos], probability: firstMax * 100 },
      { name: IMAGE_CLASS[secondPos], probability: secondMax * 100 },
      { name: IMAGE_CLASS[thirdPos], probability: thirdMax * 100 },
    ],
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
