let model;

const modelURL = '/model';
const IMAGE_SIZE = 224;

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');

const scores_tensor = tf.linspace(1.0, 10.0, 10);

const load = async () => {
    // Download the mode
    model = await tf.loadLayersModel(modelURL);

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
};

const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);

    const files = fileInput.files;

    [...files].forEach(async (img) => {
        const data = new FormData();
        data.append('file', img);

        const processedImage = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
            return response.json();
        }).then(result => {
            return tf.tensor3d(result['image']);
        });

        // shape has to be the same as it was for training of the model
        const reshapedImg = tf.reshape(processedImage, [1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        const prediction = model.predict(reshapedImg, {batchSize: 1});
        const score_tensor = tf.dot(prediction.flatten(), scores_tensor);
        const score_array = await score_tensor.data();
        const score = score_array[0];

        renderImageLabel(img, score.toFixed(2));
    })
};

const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    reader.onload = () => {
        preview.innerHTML += `<div class="image-block">
                                      <img src="${reader.result}" class="image-block_loaded" id="source"/>
                                       <h2 class="image-block__label">${label}</h2>
                              </div>`;

    };
    reader.readAsDataURL(img);
};


fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");

// Pre-load the model
load();
