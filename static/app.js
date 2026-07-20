// Find the elements we need from the page, using their ids.
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusDiv = document.getElementById("status");
const resultDiv = document.getElementById("result");

// When the Analyze button is clicked, run this function.
analyzeBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];

    // Make sure a file was actually chosen.
    if (!file) {
        statusDiv.textContent = "Please choose a .set file first.";
        return;
    }

    // Give feedback and disable the button while we wait.
    statusDiv.textContent = "Analyzing... this may take a moment.";
    analyzeBtn.disabled = true;
    resultDiv.classList.add("hidden");

    try {
        // Package the file to send it, just like the /docs form did.
        const formData = new FormData();
        formData.append("file", file);

        // Send the file to our API's /predict endpoint.
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Something went wrong.");
        }

        // Read the JSON prediction the API sent back.
        const data = await response.json();

        // Put the values into the page.
        document.getElementById("prediction").textContent = data.prediction;
        document.getElementById("adProb").textContent = data.ad_probability;
        document.getElementById("fractionAd").textContent = data.fraction_windows_ad;
        document.getElementById("nWindows").textContent = data.n_windows;

        // Show the result section and clear the status.
        resultDiv.classList.remove("hidden");
        statusDiv.textContent = "";
    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
    } finally {
        // Re-enable the button whether it succeeded or failed.
        analyzeBtn.disabled = false;
    }
});