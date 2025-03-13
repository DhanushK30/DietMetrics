document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("file-input").files[0];
    if (!fileInput) return alert("Please upload an image!");

    const formData = new FormData();
    formData.append("file", fileInput);

    // Show loading message
    document.getElementById("loading").style.display = "block";
    document.getElementById("result-container").style.display = "none";

    try {
        // Send image to backend for OCR and classification
        const response = await fetch("/process-image", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        // Hide loading and show results
        document.getElementById("loading").style.display = "none";
        document.getElementById("result-container").style.display = "block";
        document.getElementById("extracted-text").textContent = result.text || "No text detected.";
        document.getElementById("prediction").textContent = result.prediction || "Unable to classify.";
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the image.");
        document.getElementById("loading").style.display = "none";
    }
});
