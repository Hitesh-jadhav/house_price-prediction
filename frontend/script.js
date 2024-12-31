document.getElementById("prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    
    // Gather form data
    const formData = new FormData(event.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value); // Ensure data is parsed to float
    });

    console.log("Sending the following data to the backend:", data);  // Log the data
    
    // Call the FastAPI backend for prediction
    const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    // Get the prediction result
    const result = await response.json();

    console.log("Received result from backend:", result);  // Log the result

    // Display the result
    document.getElementById("result").innerHTML = `Predicted Price: ${result.predicted_price}`;
});
