
# Boston Housing Price Prediction

This project uses a Random Forest Regressor model to predict house prices in the Boston area based on various features such as crime rate, number of rooms, etc. The model is deployed using FastAPI as an API, and a front-end interface is created to interact with the model.

## Project Overview

The goal of this project is to predict the median house prices in Boston based on various features. It uses the **Boston Housing Dataset** and a **Random Forest Regressor** model trained on the dataset. The model is then deployed as an API using **FastAPI**, which can be interacted with using a web front-end interface.

## Features

- **Random Forest Model**: Trained using the Boston Housing Dataset to predict house prices.
- **FastAPI Backend**: Handles API requests for predictions.
- **HTML Front-End**: Simple form to input house features and get predictions.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/Hitesh-jadhav/house_price-prediction.git
   cd house_price-prediction
   ```

2. Set up a Python virtual environment (recommended):
   ```
   python -m venv .venv
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the trained Random Forest model (if not already present):
   - Place the `rf_model.pkl` file in the project directory.

5. Run the FastAPI backend:
   ```
   uvicorn app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

6. Open the HTML file `index.html` in a browser and fill in the input fields to predict house prices.

## API Endpoints

### POST /predict

This endpoint takes the following inputs:

- `crim`: Crime rate (float)
- `zn`: Proportion of residential land zoned for large lots (float)
- `indus`: Proportion of non-retail business acres per town (float)
- `chas`: Charles River dummy variable (1 if tract bounds river, else 0) (integer)
- `nox`: Nitrogen oxide concentration (float)
- `rm`: Average number of rooms per dwelling (float)
- `age`: Proportion of owner-occupied units built before 1940 (float)
- `dis`: Weighted distance to employment centers (float)
- `rad`: Index of accessibility to radial highways (integer)
- `tax`: Property tax rate (float)
- `ptratio`: Pupil-teacher ratio (float)
- `b`: Proportion of residents of African American descent (float)
- `lstat`: Percentage of lower status population (float)

### Response:

The response will contain the predicted house price in thousands of dollars, formatted with a dollar sign:

```json
{
    "predicted_price": "$24,670"
}
```

## Front-End

The front-end consists of a simple HTML form where users can input values for various features and receive predictions. The result is displayed on the page after submitting the form.

### Example:

```html
<form id="prediction-form">
    <input type="number" id="crim" name="crim">
    <!-- Other input fields here -->
    <button type="submit">Predict</button>
</form>
<div id="result"></div>
```

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Scikit-learn
- joblib
- HTML, CSS (for the front-end)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

