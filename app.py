from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from sklearn.datasets import load_iris
import joblib
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = joblib.load("iris_random_forest_model.pkl")

# setup templates
templates = Jinja2Templates(directory="templates")


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=IrisPrediction)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    # Convert input data to a numpy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Make prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
