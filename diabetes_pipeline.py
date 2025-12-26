import kfp
from kfp import dsl
from typing import NamedTuple, List

# Step 1: Load data into memory and return as lists
@dsl.component(
    base_image="python:3.14",
    packages_to_install=["pandas"]
)
def load_data() -> NamedTuple("Outputs", [("features", List[List[float]]), ("labels", List[int])]):
    import pandas as pd

    # Load dataset from a working source (Kaggle/hosted)
    my_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(my_url)

    # Prepare data for training
    x = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
    y = df["Outcome"]
    
    print("✅ Columns:", df.columns.tolist())  
    return (x.values.tolist(), y.values.tolist())


# Step 2: Train model 
@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["scikit-learn"]
)
def train_model(
    features: List[List[float]],
    labels: List[int]
) -> NamedTuple("Output", [("accuracy", float)]):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score    

    # Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {acc}")    
    return (acc)

# Step 3: Define the pipeline
@dsl.pipeline(
    name="diabetes-pipeline",
    description="ML pipeline without artifacts"
)
def diabetes_pipeline():
    output_data = load_data()
    train_model(
        features=output_data.outputs["features"],
        labels=output_data.outputs["labels"]
    )

# Step 4: Compile
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=diabetes_pipeline,
        package_path="diabetes_pipeline.yaml"
    )
