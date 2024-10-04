from setuptools import find_packages, setup

setup(
    name="dagster_mlflow_project",
    packages=find_packages(exclude=["dagster_mlflow_project_evaluate"]),
    install_requires=[
        "dagster",
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)