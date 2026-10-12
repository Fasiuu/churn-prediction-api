from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float = Field(ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.InternetService == "No":
            for f in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                if getattr(self, f) != "No internet service":
                    raise ValueError(f"{f} must be 'No internet service' when InternetService is 'No'")
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")
        return self

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 265.3}}}


class Factor(BaseModel):
    feature: str
    effect: Literal["increases risk", "decreases risk"]
    weight: float


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: Literal["low", "medium", "high"]
    top_factors: list[Factor]
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
