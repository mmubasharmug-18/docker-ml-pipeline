from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictInt


def only_numbers(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    return value


Number = Annotated[float, BeforeValidator(only_numbers)]


class PredictIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pregnancies: StrictInt = Field(ge=0, le=20, examples=[2])
    glucose: Number = Field(ge=0, le=300, allow_inf_nan=False, examples=[120])
    blood_pressure: Number = Field(ge=0, le=200, allow_inf_nan=False, examples=[70])
    skin_thickness: Number = Field(ge=0, le=100, allow_inf_nan=False, examples=[25])
    insulin: Number = Field(ge=0, le=900, allow_inf_nan=False, examples=[100])
    bmi: Number = Field(ge=0, le=70, allow_inf_nan=False, examples=[32.5])
    diabetes_pedigree: Number = Field(ge=0, le=3, allow_inf_nan=False, examples=[0.45])
    age: StrictInt = Field(ge=1, le=120, examples=[35])


class PredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    probability: float
    threshold: float
    model_version: str
    model_name: str


FIELD_NAMES = list(PredictIn.model_fields)
