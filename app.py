import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DynamicRequest(BaseModel):
    text: str
    schema: dict


def normalize(value, typ):
    if value is None:
        return None

    try:
        typ = typ.lower()

        if typ == "string":
            return str(value)

        if typ == "integer":
            return int(value)

        if typ == "float":
            return float(value)

        if typ == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in (
                "true",
                "yes",
                "1",
                "y",
            )

        if typ == "date":
            return parser.parse(str(value)).strftime("%Y-%m-%d")

        if typ == "array[string]":
            if isinstance(value, list):
                return [str(x) for x in value]
            return [str(value)]

        if typ == "array[integer]":
            if isinstance(value, list):
                return [int(x) for x in value]
            return [int(value)]

        if typ == "array[float]":
            if isinstance(value, list):
                return [float(x) for x in value]
            return [float(value)]

        if typ == "array[boolean]":
            if isinstance(value, list):
                return [
                    str(x).lower() in ("true", "yes", "1")
                    for x in value
                ]
            return [str(value).lower() in ("true", "yes", "1")]

    except Exception:
        return None

    return value


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    prompt = f"""
You are an information extraction engine.

Extract structured information from the text.

TEXT

{req.text}

SCHEMA

{json.dumps(req.schema, indent=2)}

Rules

- Return ONLY valid JSON.
- Return EXACTLY the keys in the schema.
- Do NOT add extra keys.
- Missing values must be null.
- Dates must be YYYY-MM-DD.
- Numbers must be JSON numbers.
- Arrays must be JSON arrays.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract structured information from text.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
    except Exception:
        data = {}

    result = {}

    for key, typ in req.schema.items():
        result[key] = normalize(data.get(key), typ)

    return result
