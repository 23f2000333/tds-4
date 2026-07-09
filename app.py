import os
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
        if typ == "string":
            return str(value)

        if typ == "integer":
            return int(value)

        if typ == "float":
            return float(value)

        if typ == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "yes", "1")

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

    except Exception:
        return None

    return None


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    prompt = f"""
Extract structured information.

Return ONLY valid JSON.

TEXT

{req.text}

SCHEMA

{json.dumps(req.schema, indent=2)}

Rules

Return EXACTLY the keys in the schema.

No extra keys.

Missing values must be null.

Dates must be YYYY-MM-DD.

Numbers must be JSON numbers.

Arrays must be JSON arrays.

Return only JSON.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except Exception:
        data = {}

    result = {}

    for key, typ in req.schema.items():
        result[key] = normalize(data.get(key), typ)

    return result
