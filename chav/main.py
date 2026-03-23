from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from chav.engine import analyze
from chav.schemas import ReportOut

app = FastAPI(title="Chav", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_DATA_FILE = File(...)
_REF_FILE = File(None)
_TARGET_FORM = Form(None)
_TIME_FORM = Form(None)


@app.post("/analyze", response_model=ReportOut)
async def analyze_endpoint(
    data: UploadFile = _DATA_FILE,
    reference_data: UploadFile | None = _REF_FILE,
    target: str | None = _TARGET_FORM,
    time_column: str | None = _TIME_FORM,
) -> JSONResponse:
    df = _read_upload(data)

    ref_df = None
    if reference_data is not None:
        ref_df = _read_upload(reference_data)

    report = analyze(
        data=df,
        reference_data=ref_df,
        target=target,
        time_column=time_column,
    )

    return JSONResponse(content=report.to_dict(all=True))


def _read_upload(upload: UploadFile) -> pd.DataFrame:
    filename = upload.filename or ""
    content = upload.file.read()

    if filename.endswith(".csv") or not filename:
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parse error: {e}") from None

    if filename.endswith(".parquet"):
        try:
            return pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parquet parse error: {e}") from None

    raise HTTPException(status_code=400, detail=f"Unsupported format: {filename}")
