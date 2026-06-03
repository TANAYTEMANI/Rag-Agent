"""
Pandas Query Engine.
Excel/CSV files are converted to Parquet DataFrames at ingestion time.
At query time, Claude Haiku generates a Pandas expression which is
evaluated safely via asteval.
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pandas as pd
from asteval import Interpreter

from config.settings import settings
from index.postgres_store import AsyncSessionLocal, upsert_tabular_table

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

_PANDAS_PROMPT = """\
You are a Python/Pandas expert. Write a single Pandas expression using the variable `df` \
that answers the question. Return ONLY the expression — no imports, no assignments, no explanations.

DataFrame schema:
{schema}

Sample rows (first 3):
{sample}

Question: {question}

Expression:"""


class PandasQueryEngine:
    def __init__(self):
        self.store_dir = Path(settings.table_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}

    async def ingest_file(self, file_path: str, doc_id: str,
                          file_name: str) -> list[dict]:
        """Convert Excel/CSV to Parquet. Store metadata in PostgreSQL."""
        ext = Path(file_path).suffix.lower()
        results = []

        if ext in (".xlsx", ".xls"):
            sheets: dict[str, pd.DataFrame] = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in sheets.items():
                result = await self._store_df(
                    df, doc_id=doc_id, file_name=file_name,
                    table_id=f"{doc_id}_{sheet_name}",
                    sheet_name=sheet_name,
                )
                results.append(result)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            result = await self._store_df(
                df, doc_id=doc_id, file_name=file_name,
                table_id=f"{doc_id}_csv",
                sheet_name=None,
            )
            results.append(result)

        return results

    async def _store_df(self, df: pd.DataFrame, doc_id: str, file_name: str,
                        table_id: str, sheet_name: str | None) -> dict:
        path = self.store_dir / f"{table_id}.parquet"
        df.to_parquet(path)

        schema = self._describe_schema(df)
        description = await self._generate_description(df, schema, file_name, sheet_name)

        async with AsyncSessionLocal() as session:
            await upsert_tabular_table(
                session,
                table_id=table_id,
                doc_id=doc_id,
                file_name=file_name,
                sheet_name=sheet_name,
                parquet_path=str(path),
                row_count=len(df),
                col_count=len(df.columns),
                columns=list(df.columns),
                description=description,
            )

        return {
            "table_id":    table_id,
            "row_count":   len(df),
            "col_count":   len(df.columns),
            "description": description,
        }

    async def query(self, table_id: str, question: str) -> str:
        """Generate and safely execute a Pandas expression."""
        if table_id not in self._cache:
            path = self.store_dir / f"{table_id}.parquet"
            if not path.exists():
                return f"Table {table_id} not found."
            self._cache[table_id] = pd.read_parquet(path)

        df = self._cache[table_id]
        schema = self._describe_schema(df)
        sample = df.head(3).to_string()

        response = await _client.messages.create(
            model=settings.enrichment_model,
            max_tokens=300,
            messages=[{"role": "user", "content":
                _PANDAS_PROMPT.format(schema=schema, sample=sample, question=question)
            }],
        )
        expression = response.content[0].text.strip()

        # Safe evaluation via asteval — no builtins, no imports
        aeval = Interpreter()
        aeval.symtable["df"] = df
        aeval.symtable["pd"] = pd
        result = aeval(expression)

        if aeval.error:
            errors = "; ".join(str(e.get_error()) for e in aeval.error)
            return f"Could not compute answer: {errors}"

        return f"Result: {result}"

    def _describe_schema(self, df: pd.DataFrame) -> str:
        lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "Columns:"]
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            lines.append(f"  - {col} ({df[col].dtype}): e.g. {sample!r}")
        return "\n".join(lines)

    async def _generate_description(self, df: pd.DataFrame, schema: str,
                                     file_name: str, sheet: str | None) -> str:
        label = f"{file_name}" + (f" / {sheet}" if sheet else "")
        response = await _client.messages.create(
            model=settings.enrichment_model,
            max_tokens=150,
            messages=[{"role": "user", "content":
                f"In 2 sentences, describe what questions this table can answer.\n\n"
                f"Source: {label}\n{schema}"
            }],
        )
        return response.content[0].text.strip()
