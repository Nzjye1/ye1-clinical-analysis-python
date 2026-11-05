"""Local clinical-table I/O. Never return or print individual records."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd


class DataError(ValueError):
    pass


def _normalized(name):
    return "".join(c for c in str(name).strip().casefold() if c not in " _-（）()：:")


# Exact header meanings only. Do not guess that arbitrary clinical fields are IDs.
DIRECT_FIELDS = {
    "姓名", "患者姓名", "病人姓名", "住院患者姓名", "name", "patientname", "patientfullname",
    "手机号", "手机号码", "电话号码", "联系电话", "联系手机", "手机", "电话", "phone",
    "phonenumber", "mobile", "mobilenumber", "telephone", "tel",
    "身份证", "身份证号", "身份证号码", "居民身份证号", "居民身份证号码", "idcard",
    "idcardnumber", "identitycard", "identitycardnumber", "nationalid",
}


def _direct(name):
    return _normalized(name) in DIRECT_FIELDS


def _header(values):
    names = ["" if x is None else str(x).strip() for x in values]
    while names and not names[-1]:
        names.pop()
    if not names or any(not x for x in names):
        raise DataError("EMPTY_COLUMN_NAME")
    if len(set(names)) != len(names):
        raise DataError("DUPLICATE_COLUMN_NAME")
    if any(x.startswith("__ye1_") for x in names):
        raise DataError("RESERVED_COLUMN_NAME")
    return names


def _source(path):
    p = Path(path).expanduser()
    if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx"}:
        raise DataError("SOURCE_MUST_BE_EXISTING_CSV_OR_XLSX")
    return p


def prepare(source, output):
    """Remove explicit direct-ID columns in a new local copy; retain other cells.

    It intentionally does not claim full anonymization and does not create IDs.
    Existing workbooks with formulas keep the formulas and formats in the copy.
    """
    src = _source(source)
    target = Path(output).expanduser()
    if target.suffix.lower() != src.suffix.lower() or target.exists():
        raise DataError("OUTPUT_MUST_BE_NEW_AND_MATCH_INPUT_FORMAT")
    if not target.parent.is_dir():
        raise DataError("OUTPUT_PARENT_MUST_EXIST")
    created = False
    try:
        if src.suffix.lower() == ".csv":
            with src.open("r", encoding="utf-8-sig", newline="") as inp:
                reader = csv.reader(inp)
                header = next(reader, None)
                if header is None:
                    raise DataError("EMPTY_TABLE")
                names = _header(header)
                if len(names) != len(header):
                    raise DataError("EMPTY_COLUMN_NAME")
                keep = [i for i, name in enumerate(names) if not _direct(name)]
                if not keep:
                    raise DataError("NO_RESEARCH_COLUMNS_REMAIN")
                with target.open("x", encoding="utf-8-sig", newline="") as out:
                    created = True
                    writer = csv.writer(out)
                    writer.writerow([header[i] for i in keep])
                    for row in reader:
                        if len(row) != len(header):
                            raise DataError("CSV_ROW_WIDTH_MISMATCH")
                        writer.writerow([row[i] for i in keep])
            return {"removed_direct_columns": len(header) - len(keep), "sheets": 1}
        from openpyxl import load_workbook
        book = load_workbook(src)
        removed = 0
        removal_plan = []
        for sheet in book.worksheets:
            header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
            names = _header(header)
            if any(v is not None for v in header[len(names):]):
                raise DataError("EMPTY_COLUMN_NAME")
            remove = [i + 1 for i, name in enumerate(names) if _direct(name)]
            if len(remove) == len(names):
                raise DataError("NO_RESEARCH_COLUMNS_REMAIN")
            removal_plan.append((sheet, remove))
        if any(remove for _, remove in removal_plan):
            # openpyxl cannot safely update formulas, merged regions or embedded
            # chart/table references when deleting columns. Preserve the source
            # and ask for a plain tabular working copy instead of altering them.
            for sheet in book.worksheets:
                if (sheet.merged_cells.ranges or sheet.tables or sheet._charts or
                        any(cell.data_type == "f" for row in sheet.iter_rows() for cell in row)):
                    raise DataError("COMPLEX_XLSX_REQUIRES_LOCAL_REVIEW")
        for sheet, remove in removal_plan:
            for index in reversed(remove):
                sheet.delete_cols(index)
            removed += len(remove)
        with target.open("xb") as out:
            created = True
            book.save(out)
        return {"removed_direct_columns": removed, "sheets": len(book.worksheets)}
    except Exception:
        # Only remove a file created by this failed attempt, never a prior file.
        if created and target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        raise


def load(path, sheet=None):
    """Read a local deidentified working copy without echoing record contents."""
    src = _source(path)
    if src.suffix.lower() == ".csv":
        if sheet is not None:
            raise DataError("CSV_HAS_NO_SHEET")
        with src.open("r", encoding="utf-8-sig", newline="") as inp:
            reader = csv.reader(inp)
            raw = next(reader, None)
            if raw is not None and any(len(row) != len(raw) for row in reader):
                raise DataError("CSV_ROW_WIDTH_MISMATCH")
        if raw is None:
            raise DataError("EMPTY_TABLE")
        names = _header(raw)
        if len(names) != len(raw):
            raise DataError("EMPTY_COLUMN_NAME")
        if any(_direct(x) for x in names):
            raise DataError("DIRECT_ID_COLUMN_REQUIRES_LOCAL_PREPARE")
        frame = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    else:
        from openpyxl import load_workbook
        book = load_workbook(src, read_only=True, data_only=False)
        if sheet is None:
            if len(book.sheetnames) != 1:
                raise DataError("SELECT_ONE_SHEET")
            sheet = book.sheetnames[0]
        if not isinstance(sheet, str) or sheet not in book.sheetnames:
            raise DataError("SHEET_NOT_FOUND")
        if any(cell.data_type == "f" for row in book[sheet].iter_rows() for cell in row):
            raise DataError("FORMULA_CELLS_REQUIRE_LOCAL_REVIEW")
        raw = [cell.value for cell in next(book[sheet].iter_rows(min_row=1, max_row=1))]
        names = _header(raw)
        if any(_direct(x) for x in names):
            raise DataError("DIRECT_ID_COLUMN_REQUIRES_LOCAL_PREPARE")
        frame = pd.read_excel(src, sheet_name=sheet, dtype=str, keep_default_na=False)
        book.close()
    if list(map(str, frame.columns)) != names:
        raise DataError("PARSER_HEADER_MISMATCH")
    if frame.empty:
        raise DataError("EMPTY_TABLE")
    return frame


def safe_inspect(frame):
    """Return column labels and aggregate missing counts, without values."""
    return {"n_rows": len(frame), "n_columns": len(frame.columns),
            "columns": [{"name": str(c), "blank_count": int(frame[c].map(lambda x: str(x).strip() == "").sum())}
                        for c in frame.columns],
            "notice": "Only empty cells were counted as missing; patient rows were not returned."}


def rows_for_compute(frame, identity=None, independent=False):
    if independent is not True:
        raise DataError("CONFIRM_ONE_INDEPENDENT_UNIT_PER_ROW")
    frame = frame.copy()
    if identity:
        if identity not in frame:
            raise DataError("SUBJECT_ID_COLUMN_MISSING")
        ids = frame[identity].map(lambda x: str(x).strip())
        if (ids == "").any() or ids.duplicated().any():
            raise DataError("MISSING_OR_REPEATED_SUBJECT_ID")
        id_name = identity
    else:
        # This transient key is only valid after the caller confirms one subject per row.
        id_name = "__ye1_row_id"
        frame[id_name] = [str(i + 1) for i in range(len(frame))]
    rows = [{k: (None if str(v).strip() == "" else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]
    return rows, id_name
