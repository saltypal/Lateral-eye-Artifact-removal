import hashlib
import io
import zipfile

import pytest

from vmd_eog.frontal_benchmark_eligibility import REQUIREMENTS, assess_candidate, extract_sensitivity_table


def candidate():
    return {"id": "fixture", "source": "fixture evidence", "role": "test only", "limitations": "not a real source",
            "requirements": {key: "verified" for key in REQUIREMENTS}}


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_every_unknown_requirement_blocks_independent_benchmark(requirement):
    source = candidate()
    source["requirements"][requirement] = "unresolved"
    result = assess_candidate(source)
    assert result["eligible"] is False
    assert result["blocking_requirements"] == [{"requirement": requirement, "status": "unresolved"}]


def test_fixture_contract_readiness_is_not_a_cleaning_metric():
    result = assess_candidate(candidate())
    assert result["eligible"] is True
    assert result["assessment_kind"] == "documentary contract; not an empirical cleaning metric"


@pytest.mark.parametrize("change", ["missing", "extra", "boolean"])
def test_malformed_contract_rejected(change):
    source = candidate()
    if change == "missing":
        del source["requirements"][REQUIREMENTS[0]]
    elif change == "extra":
        source["requirements"]["invented"] = "verified"
    else:
        source["requirements"][REQUIREMENTS[0]] = True
    with pytest.raises(ValueError):
        assess_candidate(source)


def document(rows):
    header = ["", "Blink (% of VEOG)", "Vertical saccade (% of VEOG)", "Horizontal saccade (% of HEOG)"]
    columns = ["Scalp EEG channel", "Lins", "COMSOL", "Lins", "COMSOL", "Lins", "COMSOL"]
    body = "".join("<w:tr>" + "".join("<w:tc><w:p><w:r><w:t>" + cell + "</w:t></w:r></w:p></w:tc>" for cell in row) + "</w:tr>"
                   for row in [header, columns, *rows])
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl>' + body + '</w:tbl></w:body></w:document>'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return stream.getvalue()


def test_source_row_grouping_and_text_are_retained_without_signs():
    raw = document([["F7/8", "17", "10", "15", "14", "22", "31"]])
    result = extract_sensitivity_table(raw, hashlib.sha256(raw).hexdigest())
    assert result["data_rows"] == [["F7/8", "17", "10", "15", "14", "22", "31"]]
    assert "no signed individual channel map inferred" in result["interpretation"]


def test_changed_source_checksum_rejected():
    raw = document([["Fz", "18", "19", "18", "25", "0", "4"]])
    with pytest.raises(ValueError, match="checksum"):
        extract_sensitivity_table(raw, "0" * 64)


@pytest.mark.parametrize("rows", [[], [["F7/8", "17"]], [["Fz", "18", "19", "18", "25", "0", "4"]] * 2])
def test_missing_short_or_duplicate_source_table_rejected(rows):
    raw = document(rows)
    with pytest.raises(ValueError):
        extract_sensitivity_table(raw, hashlib.sha256(raw).hexdigest())
