from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import streamlit as st
from Bio import Align
from Bio.Align import substitution_matrices

UNIPROT_BASE = "https://rest.uniprot.org"


class UniProtError(RuntimeError):
    pass


@dataclass
class SequenceRecord:
    source: str
    identifier: str
    description: str
    sequence: str


@dataclass
class AlignmentStats:
    alignment_type: str
    score: float
    alignment_length: int
    identities: int
    positives: int
    mismatches: int
    gaps: int
    query_coverage: float
    subject_coverage: float
    percent_identity: float
    percent_positive: float


@dataclass
class AlignmentResult:
    query: SequenceRecord
    subject: SequenceRecord
    aligned_query: str
    aligned_subject: str
    middle_line: str
    stats: AlignmentStats


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")


def clean_sequence(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty sequence input")

    if lines[0].startswith(">"):
        lines = [line for line in lines[1:] if not line.startswith(">")]

    seq = "".join(lines).replace(" ", "").upper()
    if not seq:
        raise ValueError("No sequence characters found")

    invalid = sorted(set(seq) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid sequence characters found: {''.join(invalid)}")

    return seq


def fetch_uniprot_by_accession(accession: str, timeout: int = 30) -> SequenceRecord:
    accession = accession.strip()
    fasta_url = f"{UNIPROT_BASE}/uniprotkb/{accession}.fasta"
    response = requests.get(fasta_url, timeout=timeout)
    if response.status_code == 404:
        raise UniProtError(f"UniProt accession not found: {accession}")
    response.raise_for_status()
    return parse_fasta_record(response.text, source="UniProt", identifier=accession)


@st.cache_data(show_spinner=False)
def search_uniprot(query: str, size: int = 10, reviewed: Optional[bool] = None, timeout: int = 30) -> List[SequenceRecord]:
    q = query.strip()
    if not q:
        raise ValueError("Query cannot be empty")

    if reviewed is True:
        q = f"({q}) AND reviewed:true"
    elif reviewed is False:
        q = f"({q}) AND reviewed:false"

    params = {
        "query": q,
        "format": "fasta",
        "size": size,
    }
    response = requests.get(f"{UNIPROT_BASE}/uniprotkb/search", params=params, timeout=timeout)
    response.raise_for_status()

    records = parse_fasta_records(response.text, source="UniProtSearch")
    if not records:
        raise UniProtError(f"No UniProt matches for query: {query}")
    return records


def parse_fasta_record(fasta_text: str, source: str, identifier: str) -> SequenceRecord:
    records = parse_fasta_records(fasta_text, source=source)
    if not records:
        raise ValueError("No FASTA record found")
    record = records[0]
    if identifier:
        record.identifier = identifier
    return record


def parse_fasta_records(fasta_text: str, source: str) -> List[SequenceRecord]:
    chunks = [chunk.strip() for chunk in fasta_text.strip().split(">") if chunk.strip()]
    records: List[SequenceRecord] = []
    for chunk in chunks:
        lines = chunk.splitlines()
        header = lines[0].strip()
        seq = clean_sequence("\n".join(lines[1:]))
        identifier = header.split()[0]
        records.append(
            SequenceRecord(
                source=source,
                identifier=identifier,
                description=header,
                sequence=seq,
            )
        )
    return records


def build_aligner(mode: str = "local") -> Align.PairwiseAligner:
    mode = mode.lower()
    if mode not in {"local", "global"}:
        raise ValueError("mode must be 'local' or 'global'")

    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def _is_positive(a: str, b: str, matrix) -> bool:
    if a == "-" or b == "-":
        return False
    if a == b:
        return True
    try:
        return matrix[a, b] > 0
    except Exception:
        return False


def _middle_char(a: str, b: str, matrix) -> str:
    if a == "-" or b == "-":
        return " "
    if a == b:
        return "|"
    if _is_positive(a, b, matrix):
        return "+"
    return " "


def _aligned_strings_from_alignment(alignment, query_seq: str, subject_seq: str) -> Tuple[str, str]:
    query_parts = []
    subject_parts = []

    q_blocks = alignment.coordinates[0]
    s_blocks = alignment.coordinates[1]

    for i in range(len(q_blocks) - 1):
        q_start, q_end = q_blocks[i], q_blocks[i + 1]
        s_start, s_end = s_blocks[i], s_blocks[i + 1]

        q_len = q_end - q_start
        s_len = s_end - s_start

        if q_len == s_len:
            query_parts.append(query_seq[q_start:q_end])
            subject_parts.append(subject_seq[s_start:s_end])
        elif q_len > 0 and s_len == 0:
            query_parts.append(query_seq[q_start:q_end])
            subject_parts.append("-" * q_len)
        elif s_len > 0 and q_len == 0:
            query_parts.append("-" * s_len)
            subject_parts.append(subject_seq[s_start:s_end])
        else:
            raise RuntimeError(
                f"Unexpected alignment block: query {q_start}-{q_end}, subject {s_start}-{s_end}"
            )

    return "".join(query_parts), "".join(subject_parts)


def align_sequences(query: SequenceRecord, subject: SequenceRecord, mode: str = "local") -> AlignmentResult:
    aligner = build_aligner(mode)
    alignment = aligner.align(query.sequence, subject.sequence)[0]
aligned_query, aligned_subject = _aligned_strings_from_alignment(
    alignment,
    query.sequence,
    subject.sequence,)

    matrix = aligner.substitution_matrix
    middle = "".join(_middle_char(a, b, matrix) for a, b in zip(aligned_query, aligned_subject))

    identities = sum(1 for a, b in zip(aligned_query, aligned_subject) if a == b and a != "-")
    positives = sum(1 for a, b in zip(aligned_query, aligned_subject) if _is_positive(a, b, matrix))
    gaps = sum(1 for a, b in zip(aligned_query, aligned_subject) if a == "-" or b == "-")
    aligned_pairs = sum(1 for a, b in zip(aligned_query, aligned_subject) if a != "-" and b != "-")
    mismatches = aligned_pairs - identities

    alignment_length = len(aligned_query)
    query_covered = sum(1 for c in aligned_query if c != "-")
    subject_covered = sum(1 for c in aligned_subject if c != "-")

    stats = AlignmentStats(
        alignment_type=mode,
        score=float(alignment.score),
        alignment_length=alignment_length,
        identities=identities,
        positives=positives,
        mismatches=mismatches,
        gaps=gaps,
        query_coverage=100.0 * query_covered / len(query.sequence),
        subject_coverage=100.0 * subject_covered / len(subject.sequence),
        percent_identity=100.0 * identities / alignment_length if alignment_length else 0.0,
        percent_positive=100.0 * positives / alignment_length if alignment_length else 0.0,
    )

    return AlignmentResult(
        query=query,
        subject=subject,
        aligned_query=aligned_query,
        aligned_subject=aligned_subject,
        middle_line=middle,
        stats=stats,
    )


def format_alignment(result: AlignmentResult, width: int = 60) -> str:
    lines: List[str] = []
    s = result.stats
    lines.append(f"Alignment type: {s.alignment_type}")
    lines.append(f"Score: {s.score:.1f}")
    lines.append(
        f"Identities: {s.identities}/{s.alignment_length} ({s.percent_identity:.2f}%), "
        f"Positives: {s.positives}/{s.alignment_length} ({s.percent_positive:.2f}%), "
        f"Gaps: {s.gaps}/{s.alignment_length} ({100.0 * s.gaps / s.alignment_length:.2f}%)"
    )
    lines.append(
        f"Query cover: {s.query_coverage:.2f}% | Subject cover: {s.subject_coverage:.2f}%"
    )
    lines.append("")

    q_pos = 1
    s_pos = 1
    aq = result.aligned_query
    as_ = result.aligned_subject
    mid = result.middle_line

    for i in range(0, len(aq), width):
        q_chunk = aq[i:i + width]
        s_chunk = as_[i:i + width]
        m_chunk = mid[i:i + width]

        q_non_gap = sum(1 for c in q_chunk if c != "-")
        s_non_gap = sum(1 for c in s_chunk if c != "-")
        q_end = q_pos + q_non_gap - 1 if q_non_gap else q_pos - 1
        s_end = s_pos + s_non_gap - 1 if s_non_gap else s_pos - 1

        lines.append(f"Query   {q_pos:>5}  {q_chunk}  {q_end:>5}")
        lines.append(f"               {m_chunk}")
        lines.append(f"Sbjct   {s_pos:>5}  {s_chunk}  {s_end:>5}")
        lines.append("")

        q_pos = q_end + 1
        s_pos = s_end + 1

    return "\n".join(lines)


def resolve_sequence(input_value: str, kind: str, reviewed_only: bool = True) -> SequenceRecord:
    if kind == "accession":
        return fetch_uniprot_by_accession(input_value)
    if kind == "query":
        return search_uniprot(input_value, size=1, reviewed=reviewed_only)[0]
    if kind == "sequence":
        sequence = clean_sequence(input_value)
        return SequenceRecord(source="input", identifier="custom", description="custom input", sequence=sequence)
    raise ValueError("kind must be accession, query, or sequence")


def preview_search_results(query: str, reviewed_only: bool) -> List[SequenceRecord]:
    return search_uniprot(query, size=5, reviewed=reviewed_only)


def input_panel(title: str, prefix: str) -> Tuple[str, str, bool]:
    st.subheader(title)
    kind = st.radio(
        "Input type",
        options=["accession", "query", "sequence"],
        horizontal=True,
        key=f"{prefix}_kind",
    )

    if kind == "accession":
        value = st.text_input(
            "UniProt accession",
            placeholder="Example: P69905",
            key=f"{prefix}_value",
        )
    elif kind == "query":
        value = st.text_area(
            "UniProt search query",
            placeholder="Example: gene:HBB AND organism_id:9606",
            height=100,
            key=f"{prefix}_value",
        )
    else:
        value = st.text_area(
            "Protein sequence or FASTA",
            placeholder=">example\nMVLSPADKTNVKAA...",
            height=180,
            key=f"{prefix}_value",
        )

    reviewed_only = st.checkbox(
        "Reviewed entries only (Swiss-Prot)",
        value=True,
        key=f"{prefix}_reviewed",
        disabled=(kind != "query"),
    )

    return kind, value, reviewed_only


def render_record_card(record: SequenceRecord, label: str) -> None:
    st.markdown(f"**{label}**")
    st.caption(record.description)
    st.code(record.sequence[:1200] + ("..." if len(record.sequence) > 1200 else ""), language=None)
    st.write(f"Length: {len(record.sequence)} aa")


def main() -> None:
    st.set_page_config(page_title="Protein Homology Tool", layout="wide")
    st.title("Protein Homology Tool")
    st.write(
        "Compare two protein sequences using UniProt retrieval plus BLAST-like pairwise alignment output. "
        "Use local alignment for best-region matching or global alignment for full-length comparison."
    )

    with st.sidebar:
        st.header("Alignment settings")
        mode = st.selectbox(
            "Alignment mode",
            options=["local", "global"],
            index=0,
            help="Local is more BLAST-like. Global compares full sequence lengths.",
        )
        width = st.slider("Alignment line width", min_value=40, max_value=120, value=60, step=10)
        st.markdown("---")
        st.markdown(
            "**Notes**\n"
            "- Percent homology is usually reported as percent identity or similarity.\n"
            "- This app performs pairwise alignment, not a full BLAST database search."
        )

    left, right = st.columns(2)
    with left:
        query_kind, query_value, query_reviewed = input_panel("Query protein", "query")
    with right:
        subject_kind, subject_value, subject_reviewed = input_panel("Subject protein", "subject")

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        if query_kind == "query" and query_value.strip() and st.button("Preview query matches"):
            try:
                records = preview_search_results(query_value, query_reviewed)
                for i, rec in enumerate(records, start=1):
                    st.markdown(f"**{i}. {rec.identifier}**")
                    st.caption(rec.description)
            except Exception as exc:
                st.error(str(exc))

    with preview_col2:
        if subject_kind == "query" and subject_value.strip() and st.button("Preview subject matches"):
            try:
                records = preview_search_results(subject_value, subject_reviewed)
                for i, rec in enumerate(records, start=1):
                    st.markdown(f"**{i}. {rec.identifier}**")
                    st.caption(rec.description)
            except Exception as exc:
                st.error(str(exc))

    if st.button("Run alignment", type="primary"):
        try:
            if not query_value.strip() or not subject_value.strip():
                raise ValueError("Both query and subject inputs are required.")

            with st.spinner("Resolving sequences and running alignment..."):
                query_record = resolve_sequence(query_value, query_kind, reviewed_only=query_reviewed)
                subject_record = resolve_sequence(subject_value, subject_kind, reviewed_only=subject_reviewed)
                result = align_sequences(query_record, subject_record, mode=mode)
                alignment_text = format_alignment(result, width=width)

            st.success("Alignment complete")

            info_left, info_right = st.columns(2)
            with info_left:
                render_record_card(query_record, "Query")
            with info_right:
                render_record_card(subject_record, "Subject")

            s = result.stats
            metric_cols = st.columns(6)
            metric_cols[0].metric("% Identity", f"{s.percent_identity:.2f}%")
            metric_cols[1].metric("% Positive", f"{s.percent_positive:.2f}%")
            metric_cols[2].metric("Gaps", str(s.gaps))
            metric_cols[3].metric("Alignment length", str(s.alignment_length))
            metric_cols[4].metric("Query coverage", f"{s.query_coverage:.2f}%")
            metric_cols[5].metric("Subject coverage", f"{s.subject_coverage:.2f}%")

            st.subheader("BLAST-like alignment")
            st.code(alignment_text, language=None)

            st.download_button(
                "Download alignment report",
                data=alignment_text,
                file_name="protein_alignment.txt",
                mime="text/plain",
            )

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
