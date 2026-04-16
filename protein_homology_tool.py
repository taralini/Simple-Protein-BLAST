from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
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
    entry_name: str = ""
    protein_name: str = ""
    organism: str = ""
    reviewed: Optional[bool] = None
    length: Optional[int] = None


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


def _parse_fasta_header(header: str) -> Tuple[str, str, str, str, bool]:
    accession = ""
    entry_name = ""
    protein_name = ""
    organism = ""
    reviewed = None

    parts = header.split("|", 2)
    if len(parts) >= 3:
        db_code, accession, rest = parts[0], parts[1], parts[2]
        reviewed = db_code == "sp"
        first_token, _, remainder = rest.partition(" ")
        entry_name = first_token.strip()
        protein_name = remainder.split(" OS=")[0].strip() if remainder else ""
        os_match = re.search(r" OS=(.+?)(?: OX=| GN=| PE=| SV=|$)", remainder)
        if os_match:
            organism = os_match.group(1).strip()
    else:
        accession = header.split()[0]
        protein_name = header

    return accession, entry_name, protein_name, organism, reviewed


def fetch_uniprot_by_accession(accession: str, timeout: int = 30) -> SequenceRecord:
    accession = accession.strip()
    fasta_url = f"{UNIPROT_BASE}/uniprotkb/{accession}.fasta"
    response = requests.get(fasta_url, timeout=timeout)
    if response.status_code == 404:
        raise UniProtError(f"UniProt accession not found: {accession}")
    response.raise_for_status()
    record = parse_fasta_record(response.text, source="UniProt", identifier=accession)
    return record


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
        accession, entry_name, protein_name, organism, reviewed = _parse_fasta_header(header)
        identifier = accession or header.split()[0]
        records.append(
            SequenceRecord(
                source=source,
                identifier=identifier,
                description=header,
                sequence=seq,
                entry_name=entry_name,
                protein_name=protein_name,
                organism=organism,
                reviewed=reviewed,
                length=len(seq),
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
    query_parts: List[str] = []
    subject_parts: List[str] = []

    q_blocks = alignment.coordinates[0]
    s_blocks = alignment.coordinates[1]

    for i in range(len(q_blocks) - 1):
        q_start, q_end = int(q_blocks[i]), int(q_blocks[i + 1])
        s_start, s_end = int(s_blocks[i]), int(s_blocks[i + 1])

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
        subject.sequence,
    )

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


def build_uniprot_query(
    accession: str = "",
    gene: str = "",
    species: str = "",
    protein_name: str = "",
    reviewed_only: bool = True,
    isoform_only: bool = False,
) -> str:
    parts: List[str] = []

    accession = accession.strip()
    gene = gene.strip()
    species = species.strip()
    protein_name = protein_name.strip()

    if accession:
        parts.append(f"accession:{accession}")
    if gene:
        parts.append(f"gene:{gene}")
    if species:
        if species.isdigit():
            parts.append(f"organism_id:{species}")
        else:
            parts.append(f'organism_name:"{species}"')
    if protein_name:
        parts.append(f'protein_name:"{protein_name}"')
    if reviewed_only:
        parts.append("reviewed:true")
    if isoform_only:
        parts.append("is_isoform:true")

    if not parts:
        raise ValueError("Provide at least one search field or use accession/raw sequence mode.")

    return " AND ".join(parts)


def resolve_sequence(input_value: str, kind: str, reviewed_only: bool = True) -> SequenceRecord:
    if kind == "accession":
        return fetch_uniprot_by_accession(input_value)
    if kind == "query":
        return search_uniprot(input_value, size=1, reviewed=reviewed_only)[0]
    if kind == "sequence":
        sequence = clean_sequence(input_value)
        return SequenceRecord(
            source="input",
            identifier="custom",
            description="custom input",
            sequence=sequence,
            protein_name="custom input",
            organism="",
            reviewed=None,
            length=len(sequence),
        )
    raise ValueError("kind must be accession, query, or sequence")


def render_record_card(record: SequenceRecord, label: str) -> None:
    st.markdown(f"**{label}**")
    if record.protein_name:
        st.write(record.protein_name)
    meta = []
    if record.identifier:
        meta.append(f"Accession: {record.identifier}")
    if record.entry_name:
        meta.append(f"Entry: {record.entry_name}")
    if record.organism:
        meta.append(f"Organism: {record.organism}")
    if record.reviewed is not None:
        meta.append("Reviewed" if record.reviewed else "Unreviewed")
    if record.length:
        meta.append(f"Length: {record.length} aa")
    if meta:
        st.caption(" | ".join(meta))
    st.code(record.sequence[:1200] + ("..." if len(record.sequence) > 1200 else ""), language=None)


def search_selector_panel(title: str, prefix: str) -> Tuple[str, str, bool, Optional[SequenceRecord]]:
    st.subheader(title)
    mode = st.radio(
        "Input mode",
        options=["Structured search", "Accession", "Raw sequence", "Advanced query"],
        horizontal=True,
        key=f"{prefix}_mode",
    )

    selected_record: Optional[SequenceRecord] = None
    resolved_kind = ""
    resolved_value = ""
    reviewed_only = True

    if mode == "Structured search":
        gene = st.text_input("Gene", key=f"{prefix}_gene", placeholder="TP53")
        species = st.text_input("Species or taxonomy ID", key=f"{prefix}_species", placeholder="Homo sapiens or 9606")
        protein_name = st.text_input("Protein name (optional)", key=f"{prefix}_protein", placeholder="cellular tumor antigen p53")
        reviewed_only = st.checkbox("Reviewed only", value=True, key=f"{prefix}_reviewed_structured")
        isoform_only = st.checkbox("Isoform entries only", value=False, key=f"{prefix}_isoform")
        max_hits = st.slider("Search results", min_value=3, max_value=10, value=5, key=f"{prefix}_max_hits")

        search_clicked = st.button(f"Search {title}", key=f"{prefix}_search_button")
        if search_clicked:
            try:
                query = build_uniprot_query(
                    gene=gene,
                    species=species,
                    protein_name=protein_name,
                    reviewed_only=reviewed_only,
                    isoform_only=isoform_only,
                )
                records = search_uniprot(query, size=max_hits, reviewed=None)
                st.session_state[f"{prefix}_records"] = records
                st.session_state[f"{prefix}_query_string"] = query
            except Exception as exc:
                st.error(str(exc))

        records = st.session_state.get(f"{prefix}_records", [])
        if records:
            options = []
            for idx, rec in enumerate(records):
                label = f"{idx + 1}. {rec.identifier} | {rec.organism or 'Unknown organism'} | {rec.protein_name or rec.description[:50]}"
                options.append(label)

            selected_label = st.selectbox(
                "Choose a UniProt hit",
                options=options,
                key=f"{prefix}_selectbox",
            )
            selected_index = options.index(selected_label)
            selected_record = records[selected_index]
            resolved_kind = "accession"
            resolved_value = selected_record.identifier

            df = pd.DataFrame(
                [
                    {
                        "Accession": rec.identifier,
                        "Entry": rec.entry_name,
                        "Protein": rec.protein_name,
                        "Organism": rec.organism,
                        "Reviewed": "Yes" if rec.reviewed else "No",
                        "Length": rec.length,
                    }
                    for rec in records
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif mode == "Accession":
        resolved_kind = "accession"
        resolved_value = st.text_input("UniProt accession", key=f"{prefix}_accession", placeholder="P04637")
        reviewed_only = True

    elif mode == "Raw sequence":
        resolved_kind = "sequence"
        resolved_value = st.text_area(
            "Protein sequence or FASTA",
            key=f"{prefix}_sequence",
            placeholder=">example\nMEEPQSDPSVEPPLSQETFSDLWKLL...",
            height=180,
        )
        reviewed_only = True

    else:
        resolved_kind = "query"
        resolved_value = st.text_area(
            "UniProt query",
            key=f"{prefix}_advanced_query",
            placeholder="gene:TP53 AND organism_id:9606 AND reviewed:true",
            height=100,
        )
        reviewed_only = st.checkbox("Reviewed only", value=True, key=f"{prefix}_reviewed_advanced")

    return resolved_kind, resolved_value, reviewed_only, selected_record


def main() -> None:
    st.set_page_config(page_title="Protein Homology Tool", layout="wide")
    st.title("Protein Homology Tool - Version 2 Draft")
    st.write(
        "Compare proteins across species, within-species variants, and candidate orthologs using UniProt retrieval "
        "and BLAST-like pairwise alignment output."
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
            "**Suggested usage**\n"
            "- Across species orthologs: start with local and inspect coverage.\n"
            "- Isoforms or variants within one species: global is often more informative."
        )

    left, right = st.columns(2)
    with left:
        query_kind, query_value, query_reviewed, query_selected = search_selector_panel("Query protein", "query")
    with right:
        subject_kind, subject_value, subject_reviewed, subject_selected = search_selector_panel("Subject protein", "subject")

    if st.button("Run alignment", type="primary"):
        try:
            if not query_value.strip() or not subject_value.strip():
                raise ValueError("Both query and subject inputs are required.")

            with st.spinner("Resolving sequences and running alignment..."):
                query_record = query_selected or resolve_sequence(query_value, query_kind, reviewed_only=query_reviewed)
                subject_record = subject_selected or resolve_sequence(subject_value, subject_kind, reviewed_only=subject_reviewed)
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

            summary_df = pd.DataFrame(
                [
                    {
                        "Query accession": query_record.identifier,
                        "Query organism": query_record.organism,
                        "Subject accession": subject_record.identifier,
                        "Subject organism": subject_record.organism,
                        "Mode": s.alignment_type,
                        "% Identity": round(s.percent_identity, 2),
                        "% Positive": round(s.percent_positive, 2),
                        "Query coverage": round(s.query_coverage, 2),
                        "Subject coverage": round(s.subject_coverage, 2),
                        "Score": round(s.score, 2),
                    }
                ]
            )
            st.subheader("Comparison summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.subheader("BLAST-like alignment")
            st.code(alignment_text, language=None)

            st.download_button(
                "Download alignment report",
                data=alignment_text,
                file_name="protein_alignment.txt",
                mime="text/plain",
            )
            st.download_button(
                "Download summary CSV",
                data=summary_df.to_csv(index=False),
                file_name="protein_alignment_summary.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
