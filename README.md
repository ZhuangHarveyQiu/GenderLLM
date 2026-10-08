# Data and code for Delegating interpretive authority

This repository contains the human annotation data, large language model outputs, analysis code, and analytical results supporting the manuscript **“Delegating interpretive authority: human annotators and large language models judging implicit gender bias in news.”**

The study compares a fixed panel of 30 human annotators and six large language models classifying 57 sentences from three British news articles. The materials use English sentences with Chinese translations. Each model annotated each article three times, yielding 54 model calls.

## Accessing the files

**The repository contains data files.** On Anonymous GitHub, the opening page may display “Unavailable,” “This content can’t be shown,” or “Please select a file” until a file is selected. This message refers to the file preview pane.

1. Open the **Files** panel if the file list is hidden.
2. Expand `RawHumanData`, `RawLLMData`, or `DataAnalysis`.
3. Select a file to display its contents. Use the raw-view or download control if the preview is unsuitable for the file format.
4. To obtain all materials, use **Download repository as ZIP**, then extract the archive.

Direct access:

- [Repository home](https://anonymous.4open.science/r/GenderLLM-C10F/)
- [Individual human annotation data](https://anonymous.4open.science/r/GenderLLM-C10F/RawHumanData/HumanDataRaw.csv)
- [Aggregated human annotation data](https://anonymous.4open.science/r/GenderLLM-C10F/RawHumanData/HumanAnnotation.csv)
- [Parsed model annotations](https://anonymous.4open.science/r/GenderLLM-C10F/RawLLMData/parsed_annotations.csv)
- [Raw model responses](https://anonymous.4open.science/r/GenderLLM-C10F/RawLLMData/raw_responses.jsonl)
- [Analysis code](https://anonymous.4open.science/r/GenderLLM-C10F/DataAnalysis/analysis_pipeline_v5.py)
- [Download the complete repository as ZIP](https://anonymous.4open.science/api/repo/GenderLLM-C10F/zip)

## Repository contents

| Path | Contents and purpose |
| --- | --- |
| `RawHumanData/HumanDataRaw.csv` | Individual responses for 57 sentences, with one sentence column and 30 annotator columns. This is the human-data input for the analysis pipeline. |
| `RawHumanData/HumanAnnotation.csv` | Sentence IDs, English sentences, Chinese translations, and aggregate human label counts. Used to cross-check the raw counts before cleaning. |
| `RawLLMData/raw_responses.jsonl` | Archived model responses and associated model, repetition, article, and time information; one JSON record per model call. |
| `RawLLMData/parsed_annotations.csv` | Parsed model annotations, including model identifiers, provider, repetition, article, sentence ID, labels, rationale, and time. This is the model-data input for the analysis pipeline. |
| `DataAnalysis/analysis_pipeline_v5.py` | Python script that cleans and consolidates annotations and generates statistical tables, sensitivity analyses, and Figure 1. |
| `DataAnalysis/output_v5/main/` | Main analytical tables, manuscript Tables 2–4, Figure 1, and the data underlying the figure. |
| `DataAnalysis/output_v5/supplementary/` | Supplementary analyses covering label composition, agreement, model repetition stability, sentence ordering, and sensitivity to panel composition. |
| `DataAnalysis/output_v5/analysis_report_v5.json` | Machine-readable summary of data-quality checks, analysis settings, principal estimates, and output filenames. |

Files named `.DS_Store` are operating-system metadata and are not used in the analysis.

## Annotation labels and identifiers

| Code | Category |
| --- | --- |
| `RB` | Representational Bias |
| `LEB` | Linguistic Evaluation Bias |
| `M` | Misgendering |
| `NB` | No Bias |

Multiple bias categories may be assigned to a sentence. `NB` is mutually exclusive with the bias categories. The manuscript and its Appendix provide the category definitions, decision rules, worked examples, and corpus materials.

The analysis identifies articles as `P1`, `P2`, and `P3`, containing 20, 19, and 18 sentences, respectively. Sentence numbers restart within each article; a key such as `P2_1` therefore identifies sentence 1 of article 2. The aggregate human file numbers sentences consecutively from 1 to 57. Annotator columns represent the same individuals across sentences; the pipeline assigns them identifiers `A1`–`A30`.

CSV files contain Chinese text and should be opened with UTF-8 encoding. The human files can be read using `utf-8-sig`. JSONL files contain one JSON object per line.
