# Data access and responsible publication

The research involved longitudinal ATTICA cohort data and potentially sensitive health information. This public repository **must not** contain the original workbook, participant-level extracts, processed feature matrices, record identifiers, or notebook outputs that display individual data.

The source archive included files under `data/raw`, `data/processed`, and `results`; **none** of these source files were copied here. Original Jupyter cell outputs, execution counts, embedded images, and nonessential metadata were removed from the notebooks.

If you have an independently authorised dataset, keep it local in `data/raw/`. Follow the data controller's access terms; `.gitignore` is not an adequate substitute for checking files staged in Git.

Before any public GitHub publication, verify that (1) your research supervisors/institution permit code distribution, (2) third-party software or borrowed code is appropriately credited, and (3) no confidential or personal information remains in sources or commit history.
