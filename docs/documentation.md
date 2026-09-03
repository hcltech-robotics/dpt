# Build the documentation

The manual uses MkDocs Material and reads the Python API from source. Python 3.12 or later is sufficient to build it.

From the repository root:

```bash
python -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install --require-hashes -r docs/requirements.txt
python -m mkdocs build --strict
```

The rendered manual is written to `site/`. Changes on `main` publish to [the online manual](https://hcltech-robotics.github.io/dpt/) through GitHub Actions. Pull requests build a downloadable preview.

To preview it locally:

```bash
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

The build reads `CITATION.cff` for the title, author and version, and includes both root citation files as downloads. Geist Sans, Geist Mono and MathJax are bundled with the manual. Their licences and checksums are in `docs/assets/vendor/`.

Use `$...$` or `\(...\)` for inline mathematics, and `$$...$$` or `\[...\]` on separate lines for displayed equations. The API pages use static source analysis, with all build dependencies listed in `docs/requirements.txt`.
