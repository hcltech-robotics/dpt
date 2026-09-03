"""Render package metadata, citation downloads and canonical agent skills."""

from __future__ import annotations

import re
import tomllib
from os.path import relpath
from pathlib import Path
from urllib.parse import quote, urlsplit

import yaml
from mkdocs.structure.files import File


def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    bibtex = (root / "CITATION.bib").read_text(encoding="utf-8").strip()
    author = citation["authors"][0]
    display_name = f"{author['given-names']} {author['family-names']}"
    initials = " ".join(f"{part[0]}." for part in author["given-names"].split())
    year = re.search(r"\byear\s*=\s*[\{\"](\d{4})[\}\"]", bibtex)
    if year is None:
        raise ValueError("CITATION.bib must contain a four-digit year")
    title = citation["title"]
    version = str(citation["version"])
    url = citation["repository-code"]
    text = (
        f"{author['family-names']}, {initials} ({year[1]}). "
        f"{title} (version {version}) [Computer software]. {url}"
    )
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if version != project["version"] or display_name != project["authors"][0]["name"]:
        raise ValueError("CITATION.cff author and version must match pyproject.toml")
    index = (root / "docs/index.md").read_text(encoding="utf-8")
    if not index.startswith(f"# {title}\n") or (
        f'{display_name} <span class="dpt-version">Version {version}</span>' not in index
    ):
        raise ValueError("The manual title, author and version must match CITATION.cff")
    for path in ("README.md", "docs/citation.md"):
        markdown = (root / path).read_text(encoding="utf-8")
        if f"```bibtex\n{bibtex}\n```" not in markdown:
            raise ValueError(f"The BibTeX block in {path} must match CITATION.bib")
    if text not in (root / "docs/citation.md").read_text(encoding="utf-8"):
        raise ValueError("The manual citation must match the root citation files")
    config.site_name = title
    config.site_author = display_name
    config.repo_url = config.repo_url or project["urls"]["Source"]
    config.extra["dpt"] = {
        "title": title,
        "author": display_name,
        "version": version,
        "citation": text,
        "bibtex": bibtex,
    }
    return config


def skill_markdown(source, destination, root, repo_url):
    def skill_link(match):
        parts = urlsplit(match[1])
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
            return match[0]
        resolved = (source.parent / parts.path).resolve()
        fragment = f"#{parts.fragment}" if parts.fragment else ""
        if resolved.name == "SKILL.md" and resolved.parent.parent == root / "skills":
            location = f"skills/{resolved.parent.name}.md"
        elif resolved.is_relative_to(root / "docs"):
            location = resolved.relative_to(root / "docs").as_posix()
        elif resolved.is_relative_to(root):
            location = f"{repo_url}/blob/main/{quote(resolved.relative_to(root).as_posix())}"
            return f"]({location}{fragment})"
        else:
            return match[0]
        location = relpath(location, Path(destination).parent).replace("\\", "/")
        return f"]({location}{fragment})"

    return re.sub(r"\]\(([^\s)]+)\)", skill_link, source.read_text(encoding="utf-8"))


def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    for name in ("CITATION.cff", "CITATION.bib"):
        files.append(File.generated(config, name, abs_src_path=str(root / name)))
    for source in sorted((root / "skills").glob("*/SKILL.md")):
        destination = f"skills/{source.parent.name}.md"
        markdown = skill_markdown(source, destination, root, config.repo_url)
        files.append(File.generated(config, destination, content=markdown))
    return files


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == "citation.md":
        for name in ("CITATION.cff", "CITATION.bib"):
            markdown = markdown.replace(f"(../{name})", f"({name})")
    return markdown
