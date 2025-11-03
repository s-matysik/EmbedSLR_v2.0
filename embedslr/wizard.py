"""
EmbedSLR – Terminal Wizard (local)
==================================
Interactive wizard for running EmbedSLR in a local environment.
Supports both single-model and multi-model analysis.
"""

from __future__ import annotations

import os
import sys
import zipfile
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ────────── helper functions  ─────────────────
def _env_var(provider: str) -> str | None:
    """Returns the ENV variable name for the API key of the given provider."""
    return {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina":   "JINA_API_KEY",
        "nomic":  "NOMIC_API_KEY",
    }.get(provider.lower())


def _ensure_sbert_installed() -> None:
    """
    Ensures the *sentence‑transformers* library is available.
    • If missing, prompts the user and installs it (`pip install --user sentence-transformers`).
    """
    try:
        importlib.import_module("sentence_transformers")
    except ModuleNotFoundError:
        ans = _ask(
            "📦  Missing library 'sentence-transformers'. Install now? (y/N)",
            "N",
        ).lower()
        if ans == "y":
            print("⏳  Installing 'sentence-transformers'…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--user", "--quiet", "sentence-transformers"]
            )
            print("✅  Installation complete.\n")
        else:
            sys.exit("❌  The 'sbert' provider requires the 'sentence-transformers' library.")


def _ensure_matplotlib_installed() -> None:
    """Ensures matplotlib is available for visualizations."""
    try:
        importlib.import_module("matplotlib")
    except ModuleNotFoundError:
        ans = _ask(
            "📦  Missing library 'matplotlib' (required for visualizations). Install now? (y/N)",
            "N",
        ).lower()
        if ans == "y":
            print("⏳  Installing 'matplotlib'…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--user", "--quiet", "matplotlib"]
            )
            print("✅  Installation complete.\n")
        else:
            print("⚠️  Visualizations will not be available without matplotlib.")


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures presence of columns:
      • Title
      • Author Keywords
      • Parsed_References  (set[str])
    """
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


# ────────── local‑model utilities for SBERT ──────────────────────────────
def _local_model_dir(model_name: str) -> Path:
    """
    Returns a path where the given SBERT model should live inside the project
    (…/embedslr/sbert_models/<model_name_with__>).
    """
    safe = model_name.replace("/", "__")
    base = Path(__file__).resolve().parent / "sbert_models"
    return base / safe


def _get_or_download_local_sbert(model_name: str) -> Path:
    """
    Ensures that *model_name* is present in the project folder and returns its path.
    If missing – downloads it once and saves permanently.
    """
    local_dir = _local_model_dir(model_name)
    if local_dir.exists():
        print(f"✅  Local model found: {local_dir}")
    else:
        print(f"⏳  Downloading model '{model_name}' do '{local_dir}' …")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(model_name).save(str(local_dir))
        print("✅  Model downloaded and saved.\n")
    # wymuszenie trybu offline dla HuggingFace Hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return local_dir


# ────────── core pipelines ────────────────────────────────────────────────
def _pipeline(
    df: pd.DataFrame,
    query: str,
    provider: str,
    model: str,
    out: Path,
    top_n: int | None,
) -> Path:
    """
    Original single-model pipeline.
    Executes the full EmbedSLR workflow and returns the path to the ZIP of results.
    """
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    df = _ensure_aux_columns(df.copy())

    # 1. Prepare text for embedding
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    # 2. Embeddings
    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]

    # 3. Ranking
    ranked = rank_by_cosine(qvec, vecs, df)

    # 4. Save ranking.csv
    out.mkdir(parents=True, exist_ok=True)
    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    # 5. Top‑N (optional)
    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    # 6. Full bibliometric report
    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)

    # 7. ZIP with results
    zf = out / "embedslr_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p_all, "ranking.csv")
        if p_top:
            z.write(p_top, "topN.csv")
        z.write(rep, "biblio_report.txt")
    return zf


def _multi_pipeline(
    df: pd.DataFrame,
    query: str,
    models_config: List[Dict[str, str]],
    out: Path,
    top_n: int = 17
) -> Path:
    """
    Multi-model pipeline for agent-based systematic literature review.
    """
    from .multi_embedding import multi_model_analysis
    
    df = _ensure_aux_columns(df.copy())
    
    # Run multi-model analysis
    results = multi_model_analysis(
        df=df,
        query=query,
        models_config=models_config,
        top_n=top_n,
        output_dir=out,
    )
    
    # Save model rankings
    rankings_path = out / "model_rankings.csv"
    results["model_rankings"].to_csv(rankings_path, index=False)
    
    # Save hierarchical analysis
    hier_path = out / "hierarchical_analysis.csv"
    results["hierarchical_analysis"].to_csv(hier_path, index=False)
    
    # Save publications grouped by consensus
    groups_dir = out / "consensus_groups"
    groups_dir.mkdir(exist_ok=True)
    
    for n_models, group_df in results["hierarchical_groups"].items():
        group_path = groups_dir / f"selected_by_{n_models}_models.csv"
        group_df.to_csv(group_path, index=False)
    
    # Save individual model selections
    models_dir = out / "model_selections"
    models_dir.mkdir(exist_ok=True)
    
    for model_name, df_sel in results["model_selections"].items():
        safe_name = model_name.replace("::", "_").replace("/", "_")
        model_path = models_dir / f"{safe_name}.csv"
        df_sel.to_csv(model_path, index=False)
    
    # Create comprehensive report
    report_lines = [
        "=" * 80,
        "MULTI-MODEL SYSTEMATIC LITERATURE REVIEW REPORT",
        "=" * 80,
        "",
        f"Query: {query}",
        f"Number of models: {len(models_config)}",
        f"Top N per model: {top_n}",
        "",
        "=" * 80,
        "MODEL RANKINGS",
        "=" * 80,
        "",
        results["model_rankings"].to_string(index=False),
        "",
        "=" * 80,
        "HIERARCHICAL GROUP ANALYSIS",
        "=" * 80,
        "",
        results["hierarchical_analysis"].to_string(index=False),
        "",
        "=" * 80,
        "PUBLICATIONS BY CONSENSUS LEVEL",
        "=" * 80,
        "",
    ]
    
    for n_models in sorted(results["hierarchical_groups"].keys(), reverse=True):
        count = len(results["hierarchical_groups"][n_models])
        report_lines.append(f"Selected by {n_models} model(s): {count} publications")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("INTERPRETATION GUIDELINES")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("• Publications selected by MORE models (≥3): Most thematically coherent")
    report_lines.append("• Publications selected by FEWER models (1-2): Unique perspectives or edge cases")
    report_lines.append("• Optimal number of models: 4 (balances diversity and coherence)")
    report_lines.append("")
    
    report_path = out / "multi_model_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # Create ZIP with all results
    zf = out / "embedslr_multi_results.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        # Add main files
        z.write(rankings_path, "model_rankings.csv")
        z.write(hier_path, "hierarchical_analysis.csv")
        z.write(report_path, "multi_model_report.txt")
        
        # Add consensus groups
        for n_models in results["hierarchical_groups"].keys():
            group_path = groups_dir / f"selected_by_{n_models}_models.csv"
            z.write(group_path, f"consensus_groups/selected_by_{n_models}_models.csv")
        
        # Add model selections
        for model_name in results["model_selections"].keys():
            safe_name = model_name.replace("::", "_").replace("/", "_")
            model_path = models_dir / f"{safe_name}.csv"
            z.write(model_path, f"model_selections/{safe_name}.csv")
        
        # Add visualizations
        for chart_path in results["radar_charts"]:
            rel_path = chart_path.relative_to(out)
            z.write(chart_path, str(rel_path))
        
        for chart_path in results["comparison_charts"]:
            rel_path = chart_path.relative_to(out)
            z.write(chart_path, str(rel_path))
    
    return zf


# ────────── simple CLI ────────────────────────────────────────────────────
def _ask(prompt: str, default: Optional[str] = None) -> str:
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    ans = input(msg).strip()
    return ans or (default or "")


def _select_provider() -> str:
    provs = list(_models())
    print("📜  Available providers:", ", ".join(provs))
    return _ask("Provider", provs[0])


def _select_model(provider: str) -> str:
    mods = _models()[provider]
    print(f"📜  Models for {provider} (first 20):")
    for m in mods[:20]:
        print("   •", m)
    return _ask("Model", mods[0])


def run(save_dir: str | os.PathLike | None = None):
    """
    Runs the EmbedSLR wizard in terminal/screen/tmux.
    Now supports both single-model and multi-model modes.
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "EmbedSLR Wizard (local)")
    print("=" * 60 + "\n")

    # Input file
    csv_path = Path(_ask("📄  Path to CSV file")).expanduser()
    if not csv_path.exists():
        sys.exit(f"❌  File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅  Loaded {len(df)} records\n")

    # Select mode
    print("🔧 Mode Selection:")
    print("   • single - Traditional single-model analysis")
    print("   • multi  - Multi-model agent-based analysis (recommended)")
    mode = _ask("Mode", "single").strip().lower()
    
    # Analysis parameters
    query = _ask("\n❓  Research query").strip()
    
    if mode == "multi":
        print("\n" + "=" * 60)
        print(" " * 10 + "MULTI-MODEL CONFIGURATION")
        print("=" * 60)
        
        # Check matplotlib
        _ensure_matplotlib_installed()
        
        n_models_raw = _ask("\n🔢  Number of models to use", "4")
        n_models = int(n_models_raw) if n_models_raw else 4
        
        if n_models < 2:
            print("⚠️  Multi-model requires at least 2 models. Setting to 2.")
            n_models = 2
        elif n_models > 8:
            print("⚠️  Using >8 models may be excessive. Consider 4 (recommended).")
        
        models_config = []
        
        for i in range(n_models):
            print(f"\n{'─' * 60}")
            print(f"    Model {i+1}/{n_models} Configuration")
            print('─' * 60)
            
            provider = _select_provider()
            
            # SBERT prerequisites
            if provider.lower() == "sbert":
                _ensure_sbert_installed()
            
            model_name = _select_model(provider)
            
            # For SBERT – ensure permanent local copy
            if provider.lower() == "sbert":
                model_path = _get_or_download_local_sbert(model_name)
                model = str(model_path)
            else:
                model = model_name
            
            # API key (if needed)
            key_env = _env_var(provider)
            if key_env and not os.getenv(key_env):
                key = _ask(f"🔑  {key_env} (ENTER = skip)")
                if key:
                    os.environ[key_env] = key
            
            models_config.append({
                "provider": provider,
                "model": model
            })
            
            print(f"✅  Model {i+1} configured: {provider}::{model_name}")
        
        n_raw = _ask("\n🔢  Top‑N publications per model", "17")
        top_n = int(n_raw) if n_raw else 17
        
        # Output folder
        out_dir = Path(save_dir or os.getcwd()).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        print("\n" + "=" * 60)
        print(" " * 10 + "EXECUTING MULTI-MODEL ANALYSIS")
        print("=" * 60)
        print("⏳  This may take several minutes...")
        print(f"    Processing {len(df)} publications with {n_models} models\n")
        
        zip_path = _multi_pipeline(
            df=df,
            query=query,
            models_config=models_config,
            out=out_dir,
            top_n=top_n,
        )
        
        print("\n" + "=" * 60)
        print(" " * 15 + "✅  ANALYSIS COMPLETE")
        print("=" * 60)
        print("\n📊  Results include:")
        print("   • Model performance rankings")
        print("   • Hierarchical publication groups (by model consensus)")
        print("   • Radar charts for each model")
        print("   • Comparison charts")
        print("   • Comprehensive multi-model report")
        
    else:
        # Single-model mode (original)
        print("\n" + "=" * 60)
        print(" " * 10 + "SINGLE-MODEL CONFIGURATION")
        print("=" * 60)
        
        provider = _select_provider()

        # SBERT prerequisites
        if provider.lower() == "sbert":
            _ensure_sbert_installed()

        # Model (prompt only ONCE)
        model_name = _select_model(provider)

        # For SBERT – ensure permanent local copy & switch to its path
        if provider.lower() == "sbert":
            model_path = _get_or_download_local_sbert(model_name)
            model = str(model_path)
        else:
            model = model_name

        n_raw = _ask("🔢  Top‑N publications for metrics (ENTER = all)")
        top_n = int(n_raw) if n_raw else None

        # API key (if needed)
        key_env = _env_var(provider)
        if key_env and not os.getenv(key_env):
            key = _ask(f"🔑  {key_env} (ENTER = skip)")
            if key:
                os.environ[key_env] = key

        # Output folder
        out_dir = Path(save_dir or os.getcwd()).absolute()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        print("\n⏳  Processing…")
        zip_path = _pipeline(
            df=df,
            query=query,
            provider=provider,
            model=model,
            out=out_dir,
            top_n=top_n,
        )

        print("\n✅  Done!")
    
    # Final output info (common for both modes)
    print("\n📁  Results saved to:", out_dir)
    print("🎁  ZIP package:", zip_path)
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run()
