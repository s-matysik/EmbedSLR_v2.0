from __future__ import annotations
import io, os, sys, tempfile, zipfile, shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from IPython.display import HTML, clear_output, display

IN_COLAB = "google.colab" in sys.modules


# helpers
def _env_var(p: str) -> str | None:
    return {"openai": "OPENAI_API_KEY", "cohere": "COHERE_API_KEY",
            "jina": "JINA_API_KEY", "nomic": "NOMIC_API_KEY"}.get(p.lower())


def _models() -> Dict[str, List[str]]:
    from .embeddings import list_models
    return list_models()


def _ensure_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy Parsed_References / Author Keywords jeżeli brak."""
    if "Parsed_References" not in df.columns:
        if "References" in df.columns:
            df["Parsed_References"] = df["References"].fillna("").apply(
                lambda x: {r.strip() for r in x.split(");") if r.strip()}
            )
        else:
            df["Parsed_References"] = [set()] * len(df)

    if "Author Keywords" not in df.columns:
        df["Author Keywords"] = ""

    # spójny Title
    if "Title" not in df.columns:
        if "Article Title" in df.columns:
            df["Title"] = df["Article Title"]
        else:
            df["Title"] = [f"Paper_{i}" for i in range(len(df))]
    return df


def _pipeline(df: pd.DataFrame, query: str, provider: str, model: str,
              out: Path, top_n: int | None) -> Path:
    """Original single-model pipeline."""
    from .io import autodetect_columns, combine_title_abstract
    from .embeddings import get_embeddings
    from .similarity import rank_by_cosine
    from .bibliometrics import full_report

    df = _ensure_aux_columns(df.copy())
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)

    vecs = get_embeddings(df["combined_text"].tolist(),
                          provider=provider, model=model)
    qvec = get_embeddings([query], provider=provider, model=model)[0]
    ranked = rank_by_cosine(qvec, vecs, df)

    p_all = out / "ranking.csv"
    ranked.to_csv(p_all, index=False)

    p_top = None
    if top_n:
        p_top = out / "topN.csv"
        ranked.head(top_n).to_csv(p_top, index=False)

    rep = out / "biblio_report.txt"
    full_report(ranked, path=rep, top_n=top_n)

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
    """New multi-model pipeline."""
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


# interactive Colab
def _colab_ui(out_dir: Path):
    from google.colab import files  # type: ignore

    display(HTML(
        "<h3>EmbedSLR – Interactive Upload</h3>"
        "<ol><li><b>Browse</b> → CSV</li><li>Wait for ✅</li>"
        "<li>Answer prompts in console</li></ol>"
    ))
    up = files.upload()
    if not up:
        display(HTML("<b style='color:red'>Abort – no file</b>"))
        return
    name, data = next(iter(up.items()))
    df = pd.read_csv(io.BytesIO(data))
    display(HTML(f"✅ Loaded <code>{name}</code> ({len(df)} rows)<br>"))

    # Ask for mode
    mode = input("🔧 Mode [single/multi]: ").strip().lower() or "single"
    
    q = input("❓ Research query: ").strip()
    
    if mode == "multi":
        # Multi-model mode
        print("\n📚 Multi-Model Mode")
        print("=" * 50)
        
        n_models_raw = input("How many models to use? [4]: ").strip()
        n_models = int(n_models_raw) if n_models_raw else 4
        
        models_config = []
        provs = list(_models())
        
        for i in range(n_models):
            print(f"\n--- Model {i+1}/{n_models} ---")
            print("Providers:", ", ".join(provs))
            prov = input(f"Provider [default={provs[0]}]: ").strip() or provs[0]
            
            print(f"Models for {prov} (showing first 10):")
            for m in _models()[prov][:10]:
                print("  •", m)
            mod = input("Model [ENTER=1st]: ").strip() or _models()[prov][0]
            
            models_config.append({"provider": prov, "model": mod})
            
            # API key if needed
            key = input(f"API key for {prov} (ENTER skip): ").strip()
            if key and (ev := _env_var(prov)):
                os.environ[ev] = key
        
        n_raw = input("\n🔢 Top‑N per model [17]: ").strip()
        top_n = int(n_raw) if n_raw else 17
        
        print("\n⏳ Running multi-model analysis...")
        zip_tmp = _multi_pipeline(df, q, models_config, out_dir, top_n)
        
    else:
        # Single-model mode (original)
        print("\n📖 Single-Model Mode")
        provs = list(_models())
        print("Providers:", provs)
        prov = input(f"Provider [default={provs[0]}]: ").strip() or provs[0]

        print("Models for", prov)
        for m in _models()[prov]:
            print("  •", m)
        mod = input("Model [ENTER=1st]: ").strip() or _models()[prov][0]

        n_raw = input("🔢 Top‑N for metrics [ENTER=all]: ").strip()
        top_n = int(n_raw) if n_raw else None

        key = input("API key (ENTER skip): ").strip()
        if key and (ev := _env_var(prov)):
            os.environ[ev] = key

        print("⏳ Computing…")
        zip_tmp = _pipeline(df, q, prov, mod, out_dir, top_n)

    dst = Path.cwd() / zip_tmp.name
    shutil.copy(zip_tmp, dst)
    print("✅ Finished – downloading ZIP")
    files.download(str(dst))


# CLI fallback
def _cli(out_dir: Path):
    print("== EmbedSLR CLI ==")
    csv_p = Path(input("CSV path: ").strip())
    df = pd.read_csv(csv_p)
    
    mode = input("Mode [single/multi]: ").strip().lower() or "single"
    q = input("Query: ").strip()
    
    if mode == "multi":
        n_models_raw = input("Number of models [4]: ").strip()
        n_models = int(n_models_raw) if n_models_raw else 4
        
        models_config = []
        for i in range(n_models):
            print(f"\n--- Model {i+1} ---")
            prov = input("Provider [sbert]: ").strip() or "sbert"
            mod = input(f"Model [default]: ").strip() or _models()[prov][0]
            models_config.append({"provider": prov, "model": mod})
            
            key = input(f"API key for {prov} [skip]: ").strip()
            if key and (ev := _env_var(prov)):
                os.environ[ev] = key
        
        n_raw = input("Top‑N per model [17]: ").strip()
        top_n = int(n_raw) if n_raw else 17
        
        z = _multi_pipeline(df, q, models_config, out_dir, top_n)
        
    else:
        prov = input("Provider [sbert]: ").strip() or "sbert"
        mod = input("Model [ENTER=default]: ").strip() or _models()[prov][0]
        n_raw = input("Top‑N [ENTER=all]: ").strip()
        top_n = int(n_raw) if n_raw else None
        key = input("API key [skip]: ").strip()
        if key and (ev := _env_var(prov)):
            os.environ[ev] = key
        z = _pipeline(df, q, prov, mod, out_dir, top_n)
    
    print("ZIP saved:", z)


# public
def run(save_dir: str | os.PathLike | None = None):
    save_dir = Path(save_dir or tempfile.mkdtemp(prefix="embedslr_"))
    clear_output()
    (_colab_ui if IN_COLAB else _cli)(save_dir)
