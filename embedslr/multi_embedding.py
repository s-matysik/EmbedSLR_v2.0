"""
embedslr.multi_embedding
========================

Multi-model embedding analysis for systematic literature reviews.
Implements the methodology from "Efficient AI-powered Decision-Making in 
Systematic Literature Reviews Using Multi-Embedding Models".

Key features:
- Parallel execution of multiple embedding models
- Hierarchical ranking based on model consensus
- Multi-criteria analysis with radar charts
- Bibliometric validation across model selections
"""

from __future__ import annotations

import itertools as it
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ────────────────────────────────────────────────────────────────────────────
# Model Selection and Ranking
# ────────────────────────────────────────────────────────────────────────────

def select_top_publications(
    df: pd.DataFrame,
    distance_col: str = "distance_cosine",
    top_n: int = 17
) -> pd.DataFrame:
    """
    Selects top N publications with lowest cosine distance.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with publications and distance column
    distance_col : str
        Name of the column containing cosine distances
    top_n : int
        Number of top publications to select
        
    Returns
    -------
    pd.DataFrame
        Top N publications sorted by distance
    """
    return df.nsmallest(top_n, distance_col).copy()


def calculate_model_metrics(
    df_top: pd.DataFrame,
    refs_col: str = "Parsed_References",
    kw_col: str = "Author Keywords"
) -> Dict[str, float]:
    """
    Calculates bibliometric metrics for a set of publications selected by one model.
    
    Metrics:
    - A: Average shared references per pair
    - A': Jaccard index for references
    - B: Average shared keywords per pair
    - B': Jaccard index for keywords
    
    Parameters
    ----------
    df_top : pd.DataFrame
        DataFrame with top publications from one model
    refs_col : str
        Column name for parsed references (sets)
    kw_col : str
        Column name for author keywords
        
    Returns
    -------
    Dict[str, float]
        Dictionary with metrics A, A', B, B'
    """
    n = len(df_top)
    if n < 2:
        return {"A": 0.0, "A'": 0.0, "B": 0.0, "B'": 0.0}
    
    # Prepare references
    refs = []
    for val in df_top[refs_col]:
        if isinstance(val, set):
            refs.append(val)
        elif isinstance(val, str):
            refs.append({r.strip() for r in val.split(");") if r.strip()})
        else:
            refs.append(set())
    
    # Prepare keywords
    kws = []
    for val in df_top[kw_col].fillna(""):
        kws.append({w.strip().lower() for w in str(val).split(";") if w.strip()})
    
    pairs_count = n * (n - 1) / 2
    
    # Calculate metrics
    total_ref_int = 0
    total_ref_jac = 0
    total_kw_int = 0
    total_kw_jac = 0
    
    for i, j in it.combinations(range(n), 2):
        # References
        inter_r = refs[i] & refs[j]
        union_r = refs[i] | refs[j]
        total_ref_int += len(inter_r)
        total_ref_jac += len(inter_r) / len(union_r) if union_r else 0.0
        
        # Keywords
        inter_k = kws[i] & kws[j]
        union_k = kws[i] | kws[j]
        total_kw_int += len(inter_k)
        total_kw_jac += len(inter_k) / len(union_k) if union_k else 0.0
    
    return {
        "A": total_ref_int / pairs_count if pairs_count > 0 else 0.0,
        "A'": total_ref_jac / pairs_count if pairs_count > 0 else 0.0,
        "B": total_kw_int / pairs_count if pairs_count > 0 else 0.0,
        "B'": total_kw_jac / pairs_count if pairs_count > 0 else 0.0,
    }


def count_shared_and_unique(
    model_selections: Dict[str, Set[Any]],
    model_name: str
) -> Tuple[int, int]:
    """
    Counts shared and unique publications for a given model.
    
    Parameters
    ----------
    model_selections : Dict[str, Set[Any]]
        Dictionary mapping model names to sets of selected publication IDs
    model_name : str
        Name of the model to analyze
        
    Returns
    -------
    Tuple[int, int]
        (number of shared publications, number of unique publications)
    """
    current = model_selections[model_name]
    other_models = [s for name, s in model_selections.items() if name != model_name]
    
    if not other_models:
        return 0, len(current)
    
    all_others = set().union(*other_models)
    shared = len(current & all_others)
    unique = len(current - all_others)
    
    return shared, unique


def rank_models_multi_criteria(
    model_metrics: Dict[str, Dict[str, float]],
    model_selections: Dict[str, Set[Any]]
) -> pd.DataFrame:
    """
    Creates a multi-criteria ranking of models based on all metrics.
    
    Parameters
    ----------
    model_metrics : Dict[str, Dict[str, float]]
        Metrics for each model (A, A', B, B')
    model_selections : Dict[str, Set[Any]]
        Sets of selected publications for each model
        
    Returns
    -------
    pd.DataFrame
        Ranking table with positions in each criterion and final average rank
    """
    results = []
    
    for model_name in model_metrics:
        metrics = model_metrics[model_name]
        shared, unique = count_shared_and_unique(model_selections, model_name)
        
        results.append({
            "Model": model_name,
            "A": metrics["A"],
            "A'": metrics["A'"],
            "B": metrics["B"],
            "B'": metrics["B'"],
            "Shared": shared,
            "Unique": unique,
        })
    
    df = pd.DataFrame(results)
    
    # Create rankings for each metric (higher is better for all)
    rank_cols = []
    for col in ["A", "A'", "B", "B'", "Shared", "Unique"]:
        rank_col = f"{col}_Rank"
        df[rank_col] = df[col].rank(ascending=False, method='average')
        rank_cols.append(rank_col)
    
    # Calculate average rank
    df["Avg_Rank"] = df[rank_cols].mean(axis=1)
    df["Final_Rank"] = df["Avg_Rank"].rank(method='min').astype(int)
    
    return df.sort_values("Final_Rank")


# ────────────────────────────────────────────────────────────────────────────
# Hierarchical Publication Grouping
# ────────────────────────────────────────────────────────────────────────────

def group_by_model_consensus(
    model_selections: Dict[str, pd.DataFrame],
    id_col: str = "Title"
) -> Dict[int, pd.DataFrame]:
    """
    Groups publications by number of models that selected them.
    
    Parameters
    ----------
    model_selections : Dict[str, pd.DataFrame]
        Dictionary mapping model names to DataFrames of selected publications
    id_col : str
        Column name to use as publication identifier
        
    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping number of models (1-4) to DataFrames of publications
    """
    # Count how many models selected each publication
    pub_counts: Dict[str, List[str]] = defaultdict(list)
    
    for model_name, df in model_selections.items():
        for pub_id in df[id_col]:
            pub_counts[pub_id].append(model_name)
    
    # Group by consensus level
    groups: Dict[int, List[Tuple[str, List[str]]]] = defaultdict(list)
    for pub_id, model_list in pub_counts.items():
        n_models = len(model_list)
        groups[n_models].append((pub_id, model_list))
    
    # Create DataFrames for each group
    result = {}
    all_dfs = pd.concat(model_selections.values(), ignore_index=True)
    all_dfs = all_dfs.drop_duplicates(subset=[id_col])
    
    for n_models in sorted(groups.keys()):
        pub_ids = [item[0] for item in groups[n_models]]
        group_df = all_dfs[all_dfs[id_col].isin(pub_ids)].copy()
        group_df["N_Models"] = n_models
        group_df["Selected_By"] = [
            ", ".join(item[1]) 
            for item in groups[n_models]
        ]
        result[n_models] = group_df
    
    return result


def analyze_hierarchical_groups(
    groups: Dict[int, pd.DataFrame],
    refs_col: str = "Parsed_References",
    kw_col: str = "Author Keywords"
) -> pd.DataFrame:
    """
    Analyzes bibliometric coherence of each hierarchical group.
    
    Parameters
    ----------
    groups : Dict[int, pd.DataFrame]
        Groups from group_by_model_consensus()
    refs_col : str
        Column name for references
    kw_col : str
        Column name for keywords
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for each group
    """
    results = []
    
    for n_models, df in sorted(groups.items()):
        metrics = calculate_model_metrics(df, refs_col, kw_col)
        results.append({
            "N_Models": n_models,
            "Count": len(df),
            "Avg_Shared_Refs": metrics["A"],
            "Jaccard_Refs": metrics["A'"],
            "Avg_Shared_KWs": metrics["B"],
            "Jaccard_KWs": metrics["B'"],
        })
    
    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────

def create_radar_chart(
    model_name: str,
    metrics: Dict[str, float],
    shared: int,
    unique: int,
    save_path: Path | str | None = None
) -> Path | None:
    """
    Creates a radar chart for a single model showing its performance across metrics.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    metrics : Dict[str, float]
        Dictionary with A, A', B, B' values
    shared : int
        Number of shared publications
    unique : int
        Number of unique publications
    save_path : Path | str | None
        Where to save the chart (if None, not saved)
        
    Returns
    -------
    Path | None
        Path to saved chart or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping radar chart")
        return None
    
    # Prepare data
    categories = ['A (Refs)', "A' (Refs)", 'B (KWs)', "B' (KWs)", 'Shared', 'Unique']
    
    # Normalize values to 0-1 scale for visualization
    values = [
        metrics["A"],
        metrics["A'"],
        metrics["B"],
        metrics["B'"],
        shared / 50,  # assume max 50
        unique / 20,  # assume max 20
    ]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # complete the circle
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Draw one line and fill area
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color='#2E8B57')
    ax.fill(angles, values, alpha=0.25, color='#2E8B57')
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    
    # Add title
    plt.title(f'Model: {model_name}', size=14, y=1.08)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    plt.close()
    return None


def create_comparison_charts(
    hierarchical_analysis: pd.DataFrame,
    save_dir: Path | str
) -> List[Path]:
    """
    Creates bar charts comparing groups selected by different numbers of models.
    
    Parameters
    ----------
    hierarchical_analysis : pd.DataFrame
        Output from analyze_hierarchical_groups()
    save_dir : Path | str
        Directory to save charts
        
    Returns
    -------
    List[Path]
        Paths to saved charts
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping comparison charts")
        return []
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Chart 1: Publication counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(hierarchical_analysis["N_Models"], hierarchical_analysis["Count"],
           color='#4169E1', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Models', fontsize=12)
    ax.set_ylabel('Number of Publications', fontsize=12)
    ax.set_title('Publications Selected by N Models', fontsize=14)
    ax.set_xticks(hierarchical_analysis["N_Models"])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    path1 = save_dir / "publication_counts.png"
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path1)
    
    # Chart 2: References comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(hierarchical_analysis))
    width = 0.35
    
    ax.bar(x - width/2, hierarchical_analysis["Avg_Shared_Refs"],
           width, label='Avg Shared Refs', color='#FF6347', alpha=0.8)
    ax.bar(x + width/2, hierarchical_analysis["Jaccard_Refs"],
           width, label='Jaccard Refs', color='#4682B4', alpha=0.8)
    
    ax.set_xlabel('Number of Models', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Reference-Based Metrics by Model Consensus', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(hierarchical_analysis["N_Models"])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    path2 = save_dir / "references_comparison.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path2)
    
    # Chart 3: Keywords comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width/2, hierarchical_analysis["Avg_Shared_KWs"],
           width, label='Avg Shared KWs', color='#32CD32', alpha=0.8)
    ax.bar(x + width/2, hierarchical_analysis["Jaccard_KWs"],
           width, label='Jaccard KWs', color='#9370DB', alpha=0.8)
    
    ax.set_xlabel('Number of Models', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Keyword-Based Metrics by Model Consensus', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(hierarchical_analysis["N_Models"])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    path3 = save_dir / "keywords_comparison.png"
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(path3)
    
    return saved_paths


# ────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ────────────────────────────────────────────────────────────────────────────

def multi_model_analysis(
    df: pd.DataFrame,
    query: str,
    models_config: List[Dict[str, str]],
    top_n: int = 17,
    output_dir: Path | str | None = None,
    id_col: str = "Title",
    refs_col: str = "Parsed_References",
    kw_col: str = "Author Keywords"
) -> Dict[str, Any]:
    """
    Executes complete multi-model analysis pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset of publications
    query : str
        Research query/problem description
    models_config : List[Dict[str, str]]
        List of model configurations, each dict should have 'provider' and 'model' keys
        Example: [{'provider': 'sbert', 'model': 'all-MiniLM-L12-v2'}, ...]
    top_n : int
        Number of top publications to select per model
    output_dir : Path | str | None
        Directory for saving outputs
    id_col : str
        Column to use as publication identifier
    refs_col : str
        Column with parsed references
    kw_col : str
        Column with author keywords
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary containing:
        - model_rankings: DataFrame with ranked models
        - hierarchical_groups: Dict with publication groups
        - hierarchical_analysis: DataFrame with group statistics
        - model_selections: Dict with top publications per model
        - radar_charts: List of paths to radar charts
        - comparison_charts: List of paths to comparison charts
    """
    from .embeddings import get_embeddings
    from .io import autodetect_columns, combine_title_abstract
    
    # Prepare text
    tcol, acol = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, tcol, acol)
    
    model_selections: Dict[str, pd.DataFrame] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}
    model_selection_sets: Dict[str, Set[Any]] = {}
    
    print(f"\n🔄 Running multi-model analysis with {len(models_config)} models...")
    
    # Process each model
    for i, config in enumerate(models_config, 1):
        provider = config['provider']
        model = config['model']
        model_name = f"{provider}::{model}"
        
        print(f"\n[{i}/{len(models_config)}] Processing {model_name}...")
        
        # Get embeddings
        doc_vecs = get_embeddings(
            df["combined_text"].tolist(),
            provider=provider,
            model=model
        )
        query_vec = get_embeddings([query], provider=provider, model=model)[0]
        
        # Calculate similarities
        q = np.asarray(query_vec).reshape(1, -1)
        d = np.asarray(doc_vecs)
        sim = cosine_similarity(q, d)[0]
        
        df_model = df.copy()
        df_model["distance_cosine"] = 1 - sim
        
        # Select top N
        top_pubs = select_top_publications(df_model, top_n=top_n)
        model_selections[model_name] = top_pubs
        model_selection_sets[model_name] = set(top_pubs[id_col])
        
        # Calculate metrics
        metrics = calculate_model_metrics(top_pubs, refs_col, kw_col)
        model_metrics[model_name] = metrics
        
        print(f"   ✓ Selected {len(top_pubs)} publications")
        print(f"   ✓ A={metrics['A']:.4f}, A'={metrics['A']:.4f}")
        print(f"   ✓ B={metrics['B']:.4f}, B'={metrics['B']:.4f}")
    
    # Rank models
    print("\n📊 Ranking models...")
    model_rankings = rank_models_multi_criteria(model_metrics, model_selection_sets)
    
    # Group publications hierarchically
    print("\n📂 Creating hierarchical groups...")
    hierarchical_groups = group_by_model_consensus(model_selections, id_col)
    hierarchical_analysis = analyze_hierarchical_groups(
        hierarchical_groups, refs_col, kw_col
    )
    
    print(f"   ✓ Found {len(hierarchical_groups)} groups")
    for n_models, group_df in sorted(hierarchical_groups.items()):
        print(f"   ✓ {n_models} model(s): {len(group_df)} publications")
    
    # Create visualizations if output directory specified
    radar_charts = []
    comparison_charts = []
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📈 Creating visualizations...")
        
        # Radar charts for each model
        radar_dir = output_dir / "radar_charts"
        radar_dir.mkdir(exist_ok=True)
        
        for model_name in model_metrics:
            shared, unique = count_shared_and_unique(model_selection_sets, model_name)
            safe_name = model_name.replace("::", "_").replace("/", "_")
            chart_path = create_radar_chart(
                model_name,
                model_metrics[model_name],
                shared,
                unique,
                radar_dir / f"{safe_name}.png"
            )
            if chart_path:
                radar_charts.append(chart_path)
        
        print(f"   ✓ Created {len(radar_charts)} radar charts")
        
        # Comparison charts
        comp_dir = output_dir / "comparisons"
        comparison_charts = create_comparison_charts(hierarchical_analysis, comp_dir)
        print(f"   ✓ Created {len(comparison_charts)} comparison charts")
    
    print("\n✅ Multi-model analysis complete!")
    
    return {
        "model_rankings": model_rankings,
        "hierarchical_groups": hierarchical_groups,
        "hierarchical_analysis": hierarchical_analysis,
        "model_selections": model_selections,
        "model_metrics": model_metrics,
        "radar_charts": radar_charts,
        "comparison_charts": comparison_charts,
    }
