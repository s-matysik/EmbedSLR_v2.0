"""
Example: Multi-Model Systematic Literature Review
==================================================

This example demonstrates how to use EmbedSLR's multi-model functionality
to conduct a systematic literature review with multiple embedding models.
"""

import pandas as pd
from embedslr import multi_model_analysis
from pathlib import Path

def example_multi_model_slr():
    """
    Example of multi-model systematic literature review.
    
    This function demonstrates:
    1. Loading publication data
    2. Configuring multiple embedding models
    3. Running multi-model analysis
    4. Interpreting results
    """
    
    # 1. Load your data (example structure)
    # In practice, load from Scopus, Web of Science, or other database
    df = pd.DataFrame({
        "Title": [
            "Machine Learning in Healthcare",
            "Deep Learning Applications",
            "AI Ethics and Governance",
            # ... more publications
        ],
        "Abstract": [
            "This paper discusses machine learning applications in healthcare...",
            "We present deep learning methods for image recognition...",
            "This study examines ethical considerations in AI development...",
            # ... more abstracts
        ],
        "Author Keywords": [
            "machine learning; healthcare; AI",
            "deep learning; neural networks",
            "AI ethics; governance; policy",
            # ... more keywords
        ],
        "References": [
            "Author1 (2020); Author2 (2021)",
            "Author3 (2019); Author1 (2020)",
            "Author4 (2021); Author5 (2022)",
            # ... more references
        ]
    })
    
    # 2. Define research question
    research_query = """
    How does artificial intelligence impact healthcare delivery?
    What are the main applications of AI in medical diagnosis?
    What ethical considerations arise from AI use in healthcare?
    """
    
    # 3. Configure models (4 recommended based on research)
    models_config = [
        # Model 1: Fast and efficient
        {
            "provider": "sbert",
            "model": "all-MiniLM-L12-v2"
        },
        # Model 2: High quality semantic understanding
        {
            "provider": "sbert",
            "model": "all-mpnet-base-v2"
        },
        # Model 3: OpenAI (requires API key)
        {
            "provider": "openai",
            "model": "text-embedding-ada-002"
        },
        # Model 4: Good for sentence similarity
        {
            "provider": "sbert",
            "model": "all-distilroberta-v1"
        },
    ]
    
    print("=" * 80)
    print("MULTI-MODEL SYSTEMATIC LITERATURE REVIEW")
    print("=" * 80)
    print(f"\nDataset: {len(df)} publications")
    print(f"Models: {len(models_config)}")
    print(f"Query: {research_query[:100]}...")
    print("\nStarting analysis...\n")
    
    # 4. Run multi-model analysis
    results = multi_model_analysis(
        df=df,
        query=research_query,
        models_config=models_config,
        top_n=17,  # Select top 17 publications per model
        output_dir="./multi_model_results"
    )
    
    # 5. Examine results
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    
    # Model rankings
    print("\n📊 Model Performance Rankings:")
    print(results["model_rankings"][["Model", "A", "B", "Final_Rank"]])
    
    # Hierarchical groups
    print("\n📂 Publications by Consensus Level:")
    for n_models in sorted(results["hierarchical_groups"].keys(), reverse=True):
        count = len(results["hierarchical_groups"][n_models])
        print(f"  • {n_models} model(s): {count} publications")
    
    # Group statistics
    print("\n📈 Hierarchical Group Analysis:")
    print(results["hierarchical_analysis"])
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    four_model_pubs = results["hierarchical_groups"].get(4, pd.DataFrame())
    three_model_pubs = results["hierarchical_groups"].get(3, pd.DataFrame())
    
    print(f"\n✅ HIGH PRIORITY ({len(four_model_pubs)} publications):")
    print("   Publications selected by all 4 models")
    print("   → Highest reliability and thematic coherence")
    print("   → Core publications central to research question")
    
    print(f"\n⭐ IMPORTANT ({len(three_model_pubs)} publications):")
    print("   Publications selected by 3 models")
    print("   → High reliability with good coherence")
    print("   → Important contributions worth examining")
    
    # Visualization info
    print("\n📊 Visualizations created:")
    print(f"   • {len(results['radar_charts'])} radar charts (one per model)")
    print(f"   • {len(results['comparison_charts'])} comparison charts")
    
    # Output files
    print("\n💾 Output files saved to: ./multi_model_results/")
    print("   • model_rankings.csv")
    print("   • hierarchical_analysis.csv")
    print("   • consensus_groups/ (publications by consensus level)")
    print("   • model_selections/ (individual model results)")
    print("   • radar_charts/ (model performance visualizations)")
    print("   • comparisons/ (comparative analysis)")
    print("   • embedslr_multi_results.zip (complete package)")
    
    return results


def example_model_selection():
    """
    Example: How to pre-select best models for your domain.
    
    Before running full multi-model analysis, you can test different
    models and select the best ones based on bibliometric criteria.
    """
    
    print("\n" + "=" * 80)
    print("MODEL SELECTION PROCESS")
    print("=" * 80)
    
    # Test models to evaluate
    candidate_models = [
        {"provider": "sbert", "model": "all-MiniLM-L12-v2"},
        {"provider": "sbert", "model": "all-mpnet-base-v2"},
        {"provider": "sbert", "model": "all-distilroberta-v1"},
        {"provider": "sbert", "model": "multi-qa-MiniLM-L6-cos-v1"},
        # Add more models to test
    ]
    
    print(f"\n🧪 Testing {len(candidate_models)} models...")
    print("\nRecommended process:")
    print("1. Run each model on a sample of your data")
    print("2. Calculate bibliometric metrics for each model")
    print("3. Rank models using multi-criteria analysis")
    print("4. Select top 4 models for final analysis")
    print("\nMetrics to evaluate:")
    print("  • A: Average shared references")
    print("  • A': Jaccard index (references)")
    print("  • B: Average shared keywords")
    print("  • B': Jaccard index (keywords)")
    print("  • Shared: Publications overlapping with other models")
    print("  • Unique: Publications found only by this model")


def example_interpreting_results():
    """
    Example: How to interpret multi-model results.
    """
    
    print("\n" + "=" * 80)
    print("INTERPRETING MULTI-MODEL RESULTS")
    print("=" * 80)
    
    print("\n📋 Publication Consensus Levels:")
    
    print("\n🟢 4 MODELS (Highest Priority)")
    print("   Characteristics:")
    print("   • Selected by all models")
    print("   • Highest semantic relevance")
    print("   • Strong bibliographic coupling")
    print("   • Core publications for your review")
    print("   Action: Include in final review")
    
    print("\n🟡 3 MODELS (High Priority)")
    print("   Characteristics:")
    print("   • Selected by most models")
    print("   • High relevance and coherence")
    print("   • Important contributions")
    print("   Action: Include in final review")
    
    print("\n🟠 2 MODELS (Moderate Priority)")
    print("   Characteristics:")
    print("   • Selected by half the models")
    print("   • May represent specialized perspectives")
    print("   • Mixed signals on relevance")
    print("   Action: Manual review recommended")
    
    print("\n🔴 1 MODEL (Low Priority)")
    print("   Characteristics:")
    print("   • Selected by only one model")
    print("   • May be edge cases or false positives")
    print("   • Could represent unique insights")
    print("   Action: Quick scan or exclude")
    
    print("\n📊 Using Bibliometric Metrics:")
    print("   • High A and A' → Strong reference overlap")
    print("   • High B and B' → Strong thematic coherence")
    print("   • High Unique → Model finds distinct perspectives")
    print("   • High Shared → Model aligns with others")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "EmbedSLR EXAMPLES")
    print("=" * 80)
    
    # Note: In practice, you would run these examples separately
    # This is just a demonstration structure
    
    print("\n1. Multi-Model Analysis Example")
    # example_multi_model_slr()
    
    print("\n2. Model Selection Example")
    example_model_selection()
    
    print("\n3. Results Interpretation Guide")
    example_interpreting_results()
    
    print("\n" + "=" * 80)
    print("For full examples, uncomment the function calls above")
    print("and ensure you have your data prepared.")
    print("=" * 80 + "\n")
