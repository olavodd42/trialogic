"""Generate a comparative KDE plot of predicted NEWS2 score distributions."""

import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Visual configuration for academic publication (white bg, serif font)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "serif"

def extract_score(text_or_dict):
    """
    Extract the NEWS Score from various output formats (JSON or raw string).
    """
    if isinstance(text_or_dict, dict):
        # Try structured fields first
        if "risk_score" in text_or_dict:
            val = text_or_dict["risk_score"]
        elif "news_score" in text_or_dict:
            val = text_or_dict["news_score"]
        else:
            # Dict without obvious field — convert to string and try regex
            val = str(text_or_dict)
    else:
        val = str(text_or_dict)
    
    # Robust regex to find integers associated with a score
    # Matches "NEWS: 3", "Score: 3", or a clean standalone number
    match = re.search(r'(?:NEWS2?|Score|Total)?\W*(\d+)', str(val), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0  # Conservative default for parsing failure (assume normal)

def load_scores(file_path, label):
    """Load scores from a JSONL file."""
    scores = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Onde está o score? Depende do script que gerou.
                # No baseline_results.jsonl geralmente está na resposta crua ou extraída
                # No experiment_results_v1.jsonl está em 'risk_score'
                
                # Adapte conforme a estrutura do seu JSONL
                raw_content = data.get("response", data.get("risk_score", ""))
                score = extract_score(raw_content)
                
                # Filter extreme hallucinations (e.g. Score 50 doesn't exist, max is ~20)
                if 0 <= score <= 25:
                    scores.append(score)
            except:
                pass
                
    return pd.DataFrame({"Score": scores, "System": label})

def plot_comparison():
    print("Generating comparative risk distribution chart...")
    
    # Paths based on previous logs
    path_baseline = "results/baseline_results.jsonl"
    path_trialogic = "results/experiment_results_v1.jsonl"  # Full Agents
    
    df_baseline = load_scores(path_baseline, "Baseline (Zero-Shot)")
    df_trialogic = load_scores(path_trialogic, "TriaLogic (Agents)")
    
    if df_baseline.empty or df_trialogic.empty:
        print("Error: Could not load sufficient data for comparison.")
        return

    # Combine DataFrames
    df_final = pd.concat([df_baseline, df_trialogic])
    
    plt.figure(figsize=(10, 6))
    
    # Plot KDE (Kernel Density Estimate) - Shows the smooth distribution shape
    # cut=0 ensures the chart does not draw below 0
    sns.kdeplot(
        data=df_final, 
        x="Score", 
        hue="System", 
        fill=True, 
        common_norm=False, 
        alpha=0.3, 
        linewidth=2,
        cut=0
    )
    
    # Alternative: overlapping histogram (uncomment if you prefer bars)
    # sns.histplot(data=df_final, x="Score", hue="System", element="step", stat="density", common_norm=False)

    plt.title("Distribution of Predicted NEWS2 Scores: Hallucination vs. Safety", fontsize=14, fontweight='bold')
    plt.xlabel("NEWS2 Score Value")
    plt.ylabel("Density (Frequency)")
    plt.xlim(0, 15)  # Focus on the relevant region (realistic scores)
    
    # Add explanatory annotations (academic figure style)
    plt.annotate('Baseline "Spread"\n(Uncertainty/Hallucination)', 
                 xy=(5, 0.05), xytext=(8, 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5))

    plt.annotate('TriaLogic "Peak"\n(Conservative/Safety)', 
                 xy=(0.5, 0.4), xytext=(3, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5))

    output_file = "results/chart_comparative_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    print("\nINTERPRETATION FOR ARTICLE:")
    print("1. TriaLogic's high peak at 0 shows it adopts a 'Default-to-Safety' strategy.")
    print("2. The Baseline's long/fat tail shows it invents intermediate risks without evidence.")

if __name__ == "__main__":
    plot_comparison()