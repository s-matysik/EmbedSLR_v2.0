# Example Dataset and Results
This folder contains an export from Scopus
data_Scopus_CSR_influence_consumer_behavior.csv
generated on 09.10.2025, containing 271 records matching the query:

TITLE-ABS-KEY("impact" AND "AI" AND "young" AND "health")

The dataset was processed using EmbedSLR 2.0 in Google Colab.
Four embedding models were selected to represent different architectures:
•	sentence-transformers/all-mpnet-base-v2: MPNet architecture, 109M parameters
•	sentence-transformers/all-distilroberta-v1: Distilled RoBERTa, 82M parameters
•	sentence-transformers/all-MiniLM-L6-v2: Distilled BERT, 22M parameters
•	openai/text-embedding-ada-002: Proprietary transformer, 1536-dim embeddings

The bibliometric audit was limited to the Top 40 resultsfor the research problem at hand: 

"The impact of AI on young peoples health"

The computation took approximately 4 minutes to complete.
Output files are available in the examples/AI_youth_health/embedslr_multi_results.zip

This process can be replicated, and the results independently verified, when testing the software.

## Installation Google Colab 

```bash
!pip install git+https://github.com/s-matysik/EmbedSLR_v2.0.git
from embedslr.colab_app import run
run()
