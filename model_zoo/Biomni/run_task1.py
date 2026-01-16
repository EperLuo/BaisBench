from biomni.agent import A1
import pandas as pd
import re
import os
import time

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514', base_url="https://api.zyai.online",
           api_key="sk-PIgxzVnY954VFhDo8153Fd08F1Ed41Be9d2a29Ee85A7EeB4s", 
           source="Anthropic",
           expected_data_lake_files = [])

h5ad_list = [f for f in os.listdir('/data/lep/BaisBench/Task1_data') if f.endswith('raw.h5ad')]

start_time = time.time()
# Read question and background information
for h5ad in h5ad_list:

    h5ad_path = f"/data/lep/BaisBench/Task1_data/{h5ad}"
    file_name = h5ad.replace('_raw.h5ad','')
    # Execute biomedical tasks using natural language
    agent.go(f"Given this scRNA-seq dataset from a human {file_name} sample: {h5ad_path}, Perform basic analysis on cell data and annotate cell types. Annotations must be based on knowledge and cannot utilize external tools such as CellTypist. Return the annotated cell data in h5ad format. Save the annotated file in /data/lep/BaisBench/model_zoo/Biomni/output_task1/{file_name}.h5ad")

    # Save conversation history as PDF
    name = h5ad.replace('.h5ad','')
    agent.save_conversation_history(f"output_task1/analysis_results_{name}.pdf")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time} seconds")