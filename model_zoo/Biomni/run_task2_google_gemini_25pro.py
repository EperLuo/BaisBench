from biomni.agent import A1
import pandas as pd
import re
import os
import time

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='gemini-2.5-pro-preview-05-06', base_url="xxx/v1/",
           api_key="sk-xxx", 
           source="Gemini",
           expected_data_lake_files = [])

h5ad_list = [f for f in os.listdir('/data/lep/BaisBench/Task2_data/h5ad_file') if f.endswith('.h5ad')]
df = pd.read_excel('../../Task2_data/BAISBench_task2.xlsx', sheet_name='Sheet1')

start_time = time.time()
# Read question and background information
for question_id in range(41):
    question_name = df['name'][question_id]
    background_info = df[df['name']==question_name]['background'].values.item()

    question_list = []
    question_answer = []
    for i in range(1,6):
        if pd.isna(df[df['name']==question_name][f'Questions{i}'].values.item()):
            continue
        else:
            question_list.append(df[df['name']==question_name][f'Questions{i}'].values.item())
            answers = re.findall(r'\b([A-Z])\)', df[df['name']==question_name][f'Answer{i}'].values.item())
            question_answer.append(answers)

    questions = ''
    for i in range(len(question_list)):
        questions += f"Q{i+1}: {question_list[i]}\n"

    h5ad_path = ''
    for h5ad in h5ad_list:
        if question_name in h5ad:
            h5ad_path += f"/data/lep/BaisBench/Task2_data/h5ad_file/{h5ad}, "

    # Execute biomedical tasks using natural language
    agent.go("Given these single cell RNA-seq data {} and the background information: {} , analysis the data to answer the following questions: {}. Provide the letter options as answers.".format(h5ad_path, background_info, questions))

    # Save conversation history as PDF
    name = str(question_id+1)
    agent.save_conversation_history(f"output_google_gemini25pro/analysis_results_{name}.pdf")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time} seconds")