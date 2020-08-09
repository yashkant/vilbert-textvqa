import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm

sim_holder = "../../datasets/VQA/back-translate/sim-result2/rep_{}_{}.npy"
thresh_values = [0.7, 0.8, 0.9]
decontract = [True, False]
df = pd.DataFrame(columns=["question_id"])

for th_value in thresh_values:
    for decon in decontract:
        key = f"unique-thresh-{th_value}-decontract-{decon}"

        # add a new column
        df[key] = None

        for split in ["val"]:
            for seq_id in range(2):
                file_path = sim_holder.format(split, seq_id)
                file_data = np.load(file_path, allow_pickle=True).item()

                for qid, value in tqdm(file_data.items()):
                    filtered_questions = []
                    for rep in value["rephrasings_list"]:
                        if rep["sim_score"] > th_value:
                            question = rep["rephrasing"]
                            if decontract:
                                question = contractions.fix(question)
                            filtered_questions.append(question)

                    if len(df[df['question_id'] == qid]) > 0:
                        df.loc[df["question_id"] == qid, key] = len(set(filtered_questions))
                    else:
                        df = df.append({"question_id": qid, key: len(set(filtered_questions))}, ignore_index=True)

        import pdb
        pdb.set_trace()