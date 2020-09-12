import os
import json
os.chdir("/nethome/ykant3/vilbert-multi-task/save/")
dirs = [
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-alt/",
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-alt-hard/",
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-alt-sc20/",
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-ce-batches/",
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-pretrain/",
    "VQA_spatial_m4c_mmt_vqa-final-ablate-scl-scl-batches/",
    "VQA_spatial_m4c_mmt_vqa-final-baseline/",
    "VQA_spatial_m4c_mmt_vqa-final-baseline-train-minval-old/",
    "VQA_spatial_m4c_mmt_vqa-final-baseline-trainval/",
    "VQA_spatial_m4c_mmt_vqa-final-best/",
    "VQA_spatial_m4c_mmt_vqa-final-best-0.7/",
    "VQA_spatial_m4c_mmt_vqa-final-best-0.8/",
    "VQA_spatial_m4c_mmt_vqa-final-best-0.9/",
    "VQA_spatial_m4c_mmt_vqa-final-best-fourth/",
    "VQA_spatial_m4c_mmt_vqa-final-best-img/",
    "VQA_spatial_m4c_mmt_vqa-final-best-que/",
    "VQA_spatial_m4c_mmt_vqa-final-best-second/",
    "VQA_spatial_m4c_mmt_vqa-final-best-third/",
    "VQA_spatial_m4c_mmt_vqa-final-best-trainval/",
    "VQA_spatial_m4c_mmt_vqa-final-best-trainval-large/",
    "VQA_spatial_m4c_mmt_vqa-final-best-trainval-long/",
    "VQA_spatial_m4c_mmt_vqa-final-best-two-norm/",
    "VQA_spatial_m4c_mmt_vqa-final-best-two-norm-second/",
    "VQA_spatial_m4c_mmt_vqa-final-cc-baseline-allowed-18k/",
    "VQA_spatial_m4c_mmt_vqa-final-cc-best-allowed-18k/",
    "VQA_spatial_m4c_mmt_vqa-final-ce-scl-batches/",
    "VQA_spatial_m4c_mmt_vqa-final-no-aug-baseline-train/",
    "VQA_spatial_m4c_mmt_vqa-final-no-aug-baseline-trainval/",
    "VQA_spatial_m4c_mmt_vqa-final-scl-fintune/",
    "VQA_spatial_m4c_mmt_vqa-final-scl-fintune-10x/",
    "VQA_spatial_m4c_mmt_vqa-final-scl-fintune-full/",
    "VQA_spatial_m4c_mmt_vqa-final-scl-pretrain-fixed/",
]

for _dir in dirs:
    preds_file = f"{_dir}preds_revqa_val.json"
    import pdb
    pdb.set_trace()
    if os.path.exists(preds_file):
        preds = json.load(open(preds_file, "r"))
        val_scores = [x['vqa_score'] for x in list(preds.values())]
        print(f"Exp: {_dir} \n Val Score: {round(sum(val_scores)/len(val_scores), 4)}")
        import pdb
        pdb.set_trace()
