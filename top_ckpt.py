import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_path",
    type=str,
)

args = parser.parse_args()
data = open(args.file_path, "r").readlines()
data = [line[:-1] for line in data]

def get_tuple(ckpt):
    return tuple(float(x) for x in ckpt.strip('()').split(','))

best_checkpoints = [get_tuple(ckpt) for ckpt in data]
top_vqa_ckpts = [c for c in sorted(best_checkpoints, key=lambda x: x[1], reverse=True)[:3]]
top_cs_ckpts = [c for c in sorted(best_checkpoints, key=lambda x: x[2], reverse=True)[:3]]
print(f"Top CS checkpoints: {top_cs_ckpts}")
print(f"Top VQA checkpoints: {top_vqa_ckpts}")
