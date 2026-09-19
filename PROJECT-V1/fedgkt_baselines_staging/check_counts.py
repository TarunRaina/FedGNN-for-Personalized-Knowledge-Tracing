import json, torch

sp = json.load(open('staged_input/student_splits.json'))
for name in ['train', 'val', 'test']:
    lens = sorted(
        len(torch.load(f'staged_input/pkgs/pkg_{u}.pt',
                       weights_only=False)['exercise_idx'])
        for u in sp[name]
    )
    print(f"{name:6s} students={len(lens):4d}  total={sum(lens):7d}  "
          f"min={lens[0]:5d}  median={lens[len(lens)//2]:5d}  max={lens[-1]:5d}")