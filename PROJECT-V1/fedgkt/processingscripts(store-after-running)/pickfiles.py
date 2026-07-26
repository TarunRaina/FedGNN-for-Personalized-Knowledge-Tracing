import pandas as pd
import json

stats = pd.read_csv('data/processed/student_stats.csv')
with open('data/processed/student_splits.json') as f:
    splits = json.load(f)

working_ids = set(splits['train']) | set(splits['val']) | set(splits['test'])
subset = stats[stats['user_id'].isin(working_ids)].sort_values('total_interactions')

print("Smallest:")
print(subset.head(3)[['user_id', 'total_interactions']])
print("\nMiddle:")
mid = len(subset) // 2
print(subset.iloc[mid-1:mid+2][['user_id', 'total_interactions']])
print("\nLargest:")
print(subset.tail(3)[['user_id', 'total_interactions']])