import csv

path = '/scratch/speech/models/classification/andre_checkpoint_stats.txt'
out_path = '/scratch/speech/models/classification/andre_checkpoint_stats_updated.txt'
out = open(out_path, 'w')
with open(path, 'r') as f:
    lines = [line.rstrip('\t') for line in f]
f.close()

#parsed_data = csv.reader(open(path, 'rb'), delimiter='\t')
for line in lines:
    if '=================================Best Test Accuracy' in line:
        print(line)
    elif '========================== Batch Normalization' in line:
        continue
    else:
        out.write(line)

out.close()
