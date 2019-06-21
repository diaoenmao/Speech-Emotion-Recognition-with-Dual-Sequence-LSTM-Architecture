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

#with open(out_path, 'r') as f:
    #writer = csv.writer(f)

with open(out_path, 'r') as in_file:
    lines = [line.strip('\t') for line in in_file]
    lines = [line.split(",") for line in stripped if line]
    with open(out_path[0,-4] + '.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        #writer.writerow(('title', 'intro'))
        writer.writerows(lines)
