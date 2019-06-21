import csv

path = '/scratch/speech/models/classification/andre_checkpoint_stats.txt'
parsed_data = csv.reader(open(path, 'rb'), delimiter='\t')
for row in parsed_data:
    if '=================================Best Test Accuracy' in row:
        print(row)
