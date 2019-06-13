import csv
import os

def extract_trans(list_input_files, output_file):
    lines = []
    for input_file in list_input_files:
        count = 0
        with open(input_file, 'r') as f:
            lines = f.readlines()
        with open(output_file, 'a') as f:
            csv_writer = csv.writer(f)
            lines = sorted(lines)

            for line in lines:
                name = line.split(':')[0].split(' ')[0].strip()

                if name[:3] != 'Ses':
                    continue
                elif name[-3:-1] == 'XX':
                    continue
                trans = line.split(':')[1].strip()

                count += 1
                csv_writer.writerow([name, trans])


if __name__ == '__main__':
    list_files = []
    for x in range(5):
        session_name = 'Session' + str(x+1)
        path = '/scratch/speech/IEMOCAP_sample/' + session_name + '/dialog/transcriptions/'
