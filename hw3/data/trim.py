#########################################################
#   FileName:       [ trim.py ]                         #
#   PackageName:    [ Util ]                            #
#   Synopsis:       [ Trim prediction ]                 #
#   Author:         [ MedusaLafayetteDecorusSchiesse ]  #
#########################################################
import csv
import sys
import argparse

parser = argparse.ArgumentParser(prog='trim.py', description='Trim frame prediction file to phone prediction file.')
parser.add_argument('input_csv', type=str, metavar='<input-csv>',
        help='input frame prediction file')
parser.add_argument('output_csv', type=str, metavar='<output-csv>',
        help='output phone prediction file')
parser.add_argument('map_file', type=str, metavar='<map-file>',
        help='48_idx_chr.map_b')
args = parser.parse_args()
Readfile = args.input_csv      # input file name
Outputfile = args.output_csv    # after trimed data output file
Mapfile = args.map_file
idphone_dic = {}
id_list = []
phonestochr_dic = {}

ifile = open(Mapfile, "rb")
for line in ifile:
    l = line.split()
    phonestochr_dic[l[0]] = l[2]

ifile.close()

ifile = open(Readfile, "rb")
reader = csv.reader(ifile)

rownum = 0
current_id = ""
for row in reader:
    if rownum != 0:
        r = row[0].strip(' /n').split("_")
        now_id = r[0]+"_"+r[1]
        if now_id != current_id:
            current_id = now_id
            id_list.append(now_id)
            idphone_dic[now_id] = phonestochr_dic[row[1]]
        else:
            if phonestochr_dic[row[1]] != idphone_dic[now_id][-1]:
                idphone_dic[now_id] += phonestochr_dic[row[1]]

    rownum += 1


ifile.close()

for idkey in idphone_dic:
    if idphone_dic[idkey][0] == 'L':
        idphone_dic[idkey] = idphone_dic[idkey][1:]
    if idphone_dic[idkey][-1] == 'L':
        idphone_dic[idkey] = idphone_dic[idkey][:-1]


with open(Outputfile, 'w') as csvfile:
    fieldnames = ['id', 'phone_sequence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for idkey in id_list:
        writer.writerow({'id': idkey, 'phone_sequence': idphone_dic[idkey]})


