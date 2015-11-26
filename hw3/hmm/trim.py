import csv
import sys
Readfile = sys.argv[1]      # input file name
Outputfile = sys.argv[2]    # after trimed data output file
Mapfile = "48_idx_chr.map_b"
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


