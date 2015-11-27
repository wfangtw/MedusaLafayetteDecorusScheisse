import sys

with open(sys.argv[1], "r") as fr:
    with open(sys.argv[2], "w") as fw:

        pline = ""
        fw.write(fr.readline())        # pop out first line (Id,Prediction)
        for line in fr:
            data = line.strip(" \n").split(",")
            instance = data[0].rsplit("_", 1)
            s_id = instance[0]
            index = int(instance[1])
            phone = data[1]
            if index == 1:
                ppphone = phone
                fw.write(pline)     # Before new sentence starts, write previous sentence's last line
                fw.write(line)      # Write new sentence's first line
            elif index == 2:
                pphone = phone
            else:
                if pphone != ppphone and phone != pphone:
                    pphone = ppphone
                fw.write(s_id + "_" + str(index - 1) + "," + pphone + "\n")
                ppphone = pphone
                pphone = phone
            pline = line
        fw.write(pline)         # Write the last sentence's last line

min_count = 2

with open(sys.argv[1], "r") as fr:
    with open(sys.argv[2], "w") as fw:
        prev_phone = ""
        current_phone = ""
        current_sentence = ""
        current_id = 0
        count = 0
        fw.write(fr.readline())        # pop out first line (Id,Prediction)
        for line in fr:
            #lalala = raw_input("blahblah: ")
            #print line
            data = line.strip(" \n").split(",")
            instance = data[0].rsplit("_", 1)
            s_id = instance[0]
            index = int(instance[1])
            phone = data[1]
            if index == 1:
                if count <= min_count:
                    #print("writing: " + prev_phone + " x" + str(count))
                    for i in range(count):
                        fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + prev_phone + '\n')
                else:
                    #print("writing: " + current_phone + " x" + str(count))
                    for i in range(count):
                        fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + current_phone + '\n')
                prev_phone = ""
                current_phone = phone
                current_sentence = s_id
                current_id = 1
                count = 1
            elif phone == current_phone:
                current_id += 1
                count += 1
            else:
                if count <= min_count and prev_phone != "":
                    #print("writing: " + prev_phone + " x" + str(count))
                    for i in range(count):
                        fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + prev_phone + '\n')
                elif count <= min_count:
                    #print("writing \'sil\' x" + str(count))
                    for i in range(count):
                        fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',sil\n')
                    prev_phone = 'sil'
                else:
                    #print("writing: " + current_phone + " x" + str(count))
                    for i in range(count):
                        fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + current_phone + '\n')
                    prev_phone = current_phone
                current_phone = phone
                count = 1
                current_id += 1
        if count <= min_count:
            for i in range(count):
                fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + prev_phone + '\n')
        else:
            for i in range(count):
                fw.write(current_sentence + "_" + str(current_id - count + 1 + i) + ',' + current_phone + '\n')

