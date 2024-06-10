LABEL_LENGTH=6



def write_to_file(path,write_list):
    file_handle = open(path,'w')
    for write_info in write_list:
        file_handle.write(str(write_info))
        file_handle.write('\n')
    file_handle.close()
    print("Write to File: %s" % path)


def read_from_file(path):
    # get label0 for the targeted content input txt
    output_list = list()
    with open(path) as f:
        for line in f:
            this_label = line[:-1]
            if len(this_label)<LABEL_LENGTH and not this_label == '-1':
                    for jj in range(LABEL_LENGTH-len(this_label)):
                        this_label = '0'+ this_label
            # line = u"%s" % line
            output_list.append(this_label)


    return output_list