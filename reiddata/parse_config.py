import numpy as np


def parse_model_cfg(path):
    file = open(path, 'r')
    lines = file.read().split('\n') # store the lines in a list等价于readlines
    lines = [x for x in lines if x and not x.startswith('#')] # 去掉空行和以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)

    mdefs = []  # module definitions
    for line in lines: # '[net]'
        if line.startswith('['):  # 这是cfg文件中一个层(块)的开始
            mdefs.append({})      # 添加一个字典
            mdefs[-1]['type'] = line[1:-1].rstrip() # 把cfg的[]中的块名作为键type的值  <class 'list'>: [{'type': 'net'}]
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0    # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=") # 按等号分割 key:'batch' val:16
            key = key.rstrip()         #  key(去掉右空格)

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors shape: (9, 2)
            else:
                mdefs[-1][key] = val.strip()
    return mdefs


def parse_data_cfg(path):
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines() # 返回列表，包含所有的行。

    for line in lines: # 'classes= 1\n'
        line = line.strip()
        if line == '' or line.startswith('#'): # 去掉空白行和以#开头的注释行
            continue
        key, val = line.split('=') # 按等号分割 key:'classes'  value:' 1'
        options[key.strip()] = val.strip()

    return options
