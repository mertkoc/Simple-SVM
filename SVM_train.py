import numpy as np
import glob, os
import json
import time

training_list = []
test_list = []

with open('ncRNA_s.train.txt') as f:
    lines = f.readlines()  # list containing lines of file
    columns = []  # To store column names
    # missing_lines = []
    i = 1
    for line in lines:
        # line = line.strip()  # remove leading/trailing white spaces
        if line:
            # line_added = False
            # if i == 1:
            d = {}
            columns = [item.strip() for item in line.split(' ')]
            d['value'] = columns[0]
            d['attributes'] = []
            # d['attributes'] = np.zeros(8)
            prev_value = 1
            for indx, column in enumerate(columns[1:]):
                if column:
                    data = [item.strip() for item in column.split(':')]
                    attribute_val = int(data[0])
                    if attribute_val == prev_value:
                        # d['attributes'][int(data[0]) - 1] = float(data[1])
                        d['attributes'].append(float(data[1]))
                    else:
                        # if not line_added:
                            # missing_lines.append(i)
                        # line_added = True
                        while prev_value != attribute_val:
                            d['attributes'].append(0.0)
                            prev_value += 1
                        d['attributes'].append(float(data[1]))
                    prev_value += 1
            # for indx, data in enumerate(columns):
            #     if indx == 1:
            i = i + 1
            training_list.append(d)

with open('ncRNA_s.test.txt') as f:
    lines = f.readlines()  # list containing lines of file
    columns = []  # To store column names
    # missing_lines = []
    i = 1
    for line in lines:
        # line = line.strip()  # remove leading/trailing white spaces
        if line:
            # line_added = False
            # if i == 1:
            d = {}
            columns = [item.strip() for item in line.split(' ')]
            d['value'] = columns[0]
            d['attributes'] = []
            # d['attributes'] = np.zeros(8)
            prev_value = 1
            for indx, column in enumerate(columns[1:]):
                if column:
                    data = [item.strip() for item in column.split(':')]
                    attribute_val = int(data[0])
                    if attribute_val == prev_value:
                        # d['attributes'][int(data[0]) - 1] = float(data[1])
                        d['attributes'].append(float(data[1]))
                    else:
                        # if not line_added:
                            # missing_lines.append(i)
                        # line_added = True
                        while prev_value != attribute_val:
                            d['attributes'].append(0.0)
                            prev_value += 1
                        d['attributes'].append(float(data[1]))
                    prev_value += 1
            # for indx, data in enumerate(columns):
            #     if indx == 1:


            i = i + 1
            test_list.append(d)


### TRAINING SESSION ###

for c in range(-4, 9):
    C = 2 ** c
