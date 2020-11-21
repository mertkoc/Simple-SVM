#!/usr/bin/env python

import sys
import os
from subprocess import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import unravel_index
import array_to_latex as a2l

# A = np.array([[1.23456, 23.45678],[456.23, 8.239521]])


if len(sys.argv) <= 1:
    print('Usage: {0} training_file [testing_file]'.format(sys.argv[0]))
    raise SystemExit

# svm, grid, and gnuplot executable files

is_win32 = (sys.platform == 'win32')
if not is_win32:
    svmscale_exe = "../svm-scale"
    svmtrain_exe = "../svm-train"
    svmpredict_exe = "../svm-predict"
    grid_py = "./grid.py"
    gnuplot_exe = "/usr/bin/gnuplot"
else:
    # example for windows
    svmscale_exe = r"..\windows\svm-scale.exe"
    svmtrain_exe = r"..\windows\svm-train.exe"
    svmpredict_exe = r"..\windows\svm-predict.exe"
    gnuplot_exe = r"c:\tmp\gnuplot\binary\pgnuplot.exe"
    grid_py = r".\grid.py"

assert os.path.exists(svmscale_exe), "svm-scale executable not found"
assert os.path.exists(svmtrain_exe), "svm-train executable not found"
assert os.path.exists(svmpredict_exe), "svm-predict executable not found"
assert os.path.exists(gnuplot_exe), "gnuplot executable not found"
assert os.path.exists(grid_py), "grid.py not found"

train_pathname = sys.argv[1]
assert os.path.exists(train_pathname), "training file not found"
file_name = os.path.split(train_pathname)[1]
# scaled_file = file_name + ".scale"
model_file = file_name + ".model"
# range_file = file_name + ".range"

if len(sys.argv) > 2:
    test_pathname = sys.argv[2]
    file_name = os.path.split(test_pathname)[1]
    assert os.path.exists(test_pathname), "testing file not found"
    # scaled_test_file = file_name + ".scale"
    predict_test_file = file_name + ".predict"


# cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
# print('Scaling training data...')
# Popen(cmd, shell = True, stdout = PIPE).communicate()
# cmd = '{0} -svmtrain -t 0 "{1}" -gnuplot "{2}" "{3}"'.format(grid_py, svmtrain_exe, gnuplot_exe, scaled_file)
# print('Cross validation...')
# f = Popen(cmd, shell = True, stdout = PIPE).stdout
#
# line = ''
# while True:
# 	last_line = line
# 	line = f.readline()
# 	if not line: break
# c,g,rate = map(float,last_line.split())
#
# print('Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

def give_accuracy_linear(train_pathname, test_pathname, model_file, predict_test_file, C, remove_files=True):
    cmd = '{0} -t 0 -c {1} "{2}" "{3}"'.format(svmtrain_exe, C, train_pathname, model_file)
    # print('Training...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    # print('Output model: {0}'.format(model_file))
    # if len(sys.argv) > 2:
    # cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
    # print('Scaling testing data...')
    # Popen(cmd, shell = True, stdout = PIPE).communicate()

    cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, test_pathname, model_file, predict_test_file)
    # print('Testing...')
    f = Popen(cmd, shell=True, stdout=PIPE).stdout
    line = ''
    while True:
        last_line = line
        line = f.readline()
        if not line: break

    # print('Output prediction: {0}'.format(predict_test_file))
    if remove_files:
        os.remove(model_file)
        os.remove(predict_test_file)
    return float(last_line.split()[2][:-1])


def give_accuracy_RBF(train_pathname, test_pathname, model_file, predict_test_file, gamma, C, remove_files=True):
    cmd = '{0} -t 2 -g {4} -c {1} "{2}" "{3}"'.format(svmtrain_exe, C, train_pathname, model_file, gamma)
    # print('Training...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    # print('Output model: {0}'.format(model_file))
    # if len(sys.argv) > 2:
    # cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
    # print('Scaling testing data...')
    # Popen(cmd, shell = True, stdout = PIPE).communicate()

    cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, test_pathname, model_file, predict_test_file)
    # print('Testing...')
    f = Popen(cmd, shell=True, stdout=PIPE).stdout
    line = ''
    while True:
        last_line = line
        line = f.readline()
        if not line: break

    # print('Output prediction: {0}'.format(predict_test_file))

    if remove_files:
        os.remove(model_file)
        os.remove(predict_test_file)
    return float(last_line.split()[2][:-1])


def cross_validation_n_folds(train_pathname, gamma, C, n=5, remove_files=True):
    with open(train_pathname) as f:
        lines = f.readlines()  # list containing lines of file
        number_of_training_samples = len(lines)
        half_choice = np.random.choice(2)  # Randomly choose first or second half
        # print(half_choice)
        train_vals = np.random.choice(np.arange(half_choice * number_of_training_samples // 2,
                                                (half_choice + 1) * number_of_training_samples // 2),
                                      number_of_training_samples // 2, replace=False)
        # print(train_vals)
        # Now divide the set into 5 folds
        set_size = len(train_vals) // n
        # Write these to a new file
        set_number = 1
        for iteration, value in enumerate(train_vals):
            with open(f"set{set_number}.txt", "a+") as fw:
                fw.write(lines[value - 1])
                if (iteration + 1) % set_size == 0:
                    set_number += 1

    cross_val_acc = np.zeros(n)
    for i in range(1, n + 1):
        # print(i)
        set_name = f"set{i}.txt"  # This our test file
        # Merge other text files
        to_be_merged_set = ''
        merged_text = f"merged{i}.txt"
        for j in range(1, n + 1):
            if j != i:
                # print(j)
                to_be_merged_set += f"set{j}.txt "
        cmd = f"cat {to_be_merged_set} > {merged_text}"
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        model_file = merged_text + ".model"
        predict_test_file = set_name + ".predict"
        # Train and get the accuracy
        accuracy = give_accuracy_RBF(merged_text, set_name, model_file, predict_test_file, 2, 2, remove_files=False)
        cross_val_acc[i - 1] = accuracy
    if remove_files:
        # Now remove the files
        for i in range(1, n + 1):
            # print(i)
            set_name = f"set{i}.txt"  # This our test file
            # Merge other text files
            to_be_merged_set = ''
            merged_text = f"merged{i}.txt"
            model_file = merged_text + ".model"
            predict_test_file = set_name + ".predict"
            os.remove(set_name)
            os.remove(merged_text)
            os.remove(model_file)
            os.remove(predict_test_file)
    return np.mean(cross_val_acc)


accuracies = []
for c in range(-4, 9):
    C = 2.0 ** c
    model_file = os.path.split(train_pathname)[1] + f"C_{C}" + ".model"
    predict_test_file = os.path.split(test_pathname)[1] + f"C_{C}" + ".predict"
    accuracies.append(give_accuracy_linear(train_pathname, test_pathname, model_file, predict_test_file, C))

print(accuracies)
plt.figure("Accuracies")
plt.plot([c for c in range(-4, 9)], accuracies)
plt.xlabel("lg(C)")
plt.ylabel("Accuracies (%)")
plt.savefig("Accuracies.png", bbox_inches='tight')
plt.close()

### Now cross validation ###
grid_mtx = np.zeros((13, 13))
for i in range(13):
    for j in range(13):
        C = 2.0 ** (i - 4)
        gamma = 2.0 ** (j - 4)
        avg_acc = cross_validation_n_folds(train_pathname, gamma, C, n=5)  # 5-Fold cross validation
        grid_mtx[i, j] = avg_acc

print(grid_mtx)
a2l.to_ltx(grid_mtx, frmt='{:.2f}', arraytype='array')
highest_acc = np.amax(grid_mtx)
indicies = unravel_index(grid_mtx.argmax(), grid_mtx.shape)
best_gamma = 2.0 ** (indicies[1] - 4)
best_C = 2.0 ** (indicies[0] - 4)
print(f"Highest validation accuracy is {highest_acc:.2f}% at {indicies[0] + 1}th row and {indicies[1] + 1}th column")
print(f"Which means best C={best_C}, best gamma={best_gamma}")

train_pathname = sys.argv[1]
file_name = os.path.split(train_pathname)[1]
model_file = file_name + ".model"

if len(sys.argv) > 2:
    test_pathname = sys.argv[2]
    file_name = os.path.split(test_pathname)[1]
    predict_test_file = file_name + ".predict"

# Now re-train using the highest values

accuracy = give_accuracy_RBF(train_pathname, test_pathname, model_file, predict_test_file, best_gamma, best_C)
print(f"Highest accuracy using best gamma and C on test data is {accuracy}%")

# plt.figure("Grid accuracies for Cross Validation")
# plt.plot(grid_mtx)
# plt.xlabel("lg(C)")
# plt.ylabel("lg(gamma)")
# plt.savefig("Cross_val.png", bbox_inches='tight')
# plt.close()
