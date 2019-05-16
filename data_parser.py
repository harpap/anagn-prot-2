import csv
import numpy

def load_spambase():
    file1 = 'spambase/spambase.data'
    raw_data = open(file1, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=2)
    l = list(reader)
    rev_l = list(zip(*l))
    y = rev_l.pop(-1)
    X = list(zip(*rev_l))
    data = numpy.array(X).astype('float')
    target = numpy.array(y).astype('int')
    return None, data, target

def load_occupancy_data1():
    file1 = 'occupancy_data/datatest.txt'
    raw_data = open(file1, 'rt')
    reader = csv.reader(raw_data, delimiter=',')
    l = list(reader)
    header = l.pop(0)
    rev_l = list(zip(*l))
    rev_l.pop(0)
    rev_l.pop(0)
    y = rev_l.pop(-1)
    X = list(zip(*rev_l))
    data = numpy.array(X).astype('float')
    target = numpy.array(y).astype('int')
    return header, data, target
    
def load_occupancy_data2():
    file1 = 'occupancy_data/datatest2.txt'
    raw_data = open(file1, 'rt')
    reader = csv.reader(raw_data, delimiter=',')
    l = list(reader)
    header = l.pop(0)
    rev_l = list(zip(*l))
    rev_l.pop(0)
    rev_l.pop(0)
    y = rev_l.pop(-1)
    X = list(zip(*rev_l))
    data = numpy.array(X).astype('float')
    target = numpy.array(y).astype('int')
    return header, data, target

def load_occupancy_data3():
    file1 = 'occupancy_data/datatraining.txt'
    raw_data = open(file1, 'rt')
    reader = csv.reader(raw_data, delimiter=',')
    l = list(reader)
    header = l.pop(0)
    rev_l = list(zip(*l))
    rev_l.pop(0)
    rev_l.pop(0)
    y = rev_l.pop(-1)
    X = list(zip(*rev_l))
    data = numpy.array(X).astype('float')
    target = numpy.array(y).astype('int')
    return header, data, target
    