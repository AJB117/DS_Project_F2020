#!/usr/bin/python3
import ast

class Model:
    def __init__(self, name, matrix):
        self.name = name
        self.TP = matrix[0][0]
        self.FN = matrix[0][1]
        self.FP = matrix[1][0]
        self.TN = matrix[1][1]
        try:
            self.precision = self.TP/(self.TP+self.FP)
            self.recall = self.TP/(self.TP+self.FN)
            self.F1 = (2*self.precision*self.recall) / (self.precision+self.recall)
        except:
            self.precision = None
            self.recall = None
            self.F1 = None
    
    def __str__(self):
        return f'''{self.name}:
        precision={self.precision}
        recall={self.recall}
        F1={self.F1}'''

def get_results(i, append):
    while '-' in lines[i]:
        # get name
        name = lines[i][2:-2]
        name += append
        # get matrix
        matrix1 = [int(lines[i+1][4:-2].split()[0]), int(lines[i+1][3:-2].split()[1])]
        matrix2 = [int(lines[i+2][3:-3].split()[0]), int(lines[i+2][3:-3].split()[1])]
        matrix = [matrix1, matrix2]
        m = Model(name, matrix)
        print(m)

        # next
        i += 3

output = open('modelperformance.md', 'r')
lines = output.readlines()

# line numbers passed here should be the line number of the first model name (0-indexed)
get_results(15-1, '-STANDARDSAMPLING')
get_results(62-1, '-OVERSAMPLING')
get_results(110-1, '-UNDERSAMPLING')
