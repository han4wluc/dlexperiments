
import argparse
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Initialize argument parse object
parser = argparse.ArgumentParser()

# This would be an argument you could pass in from command line
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--title', type=str, required=False)

# Parse the arguments
inargs = parser.parse_args()
# inputPath = inargs.input_path
# outputPath = inargs.output_path
title = inargs.title or 'Training Loss'


mpl.rcParams['figure.figsize'] = (10.0, 8.0)

# data = np.genfromtxt(inargs.input_path, delimiter='\t', skip_header=0, names=True)


with open(inargs.input_path) as json_file:
    losses = json.load(json_file)


print(losses)

train_loss = []
test_loss = []

for l in losses:
  train_loss.append(l["train_loss"])
  test_loss.append(l["test_loss"])


red_patch = mpatches.Patch(color='b', label='train_loss')
green_patch = mpatches.Patch(color='g', label='test_loss')
plt.legend(handles=[red_patch, green_patch])

plt.title(title)
plt.xlabel('loss')
plt.ylabel('epoch')
plt.plot(train_loss, c='b')
plt.plot(test_loss, c='g')
plt.savefig(inargs.output_path)

