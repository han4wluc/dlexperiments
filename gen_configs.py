
import json
import itertools
import shutil
import os


configs = {
  "batch_size": 64,
  "shuffle": True,
  "dataset": "MNIST",
  "network": "CNN_1",
  "criterion": "nll_loss",
  "optim_": "SGD",
  "transform": [{
    "name": "ToTensor",
  }, {
    "name": "Normalize",
    "mean": 0.1307,
    "std": 0.3081
  }],
  "momentum": 0,
  "epochs": 10
}

learning_rates = [0.01, 0.05, 0.1]
weight_decays = [0, 0.5]

config_dir = './experiment_1/configs'

if os.path.exists(config_dir):
  shutil.rmtree(config_dir)
os.mkdir(config_dir);

for i, (learning_rate,weight_decay) in enumerate(itertools.product(learning_rates,weight_decays)):
    config = configs
    config["learning_rate"] = learning_rate
    config["weight_decay"] = weight_decay

    with open('configs/config_' + str(i) + '.json', 'w') as outfile:
      json.dump(config, outfile, indent=2, sort_keys=True)

print('config jsons generated')
