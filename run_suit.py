
import os
import shutil

experiment_name = 'experiment_1'

configs = os.listdir('./' + experiment_name + '/configs')

results_dir = './' + experiment_name + '/results'
if os.path.exists(results_dir):
  shutil.rmtree(results_dir)
os.mkdir(results_dir);


for config_file in configs:
  config_name = config_file.replace('.json', '')

  command = 'python main.py ' + '--config_file ./' + experiment_name +  '/configs/' + config_file + ' --output_dir ' + './' + experiment_name + '/results/' + config_name + ' --data_dir ./' + experiment_name + '/data'
  print(command)
  os.system(command)

  command2 = 'python gen_losses_graph.py --input_path ./' + experiment_name + '/results/' + config_name + '/losses.json --output_path ' +  './' + experiment_name + '/results/' + config_name + '/graph.jpg'
  os.system(command2)
  print('image saved')

