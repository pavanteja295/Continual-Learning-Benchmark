import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
import torchvision

# hello world
def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    if args.n_permutation>0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                             args.n_permutation,
                                                                             remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=args.first_split_size,
                                                                          other_split_sz=args.other_split_size,
                                                                          rand_split=args.rand_split,
                                                                          remap_class=not args.no_class_remap)

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef, 'exp_name' : args.exp_name, 'warmup':args.warm_up, 'nesterov':args.nesterov, 'run_num' :args.run_num, 'freeze_core':args.freeze_core, 'reset_opt':args.reset_opt, 'noise_type':args.noise_type }
                    
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#parameter of model:',agent.count_parameter())
    

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)
    #import pdb; pdb.set_trace()
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    loss_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        epochs = args.epochs[-1]
        agent.learn_batch(train_loader, val_loader, [0, epochs])

        acc_table['All'] = {}
        loss_table['All'] = {}
        acc_table['All']['All'], loss_table['All']['All'] = agent.validation(val_loader)

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(len(task_names)):
            train_name = task_names[i]
            print('======================',train_name,'=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                      batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            epochs = args.epochs[i] if len(args.epochs) - 1  else args.epochs[0]
            # split the epochs into multiple sub epochs
            # to perform validation after every 10 in between if epochs are 80 --> 20, 30, 40, 50, 40 , 80 
            # helps us in better understanding how the degradation happens
            for epoch_10 in range(int(epochs / args.old_val_freq)):
                    agent.learn_batch(train_loader, val_loader, epochs=[epoch_10 * args.old_val_freq, (epoch_10 + 1)* args.old_val_freq], task_n=train_name)
                    # Evaluate
                    acc_table[train_name] = OrderedDict()
                    loss_table[train_name] = OrderedDict()
                    writer = SummaryWriter(log_dir="runs/" + agent.exp_name)
                    for j in range(i+1):
                        val_name = task_names[j]
                        print('validation split name:', val_name)
                        val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                        val_loader = torch.utils.data.DataLoader(val_data,
                                                                batch_size=args.batch_size, shuffle=False,
                                                                num_workers=args.workers)
                        acc_table[val_name][train_name], loss_table[val_name][train_name] = agent.validation(val_loader, val_name)
            
                        # tensorboard 
                        # agent.writer.reopen()
                        print('logging for Task  {} while training {}'.format(val_name, train_name))
                        print('logging', int(train_name) + (epoch_10 + 1) * 0.1 )
                
                        writer.add_scalar('Run' + str(args.run_num) +  '/CumAcc/Task' + val_name, acc_table[val_name][train_name].avg, float(int(train_name)) * 100 + (epoch_10 + 1) * args.old_val_freq)
                        writer.add_scalar('Run' + str(args.run_num) +  '/CumLoss/Task' + val_name, loss_table[val_name][train_name].avg, int(train_name) * 100 + (epoch_10 + 1)* args.old_val_freq )
                        writer.close()
            # if i == 1:
                #after the first task freeze some weights:

        # def funcname(self, parameter_list):
        #         npimg = img.numpy().transpose((1,2,0))
        #         min_val = np.min(npimg, keepdims =True)
        #         print('min',min_val)
        #         max_val = np.max(npimg, keepdims =True)
        #         print('max',max_val)
        #         inp = (npimg-min_val)/(max_val-min_val)
        #         # inp = npimg
        #         plt.imshow(inp)
        #     pass
    return acc_table, task_names

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', type=str, default='seperate', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='noise_based', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='Noise_Net', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=0, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[120], #, 120, 160, 300
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=10, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--exp_name', dest='exp_name', default='default', type=str,
                        help="Exp name to be added to the suffix")
    parser.add_argument('--warm_up', type=int, default=0, help='warm up training phase')
    parser.add_argument('--nesterov',  default=False, action='store_true', help='nesterov up training phase')
    parser.add_argument('--epochs', nargs="+", type=int, default=[4], 
                     help="Randomize the order of splits")
    parser.add_argument('--old_val_freq', type=int, default=1, 
                     help="frequency to log validation error of seen tasks")

    parser.add_argument('--freeze_core',  default=False, action='store_true',
                     help="freeze the core network")
    parser.add_argument('--reset_opt',  default=False, action='store_true',
                     help="freeze the core network")
    

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    import torch
    import random
    import numpy as np
    
    # for reproducibility



    # necessary for reproducing results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):
            torch.manual_seed(r)
            random.seed(r)
            np.random.seed(r)
            # Run the experiment
            
            args.run_num = r + 1
            acc_table, task_names = run(args)
            print(acc_table)

            # import pdb; pdb.set_trace()
            # after the end of task 5 report the accuracies
            final_tsk_acc = 0
            final_tsk = [i for i in acc_table.keys()][-1]
            for tsk in acc_table.keys():
                final_tsk_acc += acc_table[tsk][final_tsk].avg
            print('===Final accuracy on task:',final_tsk, final_tsk_acc / len(acc_table.keys()),'===')

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * len(task_names)
            for i in range(len(task_names)):
                train_name = task_names[i]
                cls_acc_sum = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name].avg
                avg_acc_history[i] = cls_acc_sum / (i + 1)
                print('Task', train_name, 'average acc:', avg_acc_history[i])

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            # Print the summary so far
            print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    for reg_coef,v in avg_final_acc.items():
        print('reg_coef:', reg_coef,'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
