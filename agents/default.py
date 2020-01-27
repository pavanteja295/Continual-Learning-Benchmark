from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        self.noise = 'Noise' in self.config['model_name'] or 'MLP_Inc_Tasks' in self.config['model_name']
        # import pdb; pdb.set_trace()
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.exp_name = agent_config['exp_name']
        self.warmup = agent_config['warmup']
        self.init_optimizer()
        self.reset_optimizer = agent_config['reset_opt']
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario

        self.writer = SummaryWriter(log_dir="runs/" + self.exp_name)
        self.task_num = 0

    def init_optimizer(self, params=None):
        if not params:
            params = self.model.parameters()
        
        optimizer_arg = {'params':params,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
            optimizer_arg['nesterov'] = self.config['nesterov']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.2)
        
    def create_model(self):
        cfg = self.config
        
        if self.noise:
            params = self.config['out_dim']
            noise_type = self.config['noise_type']
            print('====================== Noise Type ==   ',noise_type,'=======================')
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](tasks=params, noise_type= noise_type) #
        else:
            # import pdb; pdb.set_trace()
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        

  
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x, task_n=''):
        if self.noise:
            return self.model.forward(x, task_n)
        return self.model.forward(x)

    def predict(self, inputs, task_n=''):
        self.model.eval()
        out = self.forward(inputs, task_n)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader, task_n=''):
        # this might possibly change for other incremental scenario
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            # print(task_n)
            output = self.predict(input, task_n)
            loss = self.criterion(output, target, task)
            losses.update(loss, input.size(0))        
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc, losses

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t,t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:,:self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def update_model(self, inputs, targets, tasks, task_n=''):

        out = self.forward(inputs, task_n)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def freeze(self, task_n):
        
        if task_n == '2':
            if self.config['freeze_core']:
                for param in self.model.linear.parameters():
                        param.requires_grad = False
            else:
                for param in self.model.linear.parameters():
                        param.requires_grad = True

        for key, val in self.model.last.items():
            if key != task_n:
                for param in self.model.last[key].parameters():
                    param.requires_grad = False
            else:
                for param in self.model.last[key].parameters():
                    param.requires_grad = True
        if self.noise:
            for key, val in self.model.noise_list.items():
                if key != task_n:
                    for param in self.model.noise_list[key].parameters():
                        param.requires_grad = False
                else:
                    for param in self.model.noise_list[key].parameters():
                        param.requires_grad = True

        t = 0
        for p in self.model.parameters():
            if p.requires_grad:
                t = t + 1
        print('====================== trainable params ==   ',t,'=======================')
                
 
    def learn_batch(self, train_loader, val_loader=None, epochs=[0, 40], task_n=''):
        itrs = 0
        if epochs[0] == 0:  # Only for the first epoch of each task or classReset optimizer before incrementally learning
           self.task_num +=1
           if self.reset_optimizer:
                self.log('Optimizer is reset!')
                self.freeze(task_n)
                self.init_optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters()))
        
        data_timer = Timer()
        batch_timer = Timer()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        for epoch in range(epochs[0], epochs[1]):
            # grads visualization
            # for i, param in enumerate(self.model.linear.parameters()):
            #     if not i :
            #         print(param.data[0,:10])

            # # import pdb; pdb.set_trace()
            # for key, val in self.model.last.items():
            #     if key != task_n:
            #         for param in self.model.last[key].parameters():
            #         #    import pdb; pdb.set_trace()
            #             if len(param.data.shape) > 1: 
            #                 print(param.data[0,:10])
            #     else:
            #         for param in self.model.last[key].parameters():
            #             # import pdb; pdb.set_trace()
            #             if len(param.data.shape) > 1: 
            #                 print(param.data[0,:10])
            print('====================== Noise params =======================')
            if self.noise:
                for key, val in self.model.noise_list.items():
                    if key != task_n:
                        for param in self.model.noise_list[key].parameters():
                                print(param.data[:10])
                    else:
                        for param in self.model.noise_list[key].parameters():
                                print(param.data[:10])

                    
            # self.writer = SummaryWriter(log_dir="runs/" + self.exp_name)
            if epoch == 0 and self.warmup:
                self.warm = WarmUpLR(self.optimizer, len(train_loader) * self.warmup)
            
            if epoch > self.warmup:
                self.scheduler.step(epoch)
            

            # # # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc') 

            for i, (input, target, task) in enumerate(train_loader):
                # iteration count
                self.n_iter = (epoch) * len(train_loader) + i + 1
                if epoch < self.warmup:
                    self.warm.step()
                data_time.update(data_timer.toc())  # measure data loading time
                if self.gpu:
                        input = input.cuda()                                                                                                                                                                                                                                                
                        target   = target.cuda()
                
                loss, output = self.update_model(input, target, task, task_n)
                input = input.detach()
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))
                self.writer.add_scalar('Run' + str(self.config['run_num']) + '/Loss/train' + task_n, losses.avg, self.n_iter)
                self.writer.add_scalar('Run' + str(self.config['run_num']) + '/Accuracy/train' + task_n, acc.avg, self.n_iter)
                
                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))                                                                                                                                                                                                                                                                     

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
               acc_val, loss_val =  self.validation(val_loader, task_n)
               self.writer.add_scalar('Run' + str(self.config['run_num']) + 'Loss/test' + task_n, loss_val.avg, self.n_iter)
               self.writer.add_scalar('Run' + str(self.config['run_num']) + 'Accuracy/test' + task_n, acc_val.avg, self.n_iter)
            self.writer.close()

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            # Get rid of 'module' before the name of stat                               es                     
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys(): # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
