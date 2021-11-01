class StepDecay():
    """Updates optimizer parameters according to a step decay learning rate schedule.
    
    Attributes:
        optimizer: instance of torch.optim.AdamW or torch.optim.SGD
        step_size: number of epochs at each learning rate value
        gamma: decay factor
        lr: current learning rate
        epoch: track the current epoch
    """
    def __init__(self, optimizer, params):
        step_size, gamma = params
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lr = optimizer.param_groups[0]['lr']
        self.epoch = 0
        
    def step(self):
        """Increment the optimizer parameters. Called at the end of each epoch during training.
        """
        if (self.epoch+1)%self.step_size==0:
            self.lr *= self.gamma
        
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.epoch+=1
        
class OneCycle():
    """Updates optimizer parameters according to a one cycle learning rate schedule.
    
    Attributes:
        optimizer: instance of torch.optim.AdamW
        steps_up: number of steps to increase to peak learning rate
        steps_down: number of steps to decrease back to initial learning rate from peak
        steps_final: number of steps to ramp down to final learning rate at end of training
        init_lr: initial learning rate
        max_lr: peak learning rate
        final_lr: learning rate at end of final decrease
        init_beta1: initial (maximum) beta1
        min_beta1: minimum beta1 reached halfway through the cycle
        beta2: beta2
        start_epoch: Optional; steps the optimizer start_epoch times on initialization
        epoch: track the current epoch
        lr = current learning rate
        beta1 = current beta1
    """
    def __init__(self, optimizer, params):
        if len(params)==9:
            steps_up, steps_down, steps_final, init_lr, max_lr,\
            final_lr, init_beta1, min_beta1, beta2 = params 
            start_epoch = 0
        elif len(params)==10:
            steps_up, steps_down, steps_final, init_lr, max_lr,\
            final_lr, init_beta1, min_beta1, beta2, start_epoch = params 
        else:
            print('The scheduler was passed ', str(len(params)), ' parameters, but expected 9 or 10')
            
        self.optimizer = optimizer
        self.steps_up = steps_up
        self.steps_down = steps_down
        self.steps_final = steps_final
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.min_beta1 = min_beta1
        self.init_beta1 = init_beta1
        self.beta2 = beta2
        self.epoch = 0
        self.lr = init_lr
        self.beta1 = init_beta1
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[0]['betas'] = (self.init_beta1, self.beta2)
        for epoch in range(int(start_epoch)):
            self.step()
        
    def step(self):
        
        """Increment the optimizer parameters. Called at the end of each epoch during training.
        """
        if self.epoch<self.steps_up:
            self.lr += (self.max_lr-self.init_lr)/self.steps_up
            self.beta1 += (self.min_beta1-self.init_beta1)/self.steps_up
        elif self.epoch >= self.steps_up and self.epoch < self.steps_down+self.steps_up:
            self.lr -= (self.max_lr-self.init_lr)/self.steps_down
            self.beta1 -= (self.min_beta1-self.init_beta1)/self.steps_down
        elif self.epoch >= self.steps_down+self.steps_up and self.epoch < self.steps_down+self.steps_up+self.steps_final:
            self.lr -= (self.init_lr-self.final_lr)/self.steps_final
    
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[0]['betas'] = (self.beta1, self.beta2)
        self.epoch+=1