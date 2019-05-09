import torch
import opts
from dataset import Train_Data
from models import train_model as model
from vis_tool import Visualizer
import os
from test import tester as valider


class trainer():
    def __init__(self):
        # options
        self.opts = opts.train_options().parse()
        # training data
        self.train_data = Train_Data(self.opts)
        self.train_dataloader = self.train_data.train_loader
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # model
        self.model = model(self.opts, self.device)
        # visualizer
        self.train_vis = Visualizer(env='Training')

    # adjustable learning rate
    def adjust_lr(self, epoch):
        lr = self.opts.lr * (0.5 ** (epoch // self.opts.step))
        if lr < 1e-8:
            lr = 1e-8
        return lr

    def train_process(self, start_epoch):
        """the common training code for first train and resume train"""
        for epoch in range(start_epoch, self.opts.num_epochs):
            print('\nepoch: %d'%(epoch))
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            lr = self.adjust_lr(epoch - 1)
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr
            print('epoch={}, lr={}'.format(epoch, self.model.optimizer.param_groups[0]['lr']))

            for i, data in enumerate(self.train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.model.optimizer.zero_grad()

                # forward + backward
                outputs = self.model.network(inputs)
                loss = self.model.loss_func(outputs, labels)
                loss.backward()
                self.model.optimizer.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += predicted.eq(labels.data).cpu().sum()
                batch_acc = 100. * float(predicted.eq(labels.data).cpu().sum()) / self.opts.batch_size
                # batch_acc = 100. * correct / ((i+1)*self.opts.batch_size)
                batch_loss = sum_loss / (i + 1)
                print('[epoch: %d/%d] [iter: %d/%d] Loss: %.03f | Acc: %.03f%% '
                      % (epoch, self.opts.num_epochs, i, len(self.train_dataloader), loss.item(), batch_acc))
                losses = {'loss:':batch_loss, 'acc':batch_acc}
                vis_loss(self.train_vis, losses)

            if epoch % self.opts.train_interval == 0:
                # save the model and its parameters on every train_interval epochs
                save = save_model(self.opts.model_folder, self.model.network, epoch)
                save.save_checkpoint()
                # valid the model on every train_interval epochs
                valid = valider(self.opts.model_folder + '{}.pkl'.format(epoch))
                valid.test()

    # first training
    def first_train(self):
        self.train_process(self.opts.start_epoch)

    # resume training
    def resume_train(self):
        # load model parameters
        checkpoint = torch.load(self.opts.checkpoint)
        self.model.network.load_state_dict(checkpoint['model'].state_dict())
        # train
        self.train_process(self.opts.start_epoch)

    def train(self):
        if self.opts.resume:
            print('resume training at epoch {}...'.format(self.opts.start_epoch))
            self.resume_train()
        else:
            print('start first training...')
            self.first_train()

# visualizer
def vis_images(vis, images):
    vis.img_many(images)

def vis_loss(vis, losses):
    vis.plot_many(losses)

class save_model():
    def __init__(self, model_folder, models, epoch):
        self.model_folder = model_folder
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        self.models = models
        self.epoch = epoch

    def save_checkpoint(self):
        checkpoint_path = self.model_folder + '{}.pkl'.format(self.epoch)
        state_dict = {'epoch': self.epoch, 'model': self.models}
        torch.save(state_dict, checkpoint_path)
        print("Checkpoint saved to {}".format(checkpoint_path))



if __name__ == '__main__':
    train = trainer()
    train.train()