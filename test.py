import torch
import opts
from dataset import Test_Data
from models import test_model as model
from vis_tool import Visualizer


class tester():
    def __init__(self, model_path):
        # options
        self.opts = opts.test_options().parse()
        # training data
        self.test_data = Test_Data(self.opts)
        self.test_dataloader = self.test_data.test_loader
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # model
        self.model = model(self.opts, self.device, model_path)
        # visualizer
        self.test_vis = Visualizer(env='testing')


    def test_process(self):
        """test code over the entire testing data set"""
        sum_loss = 0.0
        sum_acc = 0.0
        for i, data in enumerate(self.test_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # forward
            outputs = self.model.network(inputs)

            # loss
            loss = self.model.loss_func(outputs, labels)
            sum_loss += loss.item()

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            sum_acc += 100. * float(predicted.eq(labels.data).cpu().sum())

            # visualization of loss and accuracy
            if i % self.opts.test_interval == 0:
                batch_loss = sum_loss / ((i+1) * self.opts.batch_size)
                batch_acc = sum_acc / ((i+1) * self.opts.batch_size)
                print('[iter: %d] Loss: %.03f | Acc: %.03f%% '
                      % (i // self.opts.test_interval, batch_loss, batch_acc))
                loss_acc = {'test_loss:':batch_loss, 'test_acc':batch_acc}
                self.visual(self.test_vis, loss_acc)

    # test
    def test(self):
        print('start testing...')
        self.test_process()

    # visualizer
    def visual(self, vis, loss_acc):
        vis.plot_many(loss_acc)



if __name__ == '__main__':
    test_opts = opts.test_options()
    test = tester(test_opts.checkpoint)
    test.test()