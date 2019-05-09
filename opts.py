import argparse

# parser
class train_options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='comparison of '
                                                          'squeeze-and-excitation block '
                                                          'and attention-augmented-convolution'
                                                          'on ResNet')
        # train related
        self.parser.add_argument('--num_epochs', type=int, default=1000)
        self.parser.add_argument('--batch_size', type=int, default=2000)
        self.parser.add_argument('--lr', type=float, default=0.05)
        self.parser.add_argument('--step', type=int, default=60)
        self.parser.add_argument('--train_interval', type=int, default=20)
        self.parser.add_argument('--case', type=str, default='aa_resnet50')
        self.parser.add_argument('--pretrained', type=bool, default=False)
        self.parser.add_argument('--num_classes', type=int, default=10)
        self.parser.add_argument('--model_folder', type=str, default='model_para/')

        # resume train related
        self.parser.add_argument('--resume', type=bool, default=False)
        self.parser.add_argument('--start_epoch', type=int, default=1)
        self.parser.add_argument('--checkpoint', type=str, default='model_para/60.pkl')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


# parser
class test_options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='test options')
        # test related
        self.parser.add_argument('--test_interval', type=int, default=10)
        self.parser.add_argument('--batch_size', type=int, default=100)
        self.parser.add_argument('--checkpoint', type=str, default='model_para/400.pkl')
        self.parser.add_argument('--case', type=str, default='aa_resnet50')
        self.parser.add_argument('--pretrained', type=bool, default=True)
        self.parser.add_argument('--num_classes', type=int, default=10)

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt