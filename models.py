import torch
from torch import nn
from resnet import resnet50, resnet101, resnet152
from aa_resnet import aa_resnet50, aa_resnet101, aa_resnet152
from se_resnet import se_resnet50, se_resnet101, se_resnet152

class train_model(nn.Module):
    def __init__(self, opts, device):
        super(train_model, self).__init__()
        self.opts = opts
        self.device = device
        # model
        self.network = self.model_choice(self.opts.case)
        # optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.opts.lr)
        # loss function
        self.loss_func = torch.nn.CrossEntropyLoss().to(device)

    def model_choice(self, case):
        case = case.lower()
        # resnet
        if case == 'resnet50':
            self.model = resnet50(pretrained=self.opts.pretrained,
                                  num_classes=self.opts.num_classes,
                                  model_path=self.opts.checkpoint).to(self.device)
        if case == 'resnet101':
            self.model = resnet101(pretrained=self.opts.pretrained,
                                   num_classes=self.opts.num_classes,
                                   model_path=self.opts.checkpoint).to(self.device)
        if case == 'resnet152':
            self.model = resnet152(pretrained=self.opts.pretrained,
                                   num_classes=self.opts.num_classes,
                                   model_path=self.opts.checkpoint).to(self.device)

        # aa_resnet
        if case == 'aa_resnet50':
            self.model = aa_resnet50(pretrained=self.opts.pretrained,
                                     num_classes=self.opts.num_classes,
                                     model_path=self.opts.checkpoint).to(self.device)
        if case == 'aa_resnet101':
            self.model = aa_resnet101(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.opts.checkpoint).to(self.device)
        if case == 'aa_resnet152':
            self.model = aa_resnet152(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.opts.checkpoint).to(self.device)

        # se_resnet
        if case == 'se_resnet50':
            self.model = se_resnet50(pretrained=self.opts.pretrained,
                                     num_classes=self.opts.num_classes,
                                     model_path=self.opts.checkpoint).to(self.device)
        if case == 'se_resnet101':
            self.model = se_resnet101(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.opts.checkpoint).to(self.device)
        if case == 'se_resnet152':
            self.model = se_resnet152(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.opts.checkpoint).to(self.device)
        return self.model


class test_model(nn.Module):
    def __init__(self, opts, device, model_path):
        super(test_model, self).__init__()
        self.opts = opts
        self.device = device
        self.model_path = model_path
        # model
        self.network = self.model_choice(self.opts.case)
        # loss function
        self.loss_func = torch.nn.CrossEntropyLoss().to(device)

    def model_choice(self, case):
        case = case.lower()
        # resnet
        if case == 'resnet50':
            self.model = resnet50(pretrained=self.opts.pretrained,
                                  num_classes=self.opts.num_classes,
                                  model_path=self.model_path).to(self.device)
        if case == 'resnet101':
            self.model = resnet101(pretrained=self.opts.pretrained,
                                   num_classes=self.opts.num_classes,
                                   model_path=self.model_path).to(self.device)
        if case == 'resnet152':
            self.model = resnet152(pretrained=self.opts.pretrained,
                                   num_classes=self.opts.num_classes,
                                   model_path=self.model_path).to(self.device)

        # aa_resnet
        if case == 'aa_resnet50':
            self.model = aa_resnet50(pretrained=self.opts.pretrained,
                                     num_classes=self.opts.num_classes,
                                     model_path=self.model_path).to(self.device)
        if case == 'aa_resnet101':
            self.model = aa_resnet101(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.model_path).to(self.device)
        if case == 'aa_resnet152':
            self.model = aa_resnet152(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.model_path).to(self.device)

        # se_resnet
        if case == 'se_resnet50':
            self.model = se_resnet50(pretrained=self.opts.pretrained,
                                     num_classes=self.opts.num_classes,
                                     model_path=self.model_path).to(self.device)
        if case == 'se_resnet101':
            self.model = se_resnet101(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.model_path).to(self.device)
        if case == 'se_resnet152':
            self.model = se_resnet152(pretrained=self.opts.pretrained,
                                      num_classes=self.opts.num_classes,
                                      model_path=self.model_path).to(self.device)
        return self.model