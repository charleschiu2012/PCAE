import torch
from ..config import config


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_inputs_img = None
        self.next_inputs_pc = None
        self.next_targets = None

        self.preload()

    def preload(self):
        try:
            if config.network.mode_flag == 'ae':
                self.next_inputs_pc, self.next_targets = next(self.loader)
            elif config.network.mode_flag == 'lm':
                self.next_inputs_img, self.next_inputs_pc, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_inputs_img = None
            self.next_inputs_pc = None
            self.next_targets = None
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            return

        with torch.cuda.stream(self.stream):
            if config.network.mode_flag == 'ae':
                self.next_inputs_pc = self.next_inputs_pc.cuda(non_blocking=True).float()
                self.next_targets = self.next_targets.cuda(non_blocking=True).float()
                self.next_inputs_pc = self.next_inputs_pc.permute(0, 2, 1).contiguous()  # (B, 3, N) to feet the network

            elif config.network.mode_flag == 'lm':
                self.next_inputs_img = self.next_inputs_img.cuda(non_blocking=True).float()
                self.next_inputs_pc = self.next_inputs_pc.cuda(non_blocking=True).float()
                self.next_targets = self.next_targets.cuda(non_blocking=True).float()
                self.next_inputs_img = self.next_inputs_img.permute(0, 3, 2, 1).contiguous()  # (B, C, W, H)
                self.next_inputs_pc = self.next_inputs_pc.permute(0, 2, 1).contiguous()  # (B, 3, N) to feet the network

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if config.network.mode_flag == 'ae':
            inputs_pc = self.next_inputs_pc
            targets = self.next_targets
            self.preload()
            return inputs_pc, targets

        elif config.network.mode_flag == 'lm':
            inputs_img = self.next_inputs_img
            inputs_pc = self.next_inputs_pc
            targets = self.next_targets
            self.preload()
            return inputs_img, inputs_pc, targets
