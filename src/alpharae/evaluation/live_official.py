"""Official paired JPEG metrics during distributed checkpoint validation."""

import os
from pathlib import Path
import sys
import tempfile

import torch

from alpharae.evaluation.alphavae_protocol import BACKGROUNDS, OfficialMetrics, save_composites


class LiveOfficialMetrics:
    def __init__(self, device):
        from alpharae.runtime import ROOT
        root=ROOT
        self.metric=OfficialMetrics(device,root/"weights/torch",names=("psnr","ssim","lpips"))
        self.directory=tempfile.TemporaryDirectory(prefix=f"rgba_metrics_{os.getpid()}_")
        self.root=Path(self.directory.name)

    @torch.no_grad()
    def __call__(self,pred,target):
        totals={key:0. for key in ("psnr","ssim","lpips")}
        names=list(BACKGROUNDS)
        for index in range(len(pred)):
            save_composites(pred[index],self.root/"prediction","sample")
            save_composites(target[index],self.root/"target","sample")
            for start in range(0,len(names),3):
                colors=names[start:start+3]
                scores=self.metric.paired([self.root/"prediction"/name/"sample.jpg" for name in colors],
                                          [self.root/"target"/name/"sample.jpg" for name in colors])
                for key in totals:totals[key] += sum(scores[key])
        return {f"official_{key}":pred.new_tensor(value/(len(pred)*9)) for key,value in totals.items()}
