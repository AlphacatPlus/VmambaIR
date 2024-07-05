# flake8: noqa
import os.path as osp
from VmambaIR.train_pipeline import train_pipeline

import VmambaIR.archs
import VmambaIR.data
import VmambaIR.models
import VmambaIR.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
