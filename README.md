# SVJTaggerNN

## Setup

```
cd <working_directory>
git clone git@github.com:cms-svj/SVJTaggerNN.git
cd SVJTaggerNN
./setup.sh -l
```

#Example
```
srun --account cms_svj --pty --constraint v100 --nodes=1 --partition gpu_gce --gres=gpu:1 bash
source initLCG.sh
python train.py --outf logTestEL -C configs/C1_EventLevel.py
python validation.py --model net.pth --outf logTestEL -C logTestEL/config_out.py
```
