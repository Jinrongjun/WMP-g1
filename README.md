<h1>WMP</h1>

Code for the paper: 
### World Model-based Perception for Visual Legged Locomotion
[Hang Lai](https://apex.sjtu.edu.cn/members/laihang@apexlab.org), [Jiahang Cao](https://apex.sjtu.edu.cn/members/jhcao@apexlab.org), [JiaFeng Xu](https://scholar.google.com/citations?user=GPmUxtIAAAAJ&hl=zh-CN&oi=ao), [Hongtao Wu](https://scholar.google.com/citations?user=7u0TYgIAAAAJ&hl=zh-CN&oi=ao), [Yunfeng Lin](https://apex.sjtu.edu.cn/members/yflin@apexlab.org), [Tao Kong](https://www.taokong.org/), [Yong Yu](https://scholar.google.com.hk/citations?user=-84M1m0AAAAJ&hl=zh-CN&oi=ao), [Weinan Zhang](https://wnzhang.net/) 

### [🌐 Project Website](https://wmp-loco.github.io/) | [📄 Paper](https://arxiv.org/abs/2409.16784)


## UPDATE 7.23
添加了双修正机制（h_t 和 a_t），实现DK_ymloss，阶段完成性版本。

## UPDATE 7.17
修复runner_DK的索引问题；修改history encoder；g1_config下网络大小对应调整；wm的prop修改成12维（3*base lin vel + 3*base ang vel + 3*command + 3*projected gravity）

## UPDATE 7.16
添加了g1_DK任务，把DK模块接到actor前面做当前状态压缩。

## UPDATE 7.15
添加了27 / 29 自由度下的AMP功能，并且修复了runner里面维度索引异常的问题。现在可以同时兼容27/29 dof下的模型训练。
   
## Requirements
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch:
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
4. Install other packages:
    - `sudo apt-get install build-essential --fix-missing`
    - `sudo apt-get install ninja-build`
    - `pip install setuptools==59.5.0`
    - `pip install ruamel_yaml==0.17.4`
    - `sudo apt install libgl1-mesa-glx -y`
    - `pip install opencv-contrib-python`
    - `pip install -r requirements.txt`

## Training
```
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```
Training takes about 23G GPU memory, and at least 10k iterations recommended.

## Visualization
**Please make sure you have trained the WMP before**
```
python legged_gym/scripts/play.py --task=a1_amp --sim_device=cuda:0 --terrain=climb
```


## Acknowledgments

We thank the authors of the following projects for making their code open source:

- [leggedgym](https://github.com/leggedrobotics/legged_gym)
- [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [parkour](https://github.com/ZiwenZhuang/parkour/tree/main)
- [extreme-parkour](https://github.com/chengxuxin/extreme-parkour)



## Citation

If you find this project helpful, please consider citing our paper:
```
@article{lai2024world,
  title={World Model-based Perception for Visual Legged Locomotion},
  author={Lai, Hang and Cao, Jiahang and Xu, Jiafeng and Wu, Hongtao and Lin, Yunfeng and Kong, Tao and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2409.16784},
  year={2024}
}
```
