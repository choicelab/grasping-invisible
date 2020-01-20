# A Deep Learning Approach to Grasping the Invisible

This repository is for the paper

**[A Deep Learning Approach to Grasping the Invisible][1]**<br/>
*Yang, Yang and Liang, Hengyue and Choi, Changhyun*<br/>
[arxiv.org/abs/1909.04840][2]<br/>

If you find this code useful, please consider citing our work:

```
@inproceedings{yang2019deep,
  title={A Deep Learning Approach to Grasping the Invisible},
  author={Yang, Yang and Liang, Hengyue and Choi, Changhyun},
  journal={arXiv preprint arXiv:1909.04840},
  year={2019}
}
```

## Dependencies
```
- Ubuntu 16.04
- Python 3
- PyTorch 0.4
```
The file of the conda environment is environment.yml. We use [V-REP 3.5.0][5] as the simulation environment.

## Code
We do experiments on a NVIDIA 1080 Ti GPU. It requires at least 8GB of GPU memory to run the code.

First run V-REP and open the file ```simulation/simulation.ttt``` to start the simulation. Then download the pre-trained models by running

```
sh downloads.sh
```

### Training
To train from scratch, run

```
python main.py
```

You can also resume training from checkpoint and collected data
```
python main.py
--load_ckpt --critic_ckpt CRITIC-MODEL-PATH --coordinator_ckpt COORDINATOR-MODEL-PATH
--continue_logging --logging_directory SESSION-DIRECTORY
```

### Testing
```
python main.py
--is_testing --test_preset_cases --test_target_seeking
--load_ckpt --critic_ckpt CRITIC-MODEL-PATH --coordinator_ckpt COORDINATOR-MODEL-PATH
--config_file TEST-CASE-PATH
```
The files of the test cases are available in ```simulation/preset```.

## Acknowledgments
We use the following code in our project

- [Visual Pushing and Grasping Toolbox][3]

- [Light-Weight RefineNet for Real-Time Semantic Segmentation][4]

[1]: https://sites.google.com/umn.edu/grasping-invisible
[2]: https://arxiv.org/abs/1909.04840
[3]: https://github.com/andyzeng/visual-pushing-grasping
[4]: https://github.com/DrSleep/light-weight-refinenet
[5]: http://coppeliarobotics.com/previousVersions
