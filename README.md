# Self Reward Design with Fine-grained Interpretability

This folder contains the codes for [Self Reward Design with Fine-grained Interpretability](https://arxiv.org/abs/2112.15034).

In this project, we attempt to solve reinforcement learning problem using artificial neural network (NN) attained to achieve interpretability in an extreme way. Each neuron in the NN is defined with purposeful human design.



## Version 2
All commands used to execute our experiments can be found in misc/commands.txt and misc/commands_mujoco.txt. The full results can be found in our <a href='https://drive.google.com/drive/folders/1FoeGgfcO4hdWZynxVFrzPYWvYwIWVZ0p?usp=share_link'>google drive</a>.

### Installation
We use conda environment, env.yml is available.
Some manual installation is still necessary. We use pytorch 1.12.1. Please perform the necessary installation that depends on your machine (refer to pytorch's main website) 


### Fish sale auction
In this scenario, traditional RL is not the suitable choice since interpretability is crucial. Fig. (C) is our main result, while fig. (D) shows the result where the lack of interpretability results in sabotaged result.

<img src="https://drive.google.com/uc?export=view&id=18woxwEba2NcuGddMsPLd-UxpbIdWHFZC" width="540"></img>

### MuJoCo and Half Cheetah
In this scenario, we use SRD to control half cheetah motion. 

<img src="https://drive.google.com/uc?export=view&id=1Q57N4Bw-LAacuaddAeTIKGh77dnhcDXr" width="540"></img>

In the following example, we show movement with inhibitor=2, i.e. we allow the user to give "stop" instruction.

<img src="https://drive.google.com/uc?export=view&id=1o5Omkic3IaZt8YYkRe5Sl_Gdg_ON5xvq" width="360"></img>

## Version 1

All codes for version 1 has been moved into legacy/v1.

Quick start: refer to the `_quick_start` folder.<br>
Existing results can be found in google drive <a href="https://drive.google.com/drive/folders/1FoeGgfcO4hdWZynxVFrzPYWvYwIWVZ0p?usp=sharing">link</a>.


## Toy Fish
A simple toy world where a fish either moves or eats food, while trying to stay alive.

<img src="https://drive.google.com/uc?export=view&id=1-qvG1E_AThX0-XOsw-zJvl9bAvfxKLcD" width="480"></img>

## Robot2D in Lava land
A grid world where robot tries to reach the target tile (yellow).<br>
The project features *uncertainty avoidance*, where robot tries to avoid lava tiles at all cost.<br>
<img src="https://drive.google.com/uc?export=view&id=101T_MzHh70T7y55TJdBPpvXEo8yWgntT" width="360"></img>
<img src="https://drive.google.com/uc?export=view&id=1C1pK4bOtnaBagbJc9nBfSXV2BI-P7n8g" width="360"></img><br>
