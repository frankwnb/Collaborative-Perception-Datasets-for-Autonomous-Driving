# Collaborative / Cooperative / Multi-agent Perception
## Overview

This repository consolidates **Collaborative Perception (CP)** datasets for autonomous driving, covering a wide range of communication paradigms, including pure **roadside perception**, **Vehicle-to-Vehicle (V2V)**, **Vehicle-to-Infrastructure (V2I)**, **Vehicle-to-Everything (V2X)**, and **Infrastructure-to-Infrastructure (I2I)** scenarios. It includes nearly all publicly available **CP** datasets and provides links to relevant publications, source code, and dataset downloads, offering researchers an efficient and centralized resource to aid their research and development in this field.

First, the repository introduces commonly used **autonomous driving simulation tools**, followed by categorizing **CP datasets** based on collaboration paradigms, presented in a tabular format. Each dataset is then described in detail, helping readers better understand the characteristics and applicable scenarios of each dataset. In addition, the repository also consolidates classic methods and cutting-edge research in **collaborative perception**, providing valuable insights into current trends and future directions in the field.


### :link:Jump to:
- ### [[Simulator]()]
- ### [[Roadside Datasets]()]
- ### [[V2V Datasets]()]
- ### [[V2I Datasets]()]
- ### [[V2X Datasets]()]
- ### [[I2I Datasets]()]
- ### [[Methods()]


## Simulator - Simulators for Collaborative Perception (CP) Research

**Simulators** play a critical role in **Collaborative Perception (CP)** research for **Autonomous Driving (AD)**, offering cost-effective, safe, and annotation-efficient alternatives to real-world data collection. They enable the generation of **accurate object attributes** and provide **automated annotations**, crucial for training and evaluating perception algorithms.

Several **open-source platforms** are widely used for data synthesis:

### **CARLA**:
- **Brief summary**: A versatile simulator that supports a wide variety of **driving scenarios** and **multimodal sensors**, including **RGB**, **depth**, **radar**, and **LiDAR**. While **CARLA** provides valuable simulation capabilities, it does have limitations in its **LiDAR simulation**, particularly with **point cloud intensity** and **echo dropout modeling**.

- **Introduction**:  
This paper introduces **CARLA**, an open-source urban driving simulator designed to advance autonomous driving research. The motivation behind CARLA is to address the limitations of physical world testing, which is costly and logistically difficult, by providing a flexible, cost-effective, and safe alternative for training and evaluating autonomous driving systems. The core contributions include the creation of a highly customizable and realistic simulation environment capable of replicating complex urban scenarios, including dynamic traffic, pedestrians, and variable weather conditions. CARLA allows researchers to test various autonomous driving approaches such as modular pipelines, imitation learning, and reinforcement learning. The simulator provides an open platform for researchers to explore these methods in a controlled setting while ensuring reproducibility and enabling comparisons across different approaches. 

The paper demonstrates CARLA’s utility through experiments evaluating the performance of autonomous systems on increasingly difficult driving tasks, revealing the challenges of generalization to new environments and conditions. This research is significant as it provides the community with an essential tool for autonomous driving research, promoting innovation in sensorimotor control and autonomous vehicle development.




  
- **SUMO**: A simulator primarily focused on **vehicle** and **pedestrian trajectory simulation**. It can be co-simulated with **CARLA** to improve **scene fidelity** and add more realism to the simulation of road environments.

- **AirSim**: A simulation platform designed for testing **autonomous vehicles** in **3D environments** with a focus on both **driving** and **flying** tasks.

The **OpenCDA** framework integrates both **CARLA** and **SUMO**, providing a **standardized scenario management** system and **benchmarks** for **perception** and **planning tasks**. This integration improves the overall effectiveness of collaborative perception research by offering a unified simulation platform.

These simulators are instrumental in generating data under various conditions, providing the necessary tools for **CP datasets** that are essential for autonomous driving research.












## Roadside Datasets
- **Roadside Datasets**: Collected from infrastructure-based sensors, these datasets focus on vehicle-infrastructure interactions, aiding research in traffic monitoring and cooperative perception in urban and highway settings.


## V2V Datasets

- **V2V Datasets**: Vehicle-to-vehicle datasets capture collaboration between vehicles, facilitating research on cooperative perception under occlusion, sparse observations, or dynamic driving scenarios.

## V2I Datasets

- **V2I Datasets**: These datasets involve communication between vehicles and infrastructure, supporting cooperative tasks like object detection, tracking, and decision-making in connected environments.


## V2X Datasets
- **V2X Datasets**: Covering vehicle-to-everything communication, these datasets integrate multiple agents such as vehicles, infrastructure, and other environmental elements like drones or pedestrians, enabling research in complex collaborative scenarios.


## I2I Datasets
- **I2I Datasets**: Focused on infrastructure-to-infrastructure collaboration, these datasets support research in scenarios with overlapping sensor coverage or distributed sensor fusion across intersections.


## Methods















## :bookmark:Dataset and Simulator

Note: {Real} denotes that the sensor data is obtained by real-world collection instead of simulation.

### Selected Preprint

- **Adver-City** (Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions) [[paper](https://arxiv.org/abs/2410.06380)] [[code](https://github.com/QUARRG/Adver-City)] [[project](https://labs.cs.queensu.ca/quarrg/datasets/adver-city)]
- **CP-GuardBench** (CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception) [[paper&review](https://openreview.net/forum?id=9MNzHTSDgh)] [~~code~~] [~~project~~]
- **Griffin** (Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark) [[paper](https://arxiv.org/abs/2503.06983)] [[code](https://github.com/wang-jh18-SVM/Griffin)] [[project](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g)]
- {Real} **InScope** (InScope: A New Real-world 3D Infrastructure-side Collaborative Perception Dataset for Open Traffic Scenarios) [[paper](https://arxiv.org/abs/2407.21581)] [[code](https://github.com/xf-zh/InScope)] [~~project~~]
- {Real} **Mixed Signals** (Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration) [[paper](https://arxiv.org/abs/2502.14156)] [[code](https://github.com/chinitaberrio/Mixed-Signals)] [[project](https://mixedsignalsdataset.cs.cornell.edu)]
- **Multi-V2X** (Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception) [[paper](https://arxiv.org/abs/2409.04980)] [[code](https://github.com/RadetzkyLi/Multi-V2X)] [~~project~~]
- **OPV2V-N** (RCDN: Towards Robust Camera-Insensitivity Collaborative Perception via Dynamic Feature-based 3D Neural Modeling) [[paper](https://arxiv.org/abs/2405.16868)] [~~code~~] [~~project~~]
- **V2V-QA** (V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models) [[paper](https://arxiv.org/abs/2502.09980)] [[code](https://github.com/eddyhkchiu/V2VLLM)] [[project](https://eddyhkchiu.github.io/v2vllm.github.io)]
- {Real} **V2XPnP-Seq** (V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction) [[paper](https://arxiv.org/abs/2412.01812)] [[code](https://github.com/Zewei-Zhou/V2XPnP)] [[project](https://mobility-lab.seas.ucla.edu/v2xpnp)]
- {Real} **V2X-Radar** (V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception) [[paper](https://arxiv.org/abs/2411.10962)] [[code](https://github.com/yanglei18/V2X-Radar)] [[project](http://openmpd.com/column/V2X-Radar)]
- {Real} **V2X-Real** (V2X-Real: a Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception) [[paper](https://arxiv.org/abs/2403.16034)] [~~code~~] [[project](https://mobility-lab.seas.ucla.edu/v2x-real)]
- {Real} **V2X-ReaLO** (V2X-ReaLO: An Open Online Framework and Dataset for Cooperative Perception in Reality) [[paper](https://arxiv.org/abs/2503.10034)] [~~code~~] [~~project~~]
- **WHALES** (WHALES: A Multi-Agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving) [[paper](https://arxiv.org/abs/2411.13340)] [[code](https://github.com/chensiweiTHU/WHALES)] [[project](https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A)]

### CVPR 2025

- **Mono3DVLT-V2X** (Mono3DVLT: Monocular-Video-Based 3D Visual Language Tracking) [~~paper~~] [~~code~~] [~~project~~]
- **RCP-Bench** (RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions) [~~paper~~] [~~code~~] [~~project~~]
- **V2X-R** (V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion) [[paper](https://arxiv.org/abs/2411.08402)] [[code](https://github.com/ylwhxht/V2X-R)] [~~project~~]

### CVPR 2024

- {Real} **HoloVIC** (HoloVIC: Large-Scale Dataset and Benchmark for Multi-Sensor Holographic Intersection and Vehicle-Infrastructure Cooperative) [[paper](https://arxiv.org/abs/2403.02640)] [~~code~~] [[project](https://holovic.net)]
- {Real} **Open Mars Dataset** (Multiagent Multitraversal Multimodal Self-Driving: Open MARS Dataset) [[code](https://github.com/ai4ce/MARS)] [[paper](https://arxiv.org/abs/2406.09383)] [[project](https://ai4ce.github.io/MARS)]
- {Real} **RCooper** (RCooper: A Real-World Large-Scale Dataset for Roadside Cooperative Perception) [[paper](https://arxiv.org/abs/2403.10145)] [[code](https://github.com/AIR-THU/DAIR-RCooper)] [[project](https://www.t3caic.com/qingzhen)]
- {Real} **TUMTraf-V2X** (TUMTraf V2X Cooperative Perception Dataset) [[paper](https://arxiv.org/abs/2403.01316)] [[code](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)] [[project](https://tum-traffic-dataset.github.io/tumtraf-v2x)]

### ECCV 2024

- {Real} **H-V2X** (H-V2X: A Large Scale Highway Dataset for BEV Perception) [[paper](https://eccv2024.ecva.net/virtual/2024/poster/126)] [~~code~~] [~~project~~]

### NeurIPS 2024

- {Real} **DAIR-V2X-Traj** (Learning Cooperative Trajectory Representations for Motion Forecasting) [[paper](https://arxiv.org/abs/2311.00371)] [[code](https://github.com/AIR-THU/V2X-Graph)] [[project](https://thudair.baai.ac.cn/index)]

### ICLR 2024

- **OPV2V-H** (An Extensible Framework for Open Heterogeneous Collaborative Perception) [[paper&review](https://openreview.net/forum?id=KkrDUGIASk)] [[code](https://github.com/yifanlu0227/HEAL)] [[project](https://huggingface.co/datasets/yifanlu/OPV2V-H)]

### AAAI 2024

- **DeepAccident** (DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving) [[paper](https://arxiv.org/abs/2304.01168)] [[code](https://github.com/tianqi-wang1996/DeepAccident)] [[project](https://deepaccident.github.io)]

### CVPR 2023

- **CoPerception-UAV+** (Collaboration Helps Camera Overtake LiDAR in 3D Detection) [[paper](https://arxiv.org/abs/2303.13560)] [[code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[project](https://siheng-chen.github.io/dataset/CoPerception+)]
- **OPV2V+** (Collaboration Helps Camera Overtake LiDAR in 3D Detection) [[paper](https://arxiv.org/abs/2303.13560)] [[code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[project](https://siheng-chen.github.io/dataset/CoPerception+)]
- {Real} **V2V4Real** (V2V4Real: A Large-Scale Real-World Dataset for Vehicle-to-Vehicle Cooperative Perception) [[paper](https://arxiv.org/abs/2303.07601)] [[code](https://github.com/ucla-mobility/V2V4Real)] [[project](https://mobility-lab.seas.ucla.edu/v2v4real)]
- {Real} **DAIR-V2X-Seq** (V2X-Seq: The Large-Scale Sequential Dataset for the Vehicle-Infrastructure Cooperative Perception and Forecasting) [[paper](https://arxiv.org/abs/2305.05938)] [[code](https://github.com/AIR-THU/DAIR-V2X-Seq)] [[project](https://thudair.baai.ac.cn/index)]

### NeurIPS 2023

- **IRV2V** (Robust Asynchronous Collaborative 3D Detection via Bird's Eye View Flow) [[paper&review](https://openreview.net/forum?id=UHIDdtxmVS)] [~~code~~] [~~project~~]

### ICCV 2023

- **Roadside-Opt** (Optimizing the Placement of Roadside LiDARs for Autonomous Driving) [[paper](https://arxiv.org/abs/2310.07247)] [~~code~~] [~~project~~]

### ICRA 2023

- {Real} **DAIR-V2X-C Complemented** (Robust Collaborative 3D Object Detection in Presence of Pose Errors) [[paper](https://arxiv.org/abs/2211.07214)] [[code](https://github.com/yifanlu0227/CoAlign)] [[project](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented)]
- **RLS** (Analyzing Infrastructure LiDAR Placement with Realistic LiDAR Simulation Library) [[paper](https://arxiv.org/abs/2211.15975)] [[code](https://github.com/PJLab-ADG/LiDARSimLib-and-Placement-Evaluation)] [~~project~~]
- **V2XP-ASG** (V2XP-ASG: Generating Adversarial Scenes for Vehicle-to-Everything Perception) [[paper](https://arxiv.org/abs/2209.13679)] [[code](https://github.com/XHwind/V2XP-ASG)] [~~project~~]

### CVPR 2022

- **AutoCastSim** (COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles) [[paper](https://arxiv.org/abs/2205.02222)] [[code](https://github.com/hangqiu/AutoCastSim)] [[project](https://utexas.app.box.com/v/coopernaut-dataset)]
- {Real} **DAIR-V2X** (DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection) [[paper](https://arxiv.org/abs/2204.05575)] [[code](https://github.com/AIR-THU/DAIR-V2X)] [[project](https://thudair.baai.ac.cn/index)]

### NeurIPS 2022

- **CoPerception-UAV** (Where2comm: Efficient Collaborative Perception via Spatial Confidence Maps) [[paper&review](https://openreview.net/forum?id=dLL4KXzKUpS)] [[code](https://github.com/MediaBrain-SJTU/where2comm)] [[project](https://siheng-chen.github.io/dataset/coperception-uav)]

### ECCV 2022

- **V2XSet** (V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer) [[paper](https://arxiv.org/abs/2203.10638)] [[code](https://github.com/DerrickXuNu/v2x-vit)] [[project](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6)]

### ICRA 2022

- **OPV2V** (OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication) [[paper](https://arxiv.org/abs/2109.07644)] [[code](https://github.com/DerrickXuNu/OpenCOOD)] [[project](https://mobility-lab.seas.ucla.edu/opv2v)]

### ACCV 2022

- **DOLPHINS** (DOLPHINS: Dataset for Collaborative Perception Enabled Harmonious and Interconnected Self-Driving) [[paper](https://arxiv.org/abs/2207.07609)] [[code](https://github.com/explosion5/Dolphins)] [[project](https://dolphins-dataset.net)]

### ICCV 2021

- **V2X-Sim** (V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving) [[paper](https://arxiv.org/abs/2202.08449)] [[code](https://github.com/ai4ce/V2X-Sim)] [[project](https://ai4ce.github.io/V2X-Sim)]

### CoRL 2017

- **CARLA** (CARLA: An Open Urban Driving Simulator) [[paper](https://arxiv.org/abs/1711.03938)] [[code](https://github.com/carla-simulator/carla)] [[project](https://carla.org)]
