# Collaborative / Cooperative / Perception Datasets
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
- ### [[Methods]()]


## Simulator - Simulators for Collaborative Perception (CP) Research

**Simulators** play a critical role in **Collaborative Perception (CP)** research for **Autonomous Driving (AD)**, offering cost-effective, safe, and annotation-efficient alternatives to real-world data collection. They enable the generation of **accurate object attributes** and provide **automated annotations**, crucial for training and evaluating perception algorithms.

Several **open-source platforms** are widely used for data synthesis:

| **Simulator** | **Year** | **Venue** |
|---------------|----------|-----------|
| CARLA         | 2017     | Conference on Robot Learning (CoRL) |
| SUMO          | 2002     | Proceedings of the 4th Middle East Symposium on Simulation and Modelling (MESM2002) |
| AirSim        | 2018     | Computer Vision and Pattern Recognition (CVPR) |
| OpenCDA       | 2021     | IEEE International Conference on Intelligent Transportation Systems (ITSC) |


### CARLA: An Open Urban Driving Simulator [[paper](https://arxiv.org/abs/1711.03938)] [[code](https://github.com/carla-simulator/carla)] [[project](https://carla.org)]

- **Background and Motivation**  
The paper introduces CARLA, an open-source simulator specifically designed for autonomous driving research. Traditional testing for autonomous vehicles in urban environments is costly and logistically challenging. Simulations offer an affordable alternative, enabling testing of various autonomous driving models, including perception, control, and navigation in complex urban settings. CARLA was developed to address the challenges in simulating real-world scenarios such as dynamic traffic, pedestrians, weather conditions, and other urban obstacles, making it essential for advancing autonomous driving research.

- **Key Contributions**  
  - **Open-Source Platform**: CARLA is an open-source urban driving simulator, offering free access to both the code and digital assets.
  - **Flexible Sensor and Environmental Setup**: It allows for customizable sensor suites (e.g., RGB cameras, LiDAR) and a variety of environmental conditions (weather, time of day).
  - **Realistic Urban Environment**: CARLA provides a highly detailed and dynamic urban environment with realistic traffic, pedestrians, and infrastructure.
  - **Simulation of Three Approaches**: The paper tests and compares three approaches to autonomous driving: a modular pipeline, imitation learning, and reinforcement learning, providing valuable insights into their performance in controlled urban scenarios.
  - **Evaluation Metrics**: The simulation framework provides comprehensive metrics for performance evaluation, facilitating in-depth analysis of driving policies.


### SUMO: Simulation of Urban MObility [[paper](https://elib.dlr.de/6661/2/dkrajzew_MESM2002.pdf)] [[code]()] [[project](https://elib.dlr.de/6661/)]

- **Background and Motivation**
Traffic simulation plays a crucial role in understanding and improving urban mobility, given the complexity and unpredictability of real-world traffic. Existing traffic models often fail to account for the variability introduced by individual driver behavior, diverse transportation modes, and changing traffic conditions. To address these challenges, open-source simulation tools are essential to enable researchers to model and analyze traffic behavior more effectively.

- **Key Contributions**
  - **Open-Source Traffic Simulation**: SUMO is an open-source, microscopic, and multi-modal traffic simulation platform that enables detailed simulation of urban mobility, including car movements, public transportation, and pedestrian traffic.
  - **Comprehensive Traffic Flow Modeling**: The platform uses continuous modeling to represent vehicle movements and interactions, incorporating factors like driver behavior and road network conditions.
  - **Extensibility and Flexibility**: SUMO offers a customizable framework where researchers can integrate their own models and traffic scenarios, allowing for an adaptable and scalable research tool.
  - **Support for Various Modalities**: SUMO supports multi-modal simulations, including cars, buses, trams, and pedestrian pathways, to simulate complex urban environments.
  - **Simulation of Large-Scale Networks**: SUMO is capable of simulating large urban areas and can handle high volumes of traffic data, making it suitable for analyzing traffic management strategies in large cities. 


### AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Testing autonomous vehicles in real-world environments is both costly and time-consuming. Additionally, collecting the large datasets required for training machine learning algorithms can be challenging and impractical. AirSim, built on Unreal Engine, aims to address these issues by offering a high-fidelity simulation platform that supports real-time, hardware-in-the-loop simulations. It is designed to provide realistic physical and visual simulations to aid the development of autonomous vehicles, enabling safe and cost-effective testing in various environments.

- **Key Contributions**  
  - **High-Fidelity Simulation**: AirSim offers a realistic platform for simulating both the visual and physical environments of autonomous vehicles.
  - **Extensibility**: The platform is highly modular and extensible, supporting a variety of vehicles, sensors, and hardware platforms.
  - **Realistic Physics and Sensor Models**: AirSim simulates complex environmental factors such as gravity, air pressure, and magnetic fields, along with advanced sensor models for IMU, GPS, and barometers.
  - **Integration with Unreal Engine**: By leveraging Unreal Engine, AirSim supports realistic rendering, including photorealistic graphics, object segmentation, and depth sensing.
  - **Vehicle Model and Simulation**: AirSim includes a vehicle model capable of simulating a range of vehicle types, from ground vehicles to aerial drones, with detailed dynamics and control mechanisms.


### OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-Simulation  [[code](https://github.com/ucla-mobility/OpenCDA)] [[doc](https://opencda-documentation.readthedocs.io/en/latest)]

- **Background and Motivation**  
Cooperative Driving Automation (CDA) is gaining attention but faces significant challenges, particularly the lack of simulation platforms that support multi-vehicle cooperation. Current simulators primarily focus on single-vehicle automation, hindering the evaluation and comparison of CDA algorithms in a collaborative setting. OpenCDA was developed to bridge this gap, providing a flexible and modular tool for testing CDA algorithms in both traffic-level and individual vehicle scenarios.

- **Key Contributions**  
  - **Co-Simulation Platform**: OpenCDA integrates various simulators (e.g., CARLA, SUMO) to support both vehicle and traffic-level simulations for cooperative driving tasks.
  - **Modular Design**: The framework is highly modular, allowing users to replace default algorithms with their own customized designs for different CDA applications.
  - **Full-Stack CDA System**: OpenCDA provides a complete system including sensing, computation, and actuation modules, along with cooperative features like vehicle communication and information sharing.
  - **Benchmarking and Testing**: It includes a benchmark testing database, offering standard scenarios and evaluation metrics for comparing CDA algorithms.
  - **Platooning Example**: The paper demonstrates the capabilities of OpenCDA through a platooning implementation, showcasing its flexibility and effectiveness in CDA research.



   [[paper]()] [[code]()] [[project]()]
   [[paper]()] [~~code~~] [~~project~~]

   
## Roadside Datasets

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| Ko-PER | 2014 | ITSC | C, L | Real | 3DOD, MOT | [download](https://www.uni-ulm.de/in/mrm/forschung/datensaetze.html) |
| CityFlow | 2019 | CVPR | C | Real | MTSCT/MTMCT, ReID | [download](https://cityflow-project.github.io/) |
| INTERACTION | 2019 | IROS | C, L | Real | 2DOD, TP | [download](https://interaction-dataset.com/) |
| A9-Dataset | 2022 | IV | C, L | Real | 3DOD | [download](https://a9-dataset.com/) |
| IPS300+ | 2022 | ICRA | C, L | Real | 2DOD, 3DOD | [download](http://www.openmpd.com/column/IPS300) |
| Rope3D | 2022 | CVPR | C, L | Real | 2DOD, 3DOD | [download](https://thudair.baai.ac.cn/rope) |
| LUMPI | 2022 | IV | C, L | Real | 3DOD | [download](https://data.uni-hannover.de/cs_CZ/dataset/lumpi) |
| TUMTraf-I | 2023 | ITSC | C, L | Real | 3DOD | [download](https://innovation-mobility.com/en/project-providentia/a9-dataset/) |
| RoScenes | 2024 | ECCV | C | Real | 3DOD | [download](https://roscenes.github.io./) |
| H-V2X | 2024 | ECCV | C, R | Real | BEV Det, MOT, TP | [download](https://pan.quark.cn/s/86d19da10d18) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.

Note: {Real} denotes that the sensor data is obtained by real-world collection instead of simulation.


### The Ko-PER Intersection Laserscanner and Video Dataset [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Intersections are critical areas for traffic safety, as they are often the sites of accidents. Traditional methods of assessing and improving intersection safety are limited by real-world testing challenges. The Ko-PER project addresses this by equipping a public intersection with laserscanners and video cameras, enabling comprehensive data collection for traffic perception tasks. The primary motivation is to develop better models for road user detection, classification, and tracking, thus improving intersection safety.

- **Key Contributions**  
  - **Dataset for Multi-Object Detection and Tracking**: The dataset includes data from 14 laserscanners and 8 video cameras, designed to improve object detection and classification at intersections.
  - **Reference Data for Evaluation**: It provides highly accurate reference data for vehicle positions using RTK-GPS, offering a benchmark for evaluating perception algorithms.
  - **Rich Sensor Data**: The dataset features synchronized laserscanner measurements and camera images, facilitating research in multi-object tracking and classification.
  - **Real-World Intersection Setup**: Data was collected from a complex intersection, providing a naturalistic environment for testing and validating algorithms for intersection collision avoidance systems.
  - **Public Availability**: The dataset is publicly available for use by the research community, promoting further advancements in cooperative perception and road user detection technologies.
 
### CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The CityFlow dataset addresses the challenges of tracking vehicles across large urban areas using multiple traffic cameras. Traditional tracking methods face limitations due to the small spatial coverage and limited camera setups of existing benchmarks. CityFlow aims to solve this by providing a large-scale, multi-camera dataset designed for multi-target multi-camera (MTMC) tracking and vehicle re-identification (ReID).

- **Key Contributions**  
  - **City-Scale Dataset**: CityFlow is the first dataset at a city scale with 40 cameras spanning 10 intersections, providing synchronized HD videos over a 2.5 km area.
  - **Comprehensive Annotations**: Over 200K bounding boxes across various urban scenes, with camera calibration and GPS data for precise spatio-temporal analysis.
  - **Support for MTMC Tracking and ReID**: CityFlow supports both MTMC tracking and image-based vehicle ReID, providing a benchmark for these tasks.
  - **Real-World Challenges**: The dataset covers diverse environments and traffic conditions, incorporating issues such as motion blur and overlapping camera views.
  - **Evaluation Server**: An online platform for continuous performance comparison, providing a fair and transparent benchmarking process.

### INTERACTION: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces the INTERACTION dataset, designed to support the development of autonomous driving systems in complex, interactive scenarios. Existing datasets have limitations in terms of diversity, criticality, and the inclusion of driving behaviors from different cultures. This dataset addresses these gaps by providing a variety of challenging, real-world driving scenarios, capturing interactions among vehicles with different behaviors.

- **Key Contributions**  
  - **Diverse Driving Scenarios**: The dataset includes roundabouts, intersections, ramp merging, lane changes, and other highly interactive driving scenarios from multiple countries.
  - **International Scope**: Data is collected from different continents (USA, China, Germany, Bulgaria), providing insights into various driving cultures.
  - **Complex and Critical Situations**: It includes aggressive, irrational behaviors, near-collisions, and critical driving situations, which are crucial for developing robust autonomous driving systems.
  - **Semantic Maps**: The dataset comes with detailed HD maps, which include traffic rules, lanelets, and road features, essential for motion prediction and planning tasks.
  - **Complete Interaction Data**: The dataset captures the interactions of all entities influencing vehicle behavior, enabling better modeling and prediction.

### A9-Dataset: Multi-Sensor Infrastructure-Based Dataset for Mobility Research [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces the A9-Dataset, collected from the Providentia++ test field in Germany, which aims to provide high-quality, real-world data for autonomous driving and mobility research. The dataset addresses the challenge of lacking diverse road scenarios captured by stationary multi-modal sensors, especially in infrastructure-based perception systems.

- **Key Contributions**  
  - **Multi-Sensor Setup**: The dataset includes data from cameras and LiDAR sensors mounted on overhead gantry bridges along the A9 autobahn.
  - **Diverse Traffic Scenarios**: It captures traffic on a variety of road segments including highways, rural roads, and urban intersections.
  - **High-Quality Labeled Data**: The dataset provides 3D bounding boxes for over 14,000 objects, with high-resolution camera and LiDAR frames.
  - **Real-World Traffic**: The A9-Dataset features dense traffic data recorded under real-world conditions, making it useful for training and testing perception models for autonomous vehicles.
  - **Open Access**: The dataset is publicly available, promoting further research in autonomous driving and infrastructure-based perception systems.

### IPS300+: A Challenging Multi-Modal Data Sets for Intersection Perception System [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the complexity and occlusion issues at urban intersections, which pose a significant challenge for autonomous driving systems. While on-board perception is limited in crowded, obstructed urban environments, the introduction of Cooperative Vehicle Infrastructure Systems (CVIS) offers a solution. However, there was a lack of open-source, multi-modal datasets for intersection perception, motivating the creation of the IPS300+ dataset.

- **Key Contributions**  
  - **First Open-Source Multi-Modal Dataset**: IPS300+ is the first multi-modal dataset for roadside perception in large-scale urban intersection scenes, providing data on both point clouds and images.
  - **High Label Density**: The dataset includes 14,198 frames, each with an average of 319.84 labels, significantly larger than existing datasets such as KITTI.
  - **Dense 3D Bounding Box Annotations**: Labels are provided at 5Hz, offering rich ground truth data for 3D object detection and tracking tasks.
  - **Feasible Solution for IPS Construction**: The dataset also presents an affordable approach to building Intersection Perception Systems (IPS), including a wireless solution for time synchronization and spatial calibration.
  - **Challenges in CVIS Algorithms**: The dataset introduces unique challenges for algorithms, particularly in multi-modal fusion and spatial coordination across different roadside units (RSUs), contributing valuable research avenues for CVIS and intersection perception tasks.


### Rope3D: The Roadside Perception Dataset for Autonomous Driving [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Current autonomous driving perception systems mainly focus on frontal-view data from vehicle-mounted sensors. However, this perspective creates limitations, such as blind spots and occlusions. Roadside perception systems, which provide a more comprehensive view of traffic scenarios, could enhance safety and prediction accuracy. The motivation behind this work is to address these challenges by introducing a roadside perception dataset, Rope3D, designed to improve 3D localization and object detection tasks for autonomous driving.

- **Key Contributions**  
  - **Introduction of Rope3D**: This paper presents Rope3D, the first large-scale, high-diversity roadside perception dataset for autonomous driving, containing over 50k images and 1.5M 3D objects.  
  - **Challenging Environment**: The dataset is collected under varied weather conditions, times of day, and camera specifications, introducing complexities such as ambiguous camera positions and viewpoints.  
  - **Joint Annotation**: The dataset features joint 2D-3D annotations, improving the ability to perform monocular 3D detection in roadside scenarios.  
  - **New Evaluation Metrics**: Rope3D establishes a new benchmark for 3D roadside perception, proposing unique evaluation metrics and a devkit to measure task performance.  
  - **Adapting Existing Detection Models**: The paper customizes monocular 3D detection methods for roadside perception, overcoming challenges like varied camera viewpoints and increasing object density.


### LUMPI: The Leibniz University Multi-Perspective Intersection Dataset [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces the LUMPI dataset, designed to address the limitations of single-view datasets used for autonomous driving. Traditional datasets often suffer from occlusions, making precise pose estimation difficult. LUMPI provides a multi-view dataset to improve accuracy in object detection and tracking by using multiple cameras and LiDAR sensors.

- **Key Contributions**  
  - **Multi-View Dataset**: LUMPI introduces a multi-perspective dataset combining 2D video and 3D LiDAR point clouds from multiple cameras and sensors, enhancing pose estimation and tracking accuracy.
  - **Varied Weather Conditions**: The dataset was recorded under different weather conditions, providing diverse data for more robust perception systems.
  - **Collaborative Data Processing**: It supports the development of collaborative algorithms by providing multi-sensor data that can be used to validate and compare single sensor results.
  - **Traffic Participant Labels**: Precise labels for road users are included, along with a high-density reference point cloud, aiding in accurate trajectory generation and collaboration in data processing.
  - **Use Cases**: The dataset is valuable for research in traffic forecasting, anomaly detection, intent prediction, and junction mapping.


### TUMTraf Intersection Dataset: All You Need for Urban 3D Camera-LiDAR Roadside Perception [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The TUMTraf Intersection Dataset addresses the need for high-quality 3D object detection in roadside infrastructure systems. Traditional vehicle-mounted sensors fail to cover complex intersection scenarios. This dataset was developed to enhance autonomous vehicle systems by providing labeled LiDAR point clouds and synchronized camera images from elevated roadside sensors, supporting 3D object detection tasks.

- **Key Contributions**  
  - **Comprehensive Dataset**: 4.8k labeled LiDAR point clouds and 4.8k synchronized camera images with 57.4k high-quality 3D bounding boxes.
  - **Diverse Traffic Scenarios**: The dataset includes complex maneuvers like left and right turns, overtaking, and U-turns, across varied weather and lighting conditions.
  - **Calibration Data**: Provides extrinsic calibration data for accurate sensor fusion between cameras and LiDARs.
  - **High-Class Diversity**: Includes ten object classes with a broad range of road users, including vulnerable pedestrians.
  - **Evaluation Baselines**: Offers multiple baselines for 3D object detection with monocular, LiDAR, and multi-modal setups, demonstrating robust performance in urban traffic perception tasks.

### RoScenes: A Large-scale Multi-view 3D Dataset for Roadside Perception [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces RoScenes, the largest multi-view 3D dataset designed for roadside perception tasks. Traditional roadside perception datasets face limitations due to small-scale sensing areas and insufficient camera setups. RoScenes aims to address these challenges by offering a large-scale dataset with full scene coverage and crowded traffic, designed for 3D object detection and other advanced roadside perception tasks.

- **Key Contributions**  
  - **Large-Scale Multi-View Dataset**: RoScenes provides 21.13 million 3D annotations across 64,000 m² of highway scenes, making it the largest roadside perception dataset.
  - **Innovative Annotation Pipeline**: The paper introduces a novel BEV-to-3D joint annotation pipeline that efficiently produces accurate 3D annotations without the need for expensive LiDAR sensors.
  - **Roadside Configuration**: The dataset features high-coverage camera setups, with 6-12 cameras mounted on 4-6 poles, which effectively eliminate occlusions and provide a broad perception range.
  - **RoBEV Model**: The paper presents RoBEV, a BEV detection method that uses feature-guided 3D position embedding to improve performance for 3D detection, outperforming state-of-the-art methods.
  - **Benchmark for BEV Architectures**: RoScenes serves as a benchmark for evaluating BEV architectures, offering a comprehensive study of various detection methods under real-world roadside conditions.


### H-V2X: A Large Scale Highway Dataset for BEV Perception  [[paper](https://eccv2024.ecva.net/virtual/2024/poster/126)] [~~code~~] [~~project~~]

- **Background and Motivation**  
The paper introduces H-V2X, a large-scale dataset for highway roadside perception, addressing the gap in current datasets primarily focused on urban environments. Existing datasets often lack coverage of highway scenarios, particularly those related to vehicle-to-everything (V2X) technology. H-V2X was created to advance research on highway perception by providing real-world, high-quality data from multi-sensor setups in highway settings.

- **Key Contributions**  
  - **First Large-Scale Highway Dataset**: H-V2X is the first large-scale dataset for highway roadside perception, incorporating data from real-world sensors on highways, covering over 100 km.
  - **Multi-Sensor Integration**: The dataset includes synchronized data from cameras and radars, with vector map information, ensuring comprehensive BEV-based perception.
  - **Three Key Tasks**: Introduces three critical tasks for highway perception: BEV detection, MOT (Multi-Object Tracking), and trajectory prediction, supported by ground truth data and benchmarks.
  - **New Benchmark Methods**: The paper presents innovative methods incorporating HDMap data for improved BEV detection and trajectory prediction in highway scenarios.



## V2V Datasets

- **V2V Datasets**: Vehicle-to-vehicle datasets capture collaboration between vehicles, facilitating research on cooperative perception under occlusion, sparse observations, or dynamic driving scenarios.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| COMAP | 2021 | ISPRS | L, C | Sim | 3DOD, SS | [download](https://demuc.de/colmap/) |
| CODD | 2021 | RA-L | L | Sim | Registration | [download](https://github.com/eduardohenriquearnold/fastreg) |
| OPV2V | 2022 | ICRA | C, L, R | Sim | 3DOD | [download](https://mobility-lab.seas.ucla.edu/opv2v/) |
| OPV2V+ | 2023 | CVPR | C, L, R | Sim | 3DOD | [download](https://siheng-chen.github.io/dataset/CoPerception+/) |
| V2V4Real | 2023 | CVPR | L, C | Real | 3DOD, MOT, S2R | [download](https://mobility-lab.seas.ucla.edu/v2v4real/) |
| LUCOOP | 2023 | IV | L | Real | 3DOD | [download](https://data.uni-hannover.de/vault/icsens/axmann/lucoop-leibniz-university-cooperative-perception-and-urban-navigation-dataset/) |
| MARS | 2024 | CVPR | L, C | Real | VPR, NR | [download](https://ai4ce.github.io/MARS/) |
| OPV2V-H | 2024 | ICLR | C, L, R | Sim | 3DOD | [download](https://github.com/yifanlu0227/HEAL) |
| V2V-QA | 2025 | arXiv | L, C | Real | 3DOD, PQA | [download](https://eddyhkchiu.github.io/v2vllm.github.io/) |
| CP-UAV | 2022 | NIPS | L, C | Sim | 3DOD | [download](https://siheng-chen.github.io/dataset/coperception-uav/) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.

### COMAP: A Synthetic Dataset for Collective Multi-Agent Perception of Autonomous Driving  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the challenges of single-agent perception in autonomous driving, particularly issues like occlusion and sensor noise. Traditional methods struggle with these challenges due to limited field-of-view (FOV) and imperfect data collection. Collective multi-agent perception (COMAP) enhances these systems by enabling data sharing across vehicles, improving the detection accuracy and robustness of autonomous driving.

- **Key Contributions**  
  - **Data Generator**: The paper introduces an efficient data generator for COMAP, capable of producing both image and point cloud data with ground truth for object detection and semantic segmentation.
  - **Scalable Simulation**: The generator allows for the creation of dense traffic scenarios without excessive computational resources, enabling scalability.
  - **Enhanced Perception**: Through experiments, COMAP's performance is shown to surpass single-agent perception, improving object detection accuracy, bounding box localization, and sensor noise robustness.
  - **Data Fusion Techniques**: The paper discusses different stages of data fusion (raw data fusion, deep feature fusion, and fully processed data fusion) and compares their effectiveness in improving perception performance.

### Fast and Robust Registration of Partially Overlapping Point Clouds  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the challenges faced in registering point clouds that partially overlap. Point cloud registration plays a vital role in several fields, including autonomous driving and 3D mapping, where accurate alignment of 3D sensor data from different viewpoints is critical. Existing methods often struggle with incomplete or noisy data, leading to suboptimal registration results.

- **Key Contributions**  
  - **Robust Registration Approach**: The paper proposes a novel technique for fast and robust registration of partially overlapping point clouds, improving accuracy in real-world applications.
  - **Handling Partial Overlap**: The method focuses on handling scenarios with partial overlap, which is common in practical applications like autonomous vehicle navigation and robotic mapping.
  - **Speed and Efficiency**: The proposed approach is designed to be computationally efficient, making it suitable for real-time applications in dynamic environments.
  - **Experimental Validation**: Extensive experiments demonstrate that the method outperforms traditional registration techniques in both accuracy and robustness, even in the presence of noise and partial overlap.

### OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Vehicle-to-Vehicle (V2V) communication offers the potential to improve perception performance in autonomous driving, but there is a lack of large-scale open datasets for V2V perception. This gap hampers the development and benchmarking of V2V algorithms, motivating the creation of the OPV2V dataset, the first large-scale open dataset for V2V perception.

- **Key Contributions**  
  - **First Open Dataset for V2V Perception**: OPV2V is the first large-scale open dataset specifically designed for Vehicle-to-Vehicle perception tasks.
  - **Benchmark with Multiple Fusion Strategies**: The paper introduces a benchmark that evaluates various fusion strategies (early, late, and intermediate fusion) combined with state-of-the-art 3D LiDAR detection algorithms.
  - **Proposed Attentive Intermediate Fusion Pipeline**: A new pipeline is proposed for aggregating information from multiple connected vehicles, which performs well even under high compression rates, improving the effectiveness of V2V communication.
  - **Open-source Availability**: The dataset, benchmark models, and code are publicly available, encouraging further research in V2V perception.

### Collaboration Helps Camera Overtake LiDAR in 3D Detection  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the challenge of improving camera-only 3D detection, which typically struggles with depth estimation compared to LiDAR. Traditional camera-based 3D detection has significant limitations, particularly in depth estimation, which is critical for accurate 3D object localization in autonomous driving. The authors propose a solution based on multi-agent collaboration to overcome these challenges.

- **Key Contributions**  
  - **CoCa3D Framework**: A novel collaborative camera-only 3D detection framework that utilizes multi-agent collaboration to improve depth estimation and 3D detection accuracy.
  - **Collaborative Depth Estimation**: The framework allows agents to share depth information, helping to resolve depth ambiguities and occlusions, improving overall detection accuracy.
  - **Improved Detection Performance**: With multiple agents working together, the framework significantly enhances detection performance, making camera-based systems competitive with LiDAR-based systems in certain scenarios.
  - **Efficient Communication**: The framework optimizes communication between agents by selecting and transmitting only the most informative cues, improving efficiency.
  - **Dataset Expansion**: The authors expanded existing datasets (OPV2V+, DAIR-V2X, and CoPerception-UAVs+) to include more collaborative agents and demonstrated that the collaborative camera system outperforms LiDAR in some cases, achieving state-of-the-art performance on multiple benchmarks.

### V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces V2V4Real, a real-world large-scale dataset designed to address the limitations of single-vehicle perception in autonomous driving. The lack of real-world datasets for Vehicle-to-Vehicle (V2V) cooperative perception has hindered progress in this area. This dataset aims to enhance the capabilities of V2V cooperative perception by providing multimodal data in real-world scenarios.

- **Key Contributions**  
  - **Real-World Large-Scale Dataset**: V2V4Real is the first large-scale real-world V2V dataset, collected over 410 km of road coverage.
  - **Multimodal Sensor Data**: The dataset includes 20K LiDAR frames and 40K RGB images with over 240K annotated 3D bounding boxes for five vehicle classes.
  - **Diverse Driving Scenarios**: It captures a variety of road types and driving scenarios, including intersections, highways, and city roads.
  - **Cooperative Perception Tasks**: It introduces three tasks for cooperative perception: 3D object detection, object tracking, and Sim2Real domain adaptation.
  - **Publicly Available**: The dataset and benchmarks will be made publicly available, encouraging further research and development in cooperative perception for autonomous driving.

### LUCOOP: Leibniz University Cooperative Perception and Urban Navigation Dataset  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Recent autonomous driving datasets mainly involve data collected from a single vehicle. However, to enhance cooperative driving applications like object detection and urban navigation, multi-vehicle datasets are needed. The LUCOOP dataset aims to fill this gap by providing real-world, time-synchronized multi-modal data from three interacting vehicles in an urban environment, fostering research in cooperative applications.

- **Key Contributions**  
  - **Multi-Vehicle Setup**: The LUCOOP dataset includes data from three interacting vehicles, each equipped with LiDAR, GNSS, and IMU sensors.
  - **V2V and V2X Range Measurements**: The dataset offers UWB range measurements, both Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2X).
  - **Ground Truth Trajectories**: Ground truth poses for each vehicle are provided, derived from GNSS/IMU integration, total station observations, and point cloud registration.
  - **3D Map Point Cloud**: A dense 3D map point cloud of the measurement area and an LOD2 city model are included for high-precision localization.
  - **Object Detection Annotations**: The dataset provides 3D bounding box annotations for static and dynamic vehicles, pedestrians, and other traffic participants.
  - **Large-Scale Data**: It includes over 54,000 LiDAR frames, 700,000 IMU measurements, and more than 2.5 hours of GNSS data.

### Multiagent Multitraversal Multimodal Self-Driving: Open MARS Dataset  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces the MARS dataset, aimed at addressing the gap in existing datasets by incorporating multiagent and multitraversal elements. Traditional autonomous driving datasets often lack collaborative and repeated traversals, which limit advancements in perception, prediction, and planning. MARS was developed to fill this gap, enabling richer research in multiagent systems and enhanced 3D scene understanding.

- **Key Contributions**  
  - **Multiagent Collection**: MARS includes data from multiple autonomous vehicles operating in the same geographical region, enabling collaborative 3D perception.
  - **Multitraversal Data**: It captures multiple traversals of the same locations under various conditions, enhancing 3D scene understanding over time.
  - **Multimodal Sensor Setup**: The dataset features a full 360-degree sensor suite, including LiDAR and RGB cameras, for comprehensive scene analysis.
  - **Research Opportunities**: MARS opens new avenues for research in multiagent collaborative perception, unsupervised learning, and multitraversal 3D reconstruction.
  - **Real-World Data**: The dataset was collected using May Mobility's autonomous vehicles, ensuring high scalability and diversity across locations.

### An Extensible Framework for Open Heterogeneous Collaborative Perception  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Collaborative perception enhances the capabilities of individual agents by enabling data sharing to overcome perception limitations such as occlusion. However, existing models typically assume agents are homogeneous, while real-world scenarios often involve heterogeneous agents with different sensor modalities and models. This paper addresses the challenge of integrating new heterogeneous agents with minimal cost and high performance.

- **Key Contributions**  
  - **HEAL Framework**: The paper introduces HEAL, an extensible framework for open heterogeneous collaborative perception, designed to integrate new agent types seamlessly into existing systems.
  - **Unified Feature Space**: HEAL uses a novel multi-scale, foreground-aware Pyramid Fusion network to establish a unified feature space for all agents.
  - **Backward Alignment Mechanism**: New agents are integrated via a backward alignment process that minimizes training costs by aligning their features to the unified space without requiring retraining of the entire model.
  - **OPV2V-H Dataset**: The paper presents the OPV2V-H dataset, which includes more diverse sensor types to support heterogeneous collaborative perception research.
  - **Performance**: HEAL outperforms existing methods in collaborative detection, reducing training parameters by 91.5% when integrating new agent types, demonstrating its efficiency and scalability.

### V2V-QA - V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models [[paper](https://arxiv.org/abs/2502.09980)] [[code](https://github.com/eddyhkchiu/V2VLLM)] [[project](https://eddyhkchiu.github.io/v2vllm.github.io)]

- **Background and Motivation**  
The paper addresses the limitations of autonomous vehicles relying solely on individual sensors for perception, especially when sensors are malfunctioning or occluded. Vehicle-to-vehicle (V2V) communication for cooperative perception is introduced to mitigate these issues, focusing on using Large Language Models (LLMs) for collaborative planning in autonomous driving.

- **Key Contributions**  
  - **V2V-QA Dataset**: A new dataset that includes grounding, notable object identification, and planning tasks designed for cooperative autonomous driving scenarios.
  - **V2V-LLM Model**: The proposed Vehicle-to-Vehicle Large Language Model integrates the perception data from multiple connected autonomous vehicles (CAVs) and answers driving-related questions.
  - **Improved Performance**: The V2V-LLM outperforms other baseline fusion methods in notable object identification and planning tasks, demonstrating its potential as a unified model for cooperative autonomous driving.
  - **New Research Direction**: Establishes a novel approach to cooperative autonomous driving using LLMs, improving safety and collaborative performance across multiple vehicles.


### CoPerception-UAV - Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps  [[paper&review](https://openreview.net/forum?id=dLL4KXzKUpS)] [[code](https://github.com/MediaBrain-SJTU/where2comm)] [[project](https://siheng-chen.github.io/dataset/coperception-uav)]

- **Background and Motivation**  
Collaborative perception allows multiple agents to share complementary information, enhancing overall perception. However, there is a challenge in balancing the perception performance with the communication bandwidth. Previous methods have not addressed the issue of efficiently selecting which spatial areas to communicate, leading to excessive bandwidth usage. This paper proposes a solution by focusing on sharing only perceptually critical information, optimizing communication efficiency.

- **Key Contributions**  
  - **Spatial Confidence Map**: Introduces a spatial confidence map that highlights the perceptually critical areas, helping agents focus on the most important information.
  - **Efficient Communication**: Develops a communication-efficient framework, Where2comm, which uses spatial confidence maps to reduce bandwidth consumption by transmitting sparse but critical data.
  - **Dynamic Adaptation**: Where2comm adapts to varying communication bandwidths, dynamically adjusting which spatial areas are communicated based on their perceptual importance.
  - **Superior Performance**: Evaluates Where2comm on several datasets, showing it consistently outperforms existing methods, reducing communication volume by more than 100,000 times while improving perception performance.
  - **Real-World and Simulation Scenarios**: Demonstrates the framework’s robustness in both real-world and simulation environments, with multi-agent setups including cars and drones equipped with cameras and LiDAR sensors.

















## V2I Datasets

- **V2I Datasets**: These datasets involve communication between vehicles and infrastructure, supporting cooperative tasks like object detection, tracking, and decision-making in connected environments.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| CoopInf | 2020 | TITS | L, C | Sim | 3DOD | [download](https://github.com/eduardohenriquearnold/coop-3dod-infra?tab=readme-ov-file) |
| DAIR-V2X-C | 2022 | CVPR | L, C | Real | 3DOD | [download](https://air.tsinghua.edu.cn/DAIR-V2X/index.html) |
| V2X-Seq | 2023 | CVPR | L, C | Real | 3DOT, TP | [download](https://github.com/AIR-THU/DAIR-V2X-Seq) |
| HoloVIC | 2024 | CVPR | L, C | Real | 3DOD, MOT | [download](https://holovic.net) |
| OTVIC | 2024 | IROS | L, C | Real | 3DOD | [download](https://sites.google.com/view/otvic) |
| DAIR-V2XReid | 2024 | TITS | L, C | Real | 3DOD, Re-ID | [download](https://github.com/Niuyaqing/DAIR-V2XReid) |
| TUMTraf V2X | 2024 | CVPR | L, C | Real | 3DOD, MOT | [download](https://tum-traffic-dataset.github.io/tumtraf-v2x/) |
| V2X-Radar | 2024 | arxiv | L, C, R | Real | 3DOD | [download](http://openmpd.com/column/V2X-Radar) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.


### Cooperative Perception for 3D Object Detection in Driving Scenarios Using Infrastructure Sensors  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the challenge of 3D object detection in autonomous driving scenarios, particularly in complex environments like T-junctions and roundabouts. Traditional single-sensor systems face limitations such as occlusion and restricted field-of-view, hindering reliable detection. Cooperative perception using multiple spatially diverse infrastructure sensors provides an effective solution to mitigate these issues and improve detection performance.

- **Key Contributions**  
  - **Cooperative 3D Object Detection Schemes**: Two fusion schemes—early and late fusion—are proposed for cooperative perception. Early fusion combines raw sensor data before detection, while late fusion combines detected objects post-detection.
  - **Evaluation of Fusion Approaches**: The paper compares both fusion schemes and their hybrid combination in terms of detection performance and communication costs. Early fusion outperforms late fusion but requires higher bandwidth.
  - **Novel Cooperative Dataset**: A synthetic dataset for cooperative 3D object detection in driving scenarios, including a T-junction and roundabout, is introduced for performance evaluation.
  - **Impact of Sensor Configurations**: The study also evaluates how sensor number and positioning impact detection performance, offering practical insights for real-world deployment .


### DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Autonomous driving still faces significant challenges, especially in terms of long-range perception and global awareness. While vehicle sensors have limitations, combining vehicle and infrastructure data can overcome these challenges. However, there was a lack of real-world datasets for vehicle-infrastructure cooperative problems. This paper introduces the DAIR-V2X dataset to facilitate research in this area.

- **Key Contributions**  
  - **First Vehicle-Infrastructure Cooperative Dataset**: DAIR-V2X is the first large-scale, multi-modality dataset captured from real-world scenarios for vehicle-infrastructure cooperation in 3D object detection.
  - **VIC3D Task Definition**: The paper defines the VIC3D task, which focuses on collaboratively detecting and identifying 3D objects using sensory inputs from both vehicle and infrastructure.
  - **VIC3D Benchmark and Fusion Framework**: It introduces benchmarks for VIC3D object detection and proposes the Time Compensation Late Fusion (TCLF) framework to handle temporal asynchrony.
  - **Real-World Data**: The dataset includes 71k frames of LiDAR and camera data, collected across various environments with 3D annotations.
  - **Performance Improvement**: The results demonstrate that integrating vehicle and infrastructure data leads to better performance than using single-source data.

### V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces the V2X-Seq dataset, addressing the need for real-world sequential datasets in vehicle-infrastructure cooperative perception and forecasting. Current research mainly focuses on improving perception using infrastructure data for frame-by-frame 3D detection, but lacks datasets for tracking and forecasting, which are critical for decision-making in autonomous driving.

- **Key Contributions**  
  - **Release of V2X-Seq**: The first large-scale, real-world sequential V2X dataset with data on vehicle and infrastructure cooperation.
  - **Two Main Parts**: Includes the sequential perception dataset with 15,000 frames from 95 scenarios, and the trajectory forecasting dataset with 210,000 scenarios.
  - **Three New Tasks**: Introduces VIC3D Tracking, Online-VIC Forecasting, and Offline-VIC Forecasting, with benchmarks to evaluate these tasks.
  - **Proposed FF-Tracking Method**: A middle fusion framework that efficiently solves the VIC3D tracking problem, handling latency challenges effectively.


### HoloVIC: Large-scale Dataset and Benchmark for Multi-Sensor Holographic Intersection and Vehicle-Infrastructure Cooperative  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The increasing complexity of traffic environments, including occlusions and blind spots, limits the effectiveness of single-viewpoint roadside sensing systems. To enhance the perception capabilities of roadside systems, the paper presents HoloVIC, a large-scale multi-sensor holographic dataset, aiming to improve vehicle-infrastructure cooperation (VIC) by capturing synchronized data from various sensors installed at different intersections.

- **Key Contributions**  
  - **HoloVIC Dataset**: A comprehensive multi-sensor dataset for vehicle-infrastructure cooperation, featuring data collected from holographic intersections equipped with multiple sensor layouts (Cameras, Lidar, Fisheye).
  - **Synchronized Multi-Sensor Data**: The dataset includes 100k+ synchronized frames with annotated 3D bounding boxes, covering diverse sensor configurations (e.g., 4C+2L, 12C+4F+2L).
  - **Benchmark and Tasks**: The authors formulate five key tasks (e.g., Mono3D, Lidar 3D Detection, Multi-sensor Multi-object Tracking, VIC Perception) to promote research on roadside perception and vehicle-infrastructure cooperation.
  - **High-Quality Annotations**: The dataset provides 3D bounding boxes and object IDs associated with different sensors, enabling the study of multi-sensor fusion and trajectory tracking across various intersection layouts.


### OTVIC: A Dataset with Online Transmission for Vehicle-to-Infrastructure Cooperative 3D Object Detection  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper introduces OTVIC, a real-world dataset designed for vehicle-to-infrastructure (V2I) cooperative 3D object detection. Traditional autonomous driving datasets often neglect the challenges of real-time perception data transmission from infrastructure to vehicles. OTVIC addresses these challenges by simulating the communication delays, bandwidth limitations, and high vehicle speeds encountered in real-world highway environments.

- **Key Contributions**  
  - **Real-Time Online Transmission**: OTVIC is the first multi-modality, multi-view dataset that includes online transmission from real-world scenarios, focusing on vehicle-to-infrastructure cooperative 3D object detection.
  - **Multi-Modal and Multi-View Dataset**: The dataset features synchronized data from multiple sensors, including images, LiDAR point clouds, and vehicle motion data, collected from highways at real-time speeds.
  - **Fusion Framework (LfFormer)**: The paper proposes LfFormer, a novel end-to-end multi-modality late fusion framework using transformer architecture, which proves to be effective and robust for cooperative 3D object detection.
  - **Real-World Relevance**: OTVIC emphasizes real-world challenges like varying transmission delays and environmental factors, offering a crucial resource for the development of real-time cooperative perception systems in autonomous driving.


### DAIR-V2XReid: A New Real-World Vehicle-Infrastructure Cooperative Re-ID Dataset and Cross-Shot Feature Aggregation Network Perception Method  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
Vehicle Re-Identification (Re-ID) has a critical role in enhancing Vehicle-Infrastructure Cooperative Autonomous Driving (VICAD). Existing datasets are insufficient for evaluating the performance of Re-ID algorithms in real-world scenarios, particularly for cross-shot vehicle identification. The lack of a comprehensive dataset and effective models for addressing large variations in vehicle appearance across different cameras has hindered progress in VICAD research.

- **Key Contributions**  
  - **DAIR-V2XReid Dataset**: The paper introduces the DAIR-V2XReid dataset, the first real-world VIC Re-ID dataset, constructed using both vehicle-mounted and roadside cameras for vehicle re-identification tasks.  
  - **Cross-shot Feature Aggregation Network (CFA-Net)**: To address the issue of large appearance differences across different cameras, the CFA-Net model was proposed. It combines three key modules: a camera embedding module, a cross-stage feature fusion module, and a multi-directional attention module.  
  - **State-of-the-art Performance**: The proposed CFA-Net achieves the highest reported performance on the DAIR-V2XReid dataset, significantly improving the accuracy of vehicle Re-ID across different camera views.  
  - **Versatile Application**: The model demonstrates good generalization abilities, as evidenced by experiments on the VeRi776 dataset, further confirming its robustness and efficiency for real-world applications.

### TUMTraf-V2X: Cooperative Perception Dataset for 3D Object Detection in Driving Scenarios  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
This paper presents the TUMTraf-V2X dataset, which aims to enhance vehicle perception through cooperative sensing. Roadside sensors and onboard sensors are used to overcome the limitations of single-sensor systems, especially occlusion and limited field of view. The dataset focuses on 3D object detection and tracking to improve road safety and autonomous driving capabilities.

- **Key Contributions**  
  - **High-Quality V2X Dataset**: The dataset includes 2,000 labeled point clouds and 5,000 labeled images with 30,000 3D bounding boxes. It covers challenging traffic scenarios, including near-miss events and traffic violations.
  - **Cooperative 3D Object Detection Model (CoopDet3D)**: A new cooperative fusion model, CoopDet3D, which outperforms traditional camera-LiDAR fusion methods with a +14.3 3D mAP improvement.
  - **Open-Source Tools and Resources**: The dataset and related tools, such as the 3D BAT labeling tool and a development kit, are made publicly available to facilitate integration and model training.
  - **Benchmarking and Evaluation**: Extensive experiments show that cooperative perception models lead to better detection accuracy than vehicle-only systems, proving the benefit of V2X collaboration in object detection tasks.


### V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception  [[paper](https://arxiv.org/abs/2411.10962)] [[code](https://github.com/yanglei18/V2X-Radar)] [[project](http://openmpd.com/column/V2X-Radar)]

- **Background and Motivation**  
The V2X-Radar dataset was developed to address the limitations in existing cooperative perception datasets, which often focus solely on camera and LiDAR data. The goal is to bridge the gap by incorporating 4D Radar, a sensor that excels in adverse weather conditions. The dataset aims to enhance the robustness of autonomous driving perception systems by addressing occlusions and limited range in single-vehicle systems.

- **Key Contributions**  
  - **First Real-World Multi-modal Dataset**: V2X-Radar is the first large, real-world dataset incorporating 4D Radar, alongside LiDAR and camera data.
  - **Comprehensive Data Coverage**: It includes 20K LiDAR frames, 40K camera images, and 20K 4D Radar data, with 350K annotated bounding boxes spanning five object categories.
  - **Three Specialized Sub-datasets**: The dataset is divided into V2X-Radar-C (cooperative perception), V2X-Radar-I (roadside perception), and V2X-Radar-V (single-vehicle perception).
  - **Extensive Benchmarking**: The dataset includes benchmarks for recent perception algorithms across these sub-datasets, supporting a wide range of research in cooperative perception.



## V2X Datasets
- **V2X Datasets**: Covering vehicle-to-everything communication, these datasets integrate multiple agents such as vehicles, infrastructure, and other environmental elements like drones or pedestrians, enabling research in complex collaborative scenarios.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| V2X-Sim | 2022 | RA-L | L, C | Sim | 3DOD, MOT, SS | [download](https://ai4ce.github.io/V2X-Sim/download.html) |
| V2XSet | 2022 | ECCV | L, C | Sim | 3DOD | [download](https://paperswithcode.com/dataset/v2xset) |
| DOLPHINS | 2022 | ACCV | L, C | Sim | 2DOD, 3DOD | [download](https://dolphins-dataset.net/) |
| DeepAccident | 2024 | AAAI | L, C | Sim | 3DOD, MOT, SS, TP | [download](https://deepaccident.github.io/) |
| V2X-Real | 2024 | ECCV | L, C | Real | 3DOD | [download](https://mobility-lab.seas.ucla.edu/v2x-real) |
| Multi-V2X | 2024 | arxiv | L, C | Sim | 3DOD, MOT | [download](http://github.com/RadetzkyLi/Multi-V2X) |
| Adver-City | 2024 | arxiv | L, C | Sim | 3DOD, MOT, SS | [download](https://labs.cs.queensu.ca/quarrg/datasets/adver-city/) |
| DAIR-V2X-Traj | 2024 | NIPS | L, C | Real | MF | [download](https://github.com/AIR-THU/V2X-Graph) |
| WHALES | 2024 | arxiv | L, C | Sim | 3DOD | [download](https://github.com/chensiweiTHU/WHALES) |
| V2X-R | 2024 | arxiv | L, C, R | Sim | 3DOD | [download](https://github.com/ylwhxht/V2X-R) |
| V2XPnP-Seq | 2024 | arxiv | L, C | Real | Perception and Prediction | [download](https://mobility-lab.seas.ucla.edu/v2xpnp/) |
| Mixed Signals | 2025 | arxiv | L | Real | 3DOD | [download](https://mixedsignalsdataset.cs.cornell.edu/) |
| SCOPE | 2024 | arxiv | C, L | Sim | 2DOD, 3DOD, SS, S2R | [download](https://ekut-es.github.io/scope) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.


### V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving [[paper](https://arxiv.org/abs/2202.08449)] [[code](https://github.com/ai4ce/V2X-Sim)] [[project](https://ai4ce.github.io/V2X-Sim)]

- **Background and Motivation**  
The paper addresses the limitations of single-vehicle perception in autonomous driving, particularly regarding occlusions and limited sensing range. Vehicle-to-Everything (V2X) communication enhances this by enabling collaboration among multiple agents, allowing for a broader and clearer understanding of the environment. This dataset and benchmark aim to fill the gap in the field by providing the first public multi-agent, multi-modality collaborative perception dataset, facilitating research in collaborative perception tasks.

- **Key Contributions**  
  - **V2X-Sim Dataset**: The paper introduces the V2X-Sim dataset, which supports multi-agent, multi-modality, and multi-task perception tasks, enabling research in collaborative perception for autonomous driving.
  - **Open-Source Testbed**: It provides an open-source testbed for testing collaborative perception methods, offering a benchmark for three critical tasks: collaborative detection, tracking, and segmentation.
  - **Comprehensive Sensor Suite**: The dataset includes various sensor modalities from both vehicles and road-side units (RSUs), enhancing perception across different environments.
  - **Collaborative Perception Strategies**: The study uses state-of-the-art collaboration strategies to evaluate the dataset and advance collaborative perception research .


### V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper discusses the challenges of autonomous vehicle (AV) perception in complex driving environments, particularly when vehicles suffer from occlusions and limited sensor range. Vehicle-to-Everything (V2X) communication, including collaboration with infrastructure, is introduced as a solution to enhance perception. However, integrating information from heterogeneous agents like vehicles and infrastructure presents unique challenges, which this paper aims to address.

- **Key Contributions**  
  - **Unified V2X Vision Transformer (V2X-ViT)**: Introduces a novel Transformer architecture for cooperative perception that handles the challenges of heterogeneous V2X systems, such as sensor misalignment, noise, and asynchronous data sharing.
  - **Heterogeneous Multi-Agent Attention Module**: A customized attention mechanism designed to account for the different agent types (vehicles and infrastructure) and their connections during the feature fusion process.
  - **Multi-Scale Window Attention**: A multi-resolution attention mechanism that helps mitigate the effects of localization errors and time delays in the data from agents.
  - **V2XSet Dataset**: The creation of a large-scale, open dataset designed to simulate real-world communication conditions and evaluate the V2X-ViT framework.

### DOLPHINS: Dataset for Collaborative Perception enabled Harmonious and Interconnected Self-driving [[paper](https://arxiv.org/abs/2207.07609)] [[code](https://github.com/explosion5/Dolphins)] [[project](https://dolphins-dataset.net)]

- **Background and Motivation**  
The paper addresses the limitations of standalone perception in autonomous driving, such as blind zones and occlusions. To overcome this, Vehicle-to-Everything (V2X) communication enables collaborative perception through data sharing between vehicles and road-side units (RSUs). However, a lack of large-scale, multi-view, and multi-modal datasets has hindered the development of collaborative perception algorithms, motivating the creation of the DOLPHINS dataset.

- **Key Contributions**  
  - **DOLPHINS Dataset**: A large-scale dataset featuring various autonomous driving scenarios with multi-view, multi-modality data (images and point clouds) from both vehicles and RSUs.
  - **Diverse Scenarios and High Resolution**: The dataset includes 6 typical driving scenarios, such as urban intersections, highways, and mountain roads, with high-resolution data (Full-HD images and 64-line LiDARs).
  - **Temporally-Aligned Data**: The data from multiple viewpoints (vehicles and RSUs) are temporally aligned, enabling comprehensive collaborative perception.
  - **Benchmark for Collaborative Perception**: The paper provides a benchmark for 2D and 3D object detection, as well as multi-view collaborative perception tasks, demonstrating the effectiveness of V2X communication in improving detection accuracy and reducing sensor costs.



### Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions[[paper](https://arxiv.org/abs/2410.06380)] [[code](https://github.com/QUARRG/Adver-City)] [[project](https://labs.cs.queensu.ca/quarrg/datasets/adver-city)]

- **Background and Motivation**  
Adverse weather conditions like rain, fog, and glare challenge the performance of Autonomous Vehicles (AVs) and Collaborative Perception (CP) models. The lack of datasets focusing on these conditions limits the evaluation and improvement of CP under such scenarios. Adver-City aims to address this gap by providing the first open-source CP dataset under diverse adverse weather conditions, including glare, a first for synthetic CP datasets.

- **Key Contributions**  
  - **First Open-Source CP Dataset for Adverse Weather**: Adver-City is the first synthetic, open-source CP dataset focused on weather conditions like rain, fog, and glare.
  - **Varied Weather and Object Density Scenarios**: The dataset includes 110 scenarios across six weather conditions and varying object densities to simulate real-world challenges in adverse weather.
  - **Multi-Sensor Data**: It includes data from LiDARs, RGB cameras, semantic segmentation cameras, GNSS, and IMUs, supporting tasks such as 3D object detection, tracking, and semantic segmentation.
  - **Realistic Scenario Design**: Scenarios are based on real crash reports, improving relevance for autonomous driving research.
  - **Benchmarking**: Performance benchmarks highlight the challenges posed by adverse weather on multi-modal object detection models, with performance drops observed in certain conditions.












- **CP-GuardBench** (CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception) [[paper&review](https://openreview.net/forum?id=9MNzHTSDgh)] [~~code~~] [~~project~~]
- **Griffin** (Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark) [[paper](https://arxiv.org/abs/2503.06983)] [[code](https://github.com/wang-jh18-SVM/Griffin)] [[project](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g)]
- {Real} **Mixed Signals** (Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration) [[paper](https://arxiv.org/abs/2502.14156)] [[code](https://github.com/chinitaberrio/Mixed-Signals)] [[project](https://mixedsignalsdataset.cs.cornell.edu)]
- **Multi-V2X** (Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception) [[paper](https://arxiv.org/abs/2409.04980)] [[code](https://github.com/RadetzkyLi/Multi-V2X)] [~~project~~]
- **OPV2V-N** (RCDN: Towards Robust Camera-Insensitivity Collaborative Perception via Dynamic Feature-based 3D Neural Modeling) [[paper](https://arxiv.org/abs/2405.16868)] [~~code~~] [~~project~~]
- {Real} **V2XPnP-Seq** (V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction) [[paper](https://arxiv.org/abs/2412.01812)] [[code](https://github.com/Zewei-Zhou/V2XPnP)] [[project](https://mobility-lab.seas.ucla.edu/v2xpnp)]
- {Real} **V2X-Real** (V2X-Real: a Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception) [[paper](https://arxiv.org/abs/2403.16034)] [~~code~~] [[project](https://mobility-lab.seas.ucla.edu/v2x-real)]
- {Real} **V2X-ReaLO** (V2X-ReaLO: An Open Online Framework and Dataset for Cooperative Perception in Reality) [[paper](https://arxiv.org/abs/2503.10034)] [~~code~~] [~~project~~]
- **WHALES** (WHALES: A Multi-Agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving) [[paper](https://arxiv.org/abs/2411.13340)] [[code](https://github.com/chensiweiTHU/WHALES)] [[project](https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A)]





## I2I Datasets
- **I2I Datasets**: Focused on infrastructure-to-infrastructure collaboration, these datasets support research in scenarios with overlapping sensor coverage or distributed sensor fusion across intersections.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| Rcooper | 2024 | CVPR | C, L | Real | 3DOD, MOT | [download](https://github.com/AIR-THU/DAIR-Rcooper) |
| InScope | 2024 | arxiv | L | Real | 3DOD, MOT | [download](https://github.com/xf-zh/InScope) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.

- {Real} **InScope** (InScope: A New Real-world 3D Infrastructure-side Collaborative Perception Dataset for Open Traffic Scenarios) [[paper](https://arxiv.org/abs/2407.21581)] [[code](https://github.com/xf-zh/InScope)] [~~project~~]









## Methods






### 
### 
### 
### 

### Selected Preprint


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


### ECCV 2022

- **V2XSet** (V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer) [[paper](https://arxiv.org/abs/2203.10638)] [[code](https://github.com/DerrickXuNu/v2x-vit)] [[project](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6)]

### ICRA 2022

- **OPV2V** (OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication) [[paper](https://arxiv.org/abs/2109.07644)] [[code](https://github.com/DerrickXuNu/OpenCOOD)] [[project](https://mobility-lab.seas.ucla.edu/opv2v)]

### ACCV 2022





