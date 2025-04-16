# Collaborative / Cooperative / Perception Datasets - Updating
## Overview

This repository consolidates **Collaborative Perception (CP)** datasets for autonomous driving, covering a wide range of communication paradigms, including pure **roadside perception**, **Vehicle-to-Vehicle (V2V)**, **Vehicle-to-Infrastructure (V2I)**, **Vehicle-to-Everything (V2X)**, and **Infrastructure-to-Infrastructure (I2I)** scenarios. It includes nearly all publicly available **CP** datasets and provides links to relevant publications, source code, and dataset downloads, offering researchers an efficient and centralized resource to aid their research and development in this field.

First, the repository introduces commonly used **autonomous driving simulation tools**, followed by categorizing **CP datasets** based on collaboration paradigms, presented in a tabular format. Each dataset is then described in detail, helping readers better understand the characteristics and applicable scenarios of each dataset. In addition, the repository also consolidates classic methods and cutting-edge research in **collaborative perception**, providing valuable insights into current trends and future directions in the field.




### :link:Jump to:
- ### [Simulator](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#simulator)
- ### [Roadside Datasets](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#roadside-datasets)
- ### [V2V Datasets](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#v2v-datasets)
- ### [V2I Datasets](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#v2i-datasets)
- ### [V2X Datasets](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#v2x-datasets)
- ### [I2I Datasets](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#i2i-datasets)
- ### [Methods](https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving#methods)

## :bookmark:Simulator 

- Simulators for Collaborative Perception (CP) Research

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

   
## :bookmark:Roadside Datasets

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| Ko-PER | 2014 | ITSC | C, L | Real | 3DOD, MOT | [download](https://www.uni-ulm.de/in/mrm/forschung/datensaetze.html) |
| CityFlow | 2019 | CVPR | C | Real | MTSCT/MTMCT, ReID | [download](https://cityflow-project.github.io/) |
| BAAI-VANJEE  | 2021     | arXiv     | C, L        | Real       | 2D/3D OD   | [Link](https://paperswithcode.com/dataset/baai-vanjee) |
| WIBAM        | 2021     | arXiv     | C           | Real       | 2D/3D OD   | [Link](https://github.com/MatthewHowe/WIBAM) |
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

### BAAI-VANJEE Roadside Dataset: Towards the Connected Automated Vehicle Highway Technologies in Challenging Environments of China

- **Background and Motivation**  
This paper introduces the BAAI-VANJEE dataset, aimed at enhancing roadside perception for Connected Automated Vehicle Highway (CAVH) technologies. The dataset was created in response to the limitations of vehicle-based technologies that are difficult to scale. It provides high-quality LiDAR and RGB data collected from roadside sensors, addressing the need for datasets that can help improve detection tasks, including 2D/3D object detection and multi-sensor fusion in complex traffic environments.

- **Key Contributions**  
  - **Challenging Roadside Dataset**: The dataset includes 2500 frames of LiDAR data and 5000 frames of RGB images, with annotations for 12 object classes.
  - **High-Quality Annotations**: 74K 3D object annotations and 105K 2D object annotations, collected under varying weather conditions (sunny, cloudy, rainy) and times of day (day, night).
  - **Real-World Application**: Focuses on complex urban intersections and highway scenes, providing real-world data for CAVH research.
  - **Three Core Tasks**: Supports tasks including 2D object detection, 3D object detection, and multi-sensor fusion.
  - **Diverse Scenarios**: Includes data from diverse traffic conditions, providing a more comprehensive view of roadside perception.
  - **Public Availability**: The dataset is available online to support research in intelligent transportation and big data-driven innovation.

### WIBAM  Weakly Supervised Training of Monocular 3D Object Detectors Using Wide Baseline Multi-view Traffic Camera Data


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


## :bookmark:V2V Datasets

- **V2V Datasets**: Vehicle-to-vehicle datasets capture collaboration between vehicles, facilitating research on cooperative perception under occlusion, sparse observations, or dynamic driving scenarios.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| T & J        | 2019     | ICDCS     | C, L, R     | Real       | 3D OD      | -            |
| V2V-Sim      | 2020     | ECCV      | L           | Sim        | 3D OD      | -            |
| COMAP | 2021 | ISPRS | L, C | Sim | 3DOD, SS | [download](https://demuc.de/colmap/) |
| CODD | 2021 | RA-L | L | Sim | Registration | [download](https://github.com/eduardohenriquearnold/fastreg) |
| OPV2V | 2022 | ICRA | C, L, R | Sim | 3DOD | [download](https://mobility-lab.seas.ucla.edu/opv2v/) |
| OPV2V+ | 2023 | CVPR | C, L, R | Sim | 3DOD | [download](https://siheng-chen.github.io/dataset/CoPerception+/) |
| IRV2V        | 2023     | NIPS      | L, C        | Sim        | 3D OD      | [Link](https://paperswithcode.com/dataset/irv2v) |
| V2V4Real | 2023 | CVPR | L, C | Real | 3DOD, MOT, S2R | [download](https://mobility-lab.seas.ucla.edu/v2v4real/) |
| LUCOOP | 2023 | IV | L | Real | 3DOD | [download](https://data.uni-hannover.de/vault/icsens/axmann/lucoop-leibniz-university-cooperative-perception-and-urban-navigation-dataset/) |
| MARS | 2024 | CVPR | L, C | Real | VPR, NR | [download](https://ai4ce.github.io/MARS/) |
| OPV2V-H | 2024 | ICLR | C, L, R | Sim | 3DOD | [download](https://github.com/yifanlu0227/HEAL) |
| V2V-QA | 2025 | arXiv | L, C | Real | 3DOD, PQA | [download](https://eddyhkchiu.github.io/v2vllm.github.io/) |
| CP-UAV | 2022 | NIPS | L, C | Sim | 3DOD | [download](https://siheng-chen.github.io/dataset/coperception-uav/) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.


### **T & J**  Cooper: Cooperative Perception for Connected Autonomous Vehicles Based on 3D Point Clouds

- **Background and Motivation**  
The paper introduces the Cooper system, aimed at enhancing the detection accuracy of autonomous vehicles through cooperative perception. Autonomous vehicles often suffer from sensor limitations, leading to potential detection failures. By enabling connected vehicles to share sensor data, particularly 3D LiDAR point clouds, the system aims to extend sensing areas, improve detection accuracy, and enhance safety in dynamic driving environments.

- **Key Contributions**  
  - **Sparse Point-Cloud Object Detection (SPOD)**: Introduces the SPOD method for detecting objects in sparse LiDAR point clouds, which improves detection even in low-density data.
  - **Cooperative Perception System**: Demonstrates how multiple connected autonomous vehicles can share LiDAR data, merging point clouds from different vehicles to enhance object detection.
  - **Improved Detection Performance**: Shows that cooperative perception expands the sensing area, improves detection accuracy, and complements traditional object detection methods.
  - **Data Transmission Feasibility**: Demonstrates the feasibility of transmitting LiDAR point clouds for cooperative perception using existing vehicular network technologies, maintaining efficiency even with limited bandwidth.
  - **Evaluation on Real-World Datasets**: Evaluates the Cooper system on the KITTI and T&J datasets, showing significant improvements in detection performance, especially for objects that were previously undetected by individual vehicles.


### **V2V-Sim**  V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Prediction

- **Background and Motivation**  
This paper explores the use of Vehicle-to-Vehicle (V2V) communication to enhance the perception and motion forecasting of self-driving vehicles (SDVs). The key motivation is the challenge that SDVs face in detecting and forecasting the behavior of objects that are occluded or far away, which can be critical in safety-sensitive situations. By leveraging information shared from nearby vehicles, SDVs can overcome these limitations and improve overall safety and efficiency.

- **Key Contributions**  
  - **V2V Communication for Perception and Prediction**: Introduces a novel V2V approach, V2VNet, which integrates shared information from multiple vehicles to improve detection and motion forecasting accuracy.
  - **Compression of Intermediate Representations**: The model transmits compressed intermediate feature maps from the perception and prediction (P&P) neural network, balancing accuracy and bandwidth efficiency.
  - **Graph Neural Network (GNN)**: Utilizes a spatially aware GNN to aggregate information received from other SDVs, allowing intelligent fusion of data from different time points and viewpoints.
  - **V2V-Sim Dataset**: Proposes the creation of a new dataset, V2V-Sim, that simulates the real-world conditions where multiple SDVs share information, demonstrating the effectiveness of the V2VNet approach.

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


### OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication   [[paper](https://arxiv.org/abs/2109.07644)] [[code](https://github.com/DerrickXuNu/OpenCOOD)] [[project](https://mobility-lab.seas.ucla.edu/opv2v)]

- **Background and Motivation**  
Vehicle-to-Vehicle (V2V) communication offers the potential to improve perception performance in autonomous driving, but there is a lack of large-scale open datasets for V2V perception. This gap hampers the development and benchmarking of V2V algorithms, motivating the creation of the OPV2V dataset, the first large-scale open dataset for V2V perception.

- **Key Contributions**  
  - **First Open Dataset for V2V Perception**: OPV2V is the first large-scale open dataset specifically designed for Vehicle-to-Vehicle perception tasks.
  - **Benchmark with Multiple Fusion Strategies**: The paper introduces a benchmark that evaluates various fusion strategies (early, late, and intermediate fusion) combined with state-of-the-art 3D LiDAR detection algorithms.
  - **Proposed Attentive Intermediate Fusion Pipeline**: A new pipeline is proposed for aggregating information from multiple connected vehicles, which performs well even under high compression rates, improving the effectiveness of V2V communication.
  - **Open-source Availability**: The dataset, benchmark models, and code are publicly available, encouraging further research in V2V perception.


### **OPV2V+**  Collaboration Helps Camera Overtake LiDAR in 3D Detection   [[paper](https://arxiv.org/abs/2303.13560)] [[code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[project](https://siheng-chen.github.io/dataset/CoPerception+)]
### **CoPerception-UAV+**  Collaboration Helps Camera Overtake LiDAR in 3D Detection   [[paper](https://arxiv.org/abs/2303.13560)] [[code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[project](https://siheng-chen.github.io/dataset/CoPerception+)]

- **Background and Motivation**  
The paper addresses the challenge of improving camera-only 3D detection, which typically struggles with depth estimation compared to LiDAR. Traditional camera-based 3D detection has significant limitations, particularly in depth estimation, which is critical for accurate 3D object localization in autonomous driving. The authors propose a solution based on multi-agent collaboration to overcome these challenges.

- **Key Contributions**  
  - **CoCa3D Framework**: A novel collaborative camera-only 3D detection framework that utilizes multi-agent collaboration to improve depth estimation and 3D detection accuracy.
  - **Collaborative Depth Estimation**: The framework allows agents to share depth information, helping to resolve depth ambiguities and occlusions, improving overall detection accuracy.
  - **Improved Detection Performance**: With multiple agents working together, the framework significantly enhances detection performance, making camera-based systems competitive with LiDAR-based systems in certain scenarios.
  - **Efficient Communication**: The framework optimizes communication between agents by selecting and transmitting only the most informative cues, improving efficiency.
  - **Dataset Expansion**: The authors expanded existing datasets (OPV2V+, DAIR-V2X, and CoPerception-UAVs+) to include more collaborative agents and demonstrated that the collaborative camera system outperforms LiDAR in some cases, achieving state-of-the-art performance on multiple benchmarks.

###  **IRV2V**  Asynchrony-Robust Collaborative Perception via Bird’s Eye View Flow  [[paper&review](https://openreview.net/forum?id=UHIDdtxmVS)] [~~code~~] [~~project~~]

- **Background and Motivation**  
The paper addresses the issue of temporal asynchrony in collaborative perception systems. Asynchronous communication among vehicles due to network delays, interruptions, or misalignments can cause significant issues in multi-agent collaboration. The paper proposes CoBEVFlow, a system designed to handle these asynchronies by using a bird’s-eye view (BEV) flow map to align asynchronous messages, ensuring more reliable collaborative perception in real-world autonomous driving scenarios.

- **Key Contributions**  
  - **CoBEVFlow Framework**: Introduces CoBEVFlow, an asynchrony-robust collaborative perception system that compensates for temporal misalignments in data exchanges.
  - **BEV Flow**: Proposes BEV flow to model and compensate for motion in the scene, allowing asynchronous features to be realigned accurately without generating new features, avoiding extra noise.
  - **IRV2V Dataset**: Creates the IRV2V dataset, the first synthetic dataset with various temporal asynchronies, simulating real-world scenarios to test the effectiveness of the proposed approach.
  - **Performance Validation**: Demonstrates that CoBEVFlow consistently outperforms existing methods under different latency conditions, improving detection performance by up to 30.3% compared to other state-of-the-art methods.
  - **Low Communication Cost**: CoBEVFlow is communication-efficient, transmitting only sparse features and ROI sets, reducing the overall communication bandwidth required for collaborative perception.


### V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception [[paper](https://arxiv.org/abs/2303.07601)] [[code](https://github.com/ucla-mobility/V2V4Real)] [[project](https://mobility-lab.seas.ucla.edu/v2v4real)]

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



### **Open Mars Dataset**  Multiagent Multitraversal Multimodal Self-Driving: Open MARS Dataset   [[code](https://github.com/ai4ce/MARS)] [[paper](https://arxiv.org/abs/2406.09383)] [[project](https://ai4ce.github.io/MARS)]

- **Background and Motivation**  
The paper introduces the MARS dataset, aimed at addressing the gap in existing datasets by incorporating multiagent and multitraversal elements. Traditional autonomous driving datasets often lack collaborative and repeated traversals, which limit advancements in perception, prediction, and planning. MARS was developed to fill this gap, enabling richer research in multiagent systems and enhanced 3D scene understanding.

- **Key Contributions**  
  - **Multiagent Collection**: MARS includes data from multiple autonomous vehicles operating in the same geographical region, enabling collaborative 3D perception.
  - **Multitraversal Data**: It captures multiple traversals of the same locations under various conditions, enhancing 3D scene understanding over time.
  - **Multimodal Sensor Setup**: The dataset features a full 360-degree sensor suite, including LiDAR and RGB cameras, for comprehensive scene analysis.
  - **Research Opportunities**: MARS opens new avenues for research in multiagent collaborative perception, unsupervised learning, and multitraversal 3D reconstruction.
  - **Real-World Data**: The dataset was collected using May Mobility's autonomous vehicles, ensuring high scalability and diversity across locations.


###  **OPV2V-H** An Extensible Framework for Open Heterogeneous Collaborative Perception  [[paper&review](https://openreview.net/forum?id=KkrDUGIASk)] [[code](https://github.com/yifanlu0227/HEAL)] [[project](https://huggingface.co/datasets/yifanlu/OPV2V-H)]

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


## :bookmark:V2I Datasets

- **V2I Datasets**: These datasets involve communication between vehicles and infrastructure, supporting cooperative tasks like object detection, tracking, and decision-making in connected environments.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| CoopInf | 2020 | TITS | L, C | Sim | 3DOD | [download](https://github.com/eduardohenriquearnold/coop-3dod-infra?tab=readme-ov-file) |
| CARTI        | 2022     | ITSC      | L           | Sim        | 3D OD      | -            |
| DAIR-V2X-C | 2022 | CVPR | L, C | Real | 3DOD | [download](https://air.tsinghua.edu.cn/DAIR-V2X/index.html) |
| V2X-Seq | 2023 | CVPR | L, C | Real | 3DOT, TP | [download](https://github.com/AIR-THU/DAIR-V2X-Seq) |
| HoloVIC | 2024 | CVPR | L, C | Real | 3DOD, MOT | [download](https://holovic.net) |
| OTVIC | 2024 | IROS | L, C | Real | 3DOD | [download](https://sites.google.com/view/otvic) |
| DAIR-V2XReid | 2024 | TITS | L, C | Real | 3DOD, Re-ID | [download](https://github.com/Niuyaqing/DAIR-V2XReid) |
| TUMTraf V2X | 2024 | CVPR | L, C | Real | 3DOD, MOT | [download](https://tum-traffic-dataset.github.io/tumtraf-v2x/) |
| V2X-DSI      | 2024     | IV        | L, C        | Sim        | 3D OD      | -            |
| V2X-Radar | 2024 | arxiv | L, C, R | Real | 3DOD | [download](http://openmpd.com/column/V2X-Radar) |



Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.


### **CoopInf**  Cooperative Perception for 3D Object Detection in Driving Scenarios Using Infrastructure Sensors  [[paper]()] [[code]()] [[project]()]

- **Background and Motivation**  
The paper addresses the challenge of 3D object detection in autonomous driving scenarios, particularly in complex environments like T-junctions and roundabouts. Traditional single-sensor systems face limitations such as occlusion and restricted field-of-view, hindering reliable detection. Cooperative perception using multiple spatially diverse infrastructure sensors provides an effective solution to mitigate these issues and improve detection performance.

- **Key Contributions**  
  - **Cooperative 3D Object Detection Schemes**: Two fusion schemes—early and late fusion—are proposed for cooperative perception. Early fusion combines raw sensor data before detection, while late fusion combines detected objects post-detection.
  - **Evaluation of Fusion Approaches**: The paper compares both fusion schemes and their hybrid combination in terms of detection performance and communication costs. Early fusion outperforms late fusion but requires higher bandwidth.
  - **Novel Cooperative Dataset**: A synthetic dataset for cooperative 3D object detection in driving scenarios, including a T-junction and roundabout, is introduced for performance evaluation.
  - **Impact of Sensor Configurations**: The study also evaluates how sensor number and positioning impact detection performance, offering practical insights for real-world deployment .


### **CARTI** PillarGrid: Deep Learning-Based Cooperative Perception for 3D Object Detection from Onboard-Roadside LiDAR

- **Background and Motivation**  
The paper introduces PillarGrid, a deep learning-based cooperative perception method for 3D object detection using both onboard and roadside LiDAR data. Traditional 3D object detection methods rely on single onboard LiDARs, which suffer from limitations in range and occlusion, especially in dense traffic. The motivation is to enhance detection accuracy and range by combining point cloud data from both vehicle-mounted and infrastructure-based sensors, improving detection performance in real-world scenarios.

- **Key Contributions**  
  - **PillarGrid Method**: Introduces a novel cooperative perception approach for 3D object detection that fuses data from onboard and roadside LiDAR sensors through deep learning.
  - **Grid-wise Feature Fusion (GFF)**: Proposes GFF, a feature-level fusion technique that combines information from multiple sensors to improve detection accuracy and reduce occlusion effects.
  - **Cooperative Preprocessing and Geo-Fencing**: Introduces cooperative preprocessing of point clouds and geo-fencing to align and process data from different sensors effectively.
  - **CNN-based 3D Object Detection**: Uses a convolutional neural network (CNN) to detect and generate 3D bounding boxes for objects, improving detection of vehicles and pedestrians.
  - **Dataset and Evaluation**: A new dataset, CARTI, was created using a cooperative perception platform for model training and evaluation, showing significant improvements over state-of-the-art methods in both accuracy and range.


### DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection  [[paper](https://arxiv.org/abs/2204.05575)] [[code](https://github.com/AIR-THU/DAIR-V2X)] [[project](https://thudair.baai.ac.cn/index)]

- **Background and Motivation**  
Autonomous driving still faces significant challenges, especially in terms of long-range perception and global awareness. While vehicle sensors have limitations, combining vehicle and infrastructure data can overcome these challenges. However, there was a lack of real-world datasets for vehicle-infrastructure cooperative problems. This paper introduces the DAIR-V2X dataset to facilitate research in this area.

- **Key Contributions**  
  - **First Vehicle-Infrastructure Cooperative Dataset**: DAIR-V2X is the first large-scale, multi-modality dataset captured from real-world scenarios for vehicle-infrastructure cooperation in 3D object detection.
  - **VIC3D Task Definition**: The paper defines the VIC3D task, which focuses on collaboratively detecting and identifying 3D objects using sensory inputs from both vehicle and infrastructure.
  - **VIC3D Benchmark and Fusion Framework**: It introduces benchmarks for VIC3D object detection and proposes the Time Compensation Late Fusion (TCLF) framework to handle temporal asynchrony.
  - **Real-World Data**: The dataset includes 71k frames of LiDAR and camera data, collected across various environments with 3D annotations.
  - **Performance Improvement**: The results demonstrate that integrating vehicle and infrastructure data leads to better performance than using single-source data.


### **DAIR-V2X-Seq**  V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting   [[paper](https://arxiv.org/abs/2305.05938)] [[code](https://github.com/AIR-THU/DAIR-V2X-Seq)] [[project](https://thudair.baai.ac.cn/index)]

- **Background and Motivation**  
The paper introduces the V2X-Seq dataset, addressing the need for real-world sequential datasets in vehicle-infrastructure cooperative perception and forecasting. Current research mainly focuses on improving perception using infrastructure data for frame-by-frame 3D detection, but lacks datasets for tracking and forecasting, which are critical for decision-making in autonomous driving.

- **Key Contributions**  
  - **Release of V2X-Seq**: The first large-scale, real-world sequential V2X dataset with data on vehicle and infrastructure cooperation.
  - **Two Main Parts**: Includes the sequential perception dataset with 15,000 frames from 95 scenarios, and the trajectory forecasting dataset with 210,000 scenarios.
  - **Three New Tasks**: Introduces VIC3D Tracking, Online-VIC Forecasting, and Offline-VIC Forecasting, with benchmarks to evaluate these tasks.
  - **Proposed FF-Tracking Method**: A middle fusion framework that efficiently solves the VIC3D tracking problem, handling latency challenges effectively.


### HoloVIC: Large-scale Dataset and Benchmark for Multi-Sensor Holographic Intersection and Vehicle-Infrastructure Cooperative [[paper](https://arxiv.org/abs/2403.02640)] [~~code~~] [[project](https://holovic.net)]

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


### TUMTraf-V2X: Cooperative Perception Dataset for 3D Object Detection in Driving Scenarios [[paper](https://arxiv.org/abs/2403.01316)] [[code](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)] [[project](https://tum-traffic-dataset.github.io/tumtraf-v2x)]

- **Background and Motivation**  
This paper presents the TUMTraf-V2X dataset, which aims to enhance vehicle perception through cooperative sensing. Roadside sensors and onboard sensors are used to overcome the limitations of single-sensor systems, especially occlusion and limited field of view. The dataset focuses on 3D object detection and tracking to improve road safety and autonomous driving capabilities.

- **Key Contributions**  
  - **High-Quality V2X Dataset**: The dataset includes 2,000 labeled point clouds and 5,000 labeled images with 30,000 3D bounding boxes. It covers challenging traffic scenarios, including near-miss events and traffic violations.
  - **Cooperative 3D Object Detection Model (CoopDet3D)**: A new cooperative fusion model, CoopDet3D, which outperforms traditional camera-LiDAR fusion methods with a +14.3 3D mAP improvement.
  - **Open-Source Tools and Resources**: The dataset and related tools, such as the 3D BAT labeling tool and a development kit, are made publicly available to facilitate integration and model training.
  - **Benchmarking and Evaluation**: Extensive experiments show that cooperative perception models lead to better detection accuracy than vehicle-only systems, proving the benefit of V2X collaboration in object detection tasks.


### V2X-DSI: A Density-Sensitive Infrastructure LiDAR Benchmark for Economic Vehicle-to-Everything Cooperative Perception

- **Background and Motivation**  
The paper addresses the high costs associated with the deployment of infrastructure LiDAR sensors in large-scale Vehicle-to-Everything (V2X) cooperative perception systems. It proposes a new benchmark, V2X-DSI, to explore the economic feasibility of using lower-beam infrastructure LiDAR sensors for cooperative perception. This is crucial as the deployment of high-beam LiDAR sensors on numerous infrastructures is prohibitively expensive, limiting the widespread adoption of V2X systems.

- **Key Contributions**  
  - **V2X-DSI Benchmark**: Introduces the first Density-Sensitive Infrastructure LiDAR benchmark, V2X-DSI, designed for economic V2X cooperative perception using LiDAR sensors with varying beam densities (16-beam, 32-beam, 64-beam, 128-beam).
  - **Performance Analysis**: Analyzes the impact of different beam densities on cooperative perception performance, using three state-of-the-art methods: OPV2V, V2X-ViT, and CoBEVT.
  - **Simulated Scenarios**: Utilizes a large-scale simulation in CARLA, including 56,984 frames from 57 diverse urban scenarios, to evaluate the performance of V2X cooperative perception systems.
  - **Fine-tuning for Low-Beam LiDAR**: Demonstrates that models trained on high-beam LiDAR can be fine-tuned to improve performance when deployed on low-beam LiDAR, mitigating the performance drop in real-world low-beam scenarios.
  - **Cost-Effective Deployment**: Provides a solution for reducing costs by using lower-beam LiDAR sensors without significantly compromising detection accuracy in urban traffic scenarios.


### V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception  [[paper](https://arxiv.org/abs/2411.10962)] [[code](https://github.com/yanglei18/V2X-Radar)] [[project](http://openmpd.com/column/V2X-Radar)]

- **Background and Motivation**  
The V2X-Radar dataset was developed to address the limitations in existing cooperative perception datasets, which often focus solely on camera and LiDAR data. The goal is to bridge the gap by incorporating 4D Radar, a sensor that excels in adverse weather conditions. The dataset aims to enhance the robustness of autonomous driving perception systems by addressing occlusions and limited range in single-vehicle systems.

- **Key Contributions**  
  - **First Real-World Multi-modal Dataset**: V2X-Radar is the first large, real-world dataset incorporating 4D Radar, alongside LiDAR and camera data.
  - **Comprehensive Data Coverage**: It includes 20K LiDAR frames, 40K camera images, and 20K 4D Radar data, with 350K annotated bounding boxes spanning five object categories.
  - **Three Specialized Sub-datasets**: The dataset is divided into V2X-Radar-C (cooperative perception), V2X-Radar-I (roadside perception), and V2X-Radar-V (single-vehicle perception).
  - **Extensive Benchmarking**: The dataset includes benchmarks for recent perception algorithms across these sub-datasets, supporting a wide range of research in cooperative perception.


## :bookmark:V2X Datasets
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

### **V2XSet**   V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer [[paper](https://arxiv.org/abs/2203.10638)] [[code](https://github.com/DerrickXuNu/v2x-vit)] [[project](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6)]

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


### V2X-Real: A Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception [[paper](https://arxiv.org/abs/2403.16034)] [~~code~~] [[project](https://mobility-lab.seas.ucla.edu/v2x-real)]

- **Background and Motivation**  
The paper addresses the limitations in current autonomous driving datasets, specifically in Vehicle-to-Everything (V2X) cooperative perception. Existing datasets primarily focus on Vehicle-to-Vehicle (V2V) or Vehicle-to-Infrastructure (V2I) collaboration, but real-world V2X datasets with both multi-vehicle and infrastructure collaboration are scarce. The need for real-world data to improve V2X perception systems, particularly for handling occlusions and expanding the perception range, is the central motivation.

- **Key Contributions**  
  - **Introduction of V2X-Real**: The first open, large-scale real-world dataset for V2X cooperative perception, featuring multi-modal sensing data from LiDAR and cameras.
  - **Four Sub-Datasets**: Divided into Vehicle-Centric, Infrastructure-Centric, Vehicle-to-Vehicle, and Infrastructure-to-Infrastructure cooperative perception datasets, tailored for different collaboration modes.
  - **Large-Scale Annotations**: Includes over 1.2 million annotated bounding boxes for 10 object categories, providing detailed data for multi-agent, multi-class cooperative 3D object detection tasks.
  - **Comprehensive Benchmarks**: Provides benchmarks for state-of-the-art cooperative perception methods, enhancing research on V2X interaction in complex urban environments.
  - **High-Density Traffic Data**: Collected in challenging urban environments with dense traffic, making it suitable for testing cooperative perception algorithms in real-world scenarios. 


### DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving [[paper](https://arxiv.org/abs/2304.01168)] [[code](https://github.com/tianqi-wang1996/DeepAccident)] [[project](https://deepaccident.github.io)]

- **Background and Motivation**  
The paper introduces DeepAccident, the first large-scale V2X dataset designed for motion and accident prediction in autonomous driving. While existing datasets mainly focus on perception tasks, they lack real-world accident scenarios and safety evaluations. The motivation is to provide a comprehensive dataset that includes safety-critical scenarios and supports end-to-end motion and accident prediction tasks for autonomous driving.

- **Key Contributions**  
  - **DeepAccident Dataset**: The first V2X dataset supporting motion and accident prediction, with 57K annotated frames and 285K annotated samples, offering a diverse range of accident scenarios.
  - **End-to-End Accident Prediction**: A novel task that predicts the occurrence, timing, location, and involved vehicles or pedestrians in accidents using raw sensor data.
  - **V2XFormer Model**: A baseline model demonstrating superior performance over single-vehicle models in motion and accident prediction, and 3D object detection tasks.
  - **V2X Research for Safety**: Enables V2X-based research in perception and prediction, addressing the gap in safety-critical scenario evaluation for autonomous driving algorithms.


### Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions[[paper](https://arxiv.org/abs/2410.06380)] [[code](https://github.com/QUARRG/Adver-City)] [[project](https://labs.cs.queensu.ca/quarrg/datasets/adver-city)]

- **Background and Motivation**  
Adverse weather conditions like rain, fog, and glare challenge the performance of Autonomous Vehicles (AVs) and Collaborative Perception (CP) models. The lack of datasets focusing on these conditions limits the evaluation and improvement of CP under such scenarios. Adver-City aims to address this gap by providing the first open-source CP dataset under diverse adverse weather conditions, including glare, a first for synthetic CP datasets.

- **Key Contributions**  
  - **First Open-Source CP Dataset for Adverse Weather**: Adver-City is the first synthetic, open-source CP dataset focused on weather conditions like rain, fog, and glare.
  - **Varied Weather and Object Density Scenarios**: The dataset includes 110 scenarios across six weather conditions and varying object densities to simulate real-world challenges in adverse weather.
  - **Multi-Sensor Data**: It includes data from LiDARs, RGB cameras, semantic segmentation cameras, GNSS, and IMUs, supporting tasks such as 3D object detection, tracking, and semantic segmentation.
  - **Realistic Scenario Design**: Scenarios are based on real crash reports, improving relevance for autonomous driving research.
  - **Benchmarking**: Performance benchmarks highlight the challenges posed by adverse weather on multi-modal object detection models, with performance drops observed in certain conditions.


### Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception [[paper](https://arxiv.org/abs/2409.04980)] [[code](https://github.com/RadetzkyLi/Multi-V2X)] [~~project~~]

- **Background and Motivation**  
The paper introduces the Multi-V2X dataset to address limitations in existing datasets for cooperative perception, particularly the lack of sufficient communicating agents and consideration of CAV penetration rates. Existing real-world datasets offer limited interaction, and synthetic datasets omit vulnerable road users like cyclists and pedestrians, essential for safe autonomous driving.

- **Key Contributions**  
  - **First Multi-Penetration-Rate Dataset**: Multi-V2X is the first dataset that supports varying CAV penetration rates, providing a realistic training environment for cooperative perception systems with up to 86.21% CAV penetration.
  - **Large-Scale Data**: The dataset includes 549k RGB frames, 146k LiDAR frames, and 4.2 million annotated 3D bounding boxes across six categories, supporting diverse training scenarios.
  - **Co-Simulation of SUMO and CARLA**: By co-simulating SUMO for traffic flow and CARLA for sensor simulation, the dataset captures realistic data for autonomous driving research.
  - **Comprehensive Benchmarks**: The dataset includes benchmarks for cooperative 3D object detection tasks, enabling the development of algorithms that perform under various penetration rates and cooperative settings.

### **DAIR-V2X-Traj**  Learning Cooperative Trajectory Representations for Motion Forecasting [[paper](https://arxiv.org/abs/2311.00371)] [[code](https://github.com/AIR-THU/V2X-Graph)] [[project](https://thudair.baai.ac.cn/index)]

- **Background and Motivation**  
The paper addresses the challenge of enhancing motion forecasting for autonomous driving by incorporating information from external sources, such as connected vehicles and infrastructure. Traditional methods focus on single-frame cooperative data, often underutilizing the rich motion and interaction context available from cooperative trajectories.

- **Key Contributions**  
  - **V2X-Graph Framework**: Introduces a novel graph-based framework for cooperative motion forecasting that fuses motion and interaction features from different sources, improving prediction accuracy.
  - **Forecasting-Oriented Representation Paradigm**: Proposes a new representation paradigm that uses cooperative trajectory data to enhance forecasting capabilities by considering motion and interaction features.
  - **V2X-Traj Dataset**: Develops the first real-world V2X dataset for motion forecasting, which includes both vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) cooperation in every scenario.
  - **State-of-the-Art Performance**: Demonstrates that the proposed method outperforms existing approaches on the V2X-Seq and V2X-Traj datasets, highlighting the effectiveness of cooperative data fusion in motion forecasting.


### WHALES: A Multi-Agent Scheduling Dataset for Enhanced Cooperation in Autonomous Driving [[paper](https://arxiv.org/abs/2411.13340)] [[code](https://github.com/chensiweiTHU/WHALES)] [[project](https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A)]

- **Background and Motivation**  
The paper presents the WHALES dataset, aiming to address the limitations in current cooperative perception datasets for autonomous driving. Existing datasets often involve a limited number of agents, reducing the effectiveness of multi-agent cooperation. WHALES tackles this by simulating environments with up to 8.4 agents per sequence, enhancing the scope for studying cooperative perception and agent scheduling.

- **Key Contributions**  
  - **Large-Scale Multi-Agent Dataset**: WHALES features 70K RGB images, 17K LiDAR frames, and 2.01M 3D bounding box annotations, capturing cooperative perception with multiple agents.
  - **Innovative Agent Scheduling**: Introduces the concept of agent scheduling in cooperative perception, a novel task not explored in previous datasets, optimizing cooperation for perception gains.
  - **Simulated Scenarios**: The dataset includes diverse road scenarios, including intersections and highway ramps, enabling more comprehensive research in cooperative driving.
  - **Benchmarking Tasks**: Provides benchmarks for 3D object detection and agent scheduling, supporting the development of algorithms for multi-agent cooperation in autonomous driving systems.


###  **V2X-R**  V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection [[paper](https://arxiv.org/abs/2411.08402)] [[code](https://github.com/ylwhxht/V2X-R)] [~~project~~]

- **Background and Motivation**  
The paper introduces the V2X-R dataset, a novel contribution to Vehicle-to-Everything (V2X) cooperative perception. Current datasets, which mainly focus on LiDAR and camera data, struggle under adverse weather conditions. The addition of 4D radar data, known for its weather robustness, aims to enhance 3D object detection in such challenging environments.

- **Key Contributions**  
  - **First V2X-R Dataset**: The first simulated V2X dataset that integrates LiDAR, camera, and 4D radar data, addressing weather robustness in cooperative perception.
  - **Cooperative LiDAR-4D Radar Fusion Pipeline**: Proposes a novel fusion pipeline that improves 3D object detection by combining data from multiple agents, LiDAR, and 4D radar sensors.
  - **Multi-modal Denoising Diffusion (MDD)**: Introduces a denoising diffusion module that uses 4D radar to guide the denoising process of noisy LiDAR data in adverse weather conditions, improving the detection accuracy.
  - **Comprehensive Benchmarking**: Establishes a benchmark using various fusion strategies and models, demonstrating the superior performance of the proposed approach in both normal and adverse weather conditions.


### V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction [[paper](https://arxiv.org/abs/2412.01812)] [[code](https://github.com/Zewei-Zhou/V2XPnP)] [[project](https://mobility-lab.seas.ucla.edu/v2xpnp)]

- **Background and Motivation**  
The paper addresses limitations in current Vehicle-to-Everything (V2X) cooperative perception systems that ignore temporal relationships across frames. While previous research focuses on single-frame perception, there is a lack of frameworks that handle temporal cues for perception and prediction. This work aims to integrate spatio-temporal information to improve cooperative perception and prediction in V2X environments, where vehicles and infrastructures share data to overcome occlusions and enhance safety.

- **Key Contributions**  
  - **V2XPnP Framework**: Introduces a novel spatio-temporal fusion framework using a Transformer architecture for end-to-end perception and prediction tasks, integrating temporal, spatial, and map information.
  - **First Real-World V2X Sequential Dataset**: Presents the V2XPnP Sequential Dataset, supporting all V2X collaboration modes (V2V, V2I, I2I), including 100 scenarios with diverse agent interactions and temporal consistency.
  - **Spatio-Temporal Fusion Strategies**: Proposes advanced fusion strategies—early, late, and intermediate fusion—incorporating temporal information for enhanced perception and prediction performance.
  - **Superior Performance**: Demonstrates that the V2XPnP framework significantly outperforms existing methods in both perception and prediction tasks, with a notable improvement in the EPA (End-to-End Perception and Prediction Accuracy) metric.


### Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration [[paper](https://arxiv.org/abs/2502.14156)] [[code](https://github.com/chinitaberrio/Mixed-Signals)] [[project](https://mixedsignalsdataset.cs.cornell.edu)]

- **Background and Motivation**  
The paper introduces the Mixed Signals dataset, designed to address the gaps in existing V2X datasets. While many datasets focus on homogeneous sensor setups from identical vehicles, the Mixed Signals dataset aims to capture the complexities of real-world V2X collaboration by including heterogeneous LiDAR configurations and left-hand driving scenarios, providing data for robust multi-agent perception tasks.

- **Key Contributions**  
  - **Heterogeneous LiDAR Configurations**: First V2X dataset featuring three vehicles with two different LiDAR sensors and a roadside unit with dual LiDARs, adding complexity to real-world V2X collaboration.
  - **Diverse Traffic Participants**: Includes 10 classes of objects, including 4 categories of vulnerable road users (VRUs), with over 240K 3D bounding box annotations.
  - **High-Quality Data Collection**: Precise synchronization and localization techniques for accurate sensor alignment, ensuring high-quality, annotated data that can be used for perception training and benchmarking.
  - **Real-World Scenario**: Captures data from real-world scenarios in Sydney, Australia, with left-hand traffic, increasing the dataset's applicability globally.
  - **Benchmarking and Evaluation**: Provides extensive benchmarks for collaborative object detection, enabling the development of advanced V2X perception methods.


### SCOPE: A Synthetic Multi-Modal Dataset for Collective Perception Including Physical-Correct Weather Conditions

- **Background and Motivation**  
The paper introduces the SCOPE dataset, designed to address gaps in current datasets for collective perception (CP) in autonomous driving. Existing datasets lack scenario diversity, realistic sensor models, and environmental conditions like adverse weather. SCOPE is created to support the development and testing of collective perception algorithms, especially in challenging real-world conditions.

- **Key Contributions**  
  - **First Synthetic Multi-Modal CP Dataset**: SCOPE is the first synthetic dataset that combines realistic LiDAR and camera models with physically-accurate weather simulations for both sensors.
  - **Diverse Scenarios**: The dataset includes 17,600 frames from over 40 different scenarios, including edge cases like tunnels and roundabouts, with up to 24 collaborative agents.
  - **Weather Simulations**: SCOPE introduces realistic rain and fog simulations with parameterized intensities, improving robustness against environmental effects on perception.
  - **Wide Sensor Setup**: The dataset uses a variety of sensors, including RGB and SemSeg cameras, along with three different LiDAR models, to enable comprehensive object detection and semantic segmentation tasks.
  - **Novel Maps**: It includes two novel digital-twin maps from Karlsruhe and Tübingen, enhancing the dataset’s real-world applicability.


### **OPV2V-N**  RCDN: Towards Robust Camera-Insensitivity Collaborative Perception via Dynamic Feature-based 3D Neural Modeling [[paper](https://arxiv.org/abs/2405.16868)] [~~code~~] [~~project~~]

- **Background and Motivation**  
The paper introduces RCDN, a solution to overcome the issue of noisy or failed camera perspectives in multi-agent collaborative perception. In real-world settings, cameras can be blurred, noisy, or even fail, severely affecting the performance of collaborative perception systems. RCDN aims to recover perceptual messages from failed camera perspectives using dynamic feature-based 3D neural modeling, ensuring robust collaborative performance with low calibration costs.

- **Key Contributions**  
  - **Introduction of RCDN**: A robust camera-insensitivity collaborative perception system that uses dynamic feature-based 3D neural modeling to recover failed perceptual information.
  - **Collaborative Neural Rendering**: RCDN constructs collaborative neural rendering field representations to stabilize high collaborative performance even in the presence of noisy or failed cameras.
  - **Two Collaborative Fields**: Introduces time-invariant static and time-varying dynamic fields for collaborative perception, enhancing the system’s ability to handle camera failures and recover from noisy inputs.
  - **New Dataset - OPV2V-N**: Introduces OPV2V-N, a large-scale dataset with manually labeled data, simulating various camera failure scenarios for better research on camera-insensitive collaborative perception.
  - **Improved Robustness**: Demonstrates that RCDN significantly enhances the performance of existing collaborative perception methods, improving their robustness by up to 157.91% under extreme camera-insensitivity conditions.


### **CP-GuardBench**  CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception [[paper&review](https://openreview.net/forum?id=9MNzHTSDgh)] [~~code~~] [~~project~~]

- **Background and Motivation**  
The paper introduces CP-Guard+, a solution for enhancing the security of collaborative perception (CP) systems in autonomous driving. While CP systems enable vehicles to share sensory data for improved perception, they are vulnerable to malicious agents that may inject adversarial data, potentially compromising safety. The motivation behind CP-Guard+ is to create a robust, computationally efficient framework that can detect and defend against such malicious agents, mitigating risks in autonomous driving environments.

- **Key Contributions**  
  - **Feature-Level Malicious Agent Detection**: Proposes a novel approach for detecting malicious agents directly at the feature level, eliminating the need for computationally expensive hypotheses and verifications.
  - **CP-GuardBench Dataset**: Introduces the first benchmark dataset, CP-GuardBench, designed specifically for training and evaluating malicious agent detection methods in CP systems.
  - **CP-Guard+ Defense Framework**: Develops CP-Guard+, a defense mechanism that improves malicious agent detection by enhancing the separability of benign and malicious features using a Dual-Centered Contrastive Loss (DCCLoss).
  - **Superior Performance**: Demonstrates through extensive experiments on both CP-GuardBench and V2X-Sim datasets that CP-Guard+ outperforms traditional defense methods, achieving high accuracy and low false positive rates while significantly reducing computational overhead.
  - **Efficiency and Scalability**: CP-Guard+ achieves a substantial increase in frames per second (FPS) compared to existing methods, proving its efficiency in real-time CP systems.


### V2X-ReaLO: An Open Online Framework and Dataset for Cooperative Perception in Reality [[paper](https://arxiv.org/abs/2503.10034)] [~~code~~] [~~project~~]

- **Background and Motivation**  
The paper addresses the challenges of applying Vehicle-to-Everything (V2X) cooperative perception in real-world conditions. Previous research often focuses on simulations or static datasets, which fail to capture dynamic, real-time conditions such as communication latency and sensor misalignment. V2X-ReaLO aims to bridge this gap by providing a practical, real-world framework that demonstrates the feasibility of real-time intermediate fusion.

- **Key Contributions**  
  - **V2X-ReaLO Framework**: Introduces an open online framework for cooperative perception deployed on real vehicles and smart infrastructure, integrating early, late, and intermediate fusion methods.
  - **First Practical Demonstration**: Provides the first real-world demonstration of the feasibility and performance of intermediate fusion under real-world conditions.
  - **Open Online Benchmark Dataset**: Extends the V2X-Real dataset to dynamic, synchronized ROS bags with 25,028 frames, including 6,850 annotated key frames for real-time evaluation in challenging urban scenarios.
  - **Real-Time Evaluation**: Supports online evaluations, accounting for real-world challenges such as bandwidth limitations, latency, and asynchronous message arrival.
  - **Comprehensive Benchmarking**: Conducts extensive benchmarks to assess multi-class, multi-agent V2X cooperative perception performance, highlighting the system's effectiveness in various collaboration modes (V2V, V2I, and I2I).


## :bookmark:I2I Datasets
- **I2I Datasets**: Focused on infrastructure-to-infrastructure collaboration, these datasets support research in scenarios with overlapping sensor coverage or distributed sensor fusion across intersections.

### Table

| **Dataset** | **Year** | **Venue** | **Sensors** | **Source** | **Tasks** | **download** |
|-------------|----------|-----------|-------------|------------|-----------|----------|
| Rcooper | 2024 | CVPR | C, L | Real | 3DOD, MOT | [download](https://github.com/AIR-THU/DAIR-Rcooper) |
| InScope | 2024 | arxiv | L | Real | 3DOD, MOT | [download](https://github.com/xf-zh/InScope) |

Note: Sensors: Camera (C), LiDAR (L), Radar (R). Source: Real = collected in the real world; Sim = generated via simulation. Tasks: 2DOD = 2D Object Detection, 3DOD = 3D Object Detection, MOT = Multi-Object Tracking, MTSCT = Multi-target Single-camera Tracking, MTMCT = Multi-target Multi-camera Tracking, SS = Semantic Segmentation, TP = Trajectory Prediction, VPR = Visual Place Recognition, NR = Neural Reconstruction, Re-ID = Re-Identification, S2R = Sim2Real, MF = Motion Forecasting, PQA = Planning Q&A.


### RCooper: A Real-world Large-scale Dataset for Roadside Cooperative Perception  [[paper](https://arxiv.org/abs/2403.10145)] [[code](https://github.com/AIR-THU/DAIR-RCooper)] [[project](https://www.t3caic.com/qingzhen)]

- **Background and Motivation**  
The paper introduces RCooper, the first real-world, large-scale dataset for roadside cooperative perception. Existing roadside perception systems focus on independent sensors, leading to limited sensing range and blind spots. Roadside cooperative perception (RCooper) aims to enhance traffic monitoring and autonomous driving by using data from multiple roadside sensors to overcome these limitations, providing more comprehensive coverage.

- **Key Contributions**  
  - **First Real-World RCooper Dataset**: RCooper is the first large-scale dataset dedicated to roadside cooperative perception, including 50k images and 30k point clouds with manual annotations.
  - **Two Main Traffic Scenes**: The dataset includes two representative traffic scenes: intersections and corridors, capturing diverse traffic flow and environmental conditions.
  - **Challenges in Roadside Perception**: The dataset addresses challenges such as data heterogeneity, sensor alignment, and cooperative representation for roadside systems.
  - **Cooperative Detection and Tracking Tasks**: RCooper provides benchmarks for two tasks—3D object detection and tracking—using multi-agent cooperation, with state-of-the-art methods included for comparison.
  - **Comprehensive Annotation and Data Diversity**: The dataset is annotated with 3D bounding boxes and trajectories across ten semantic classes and includes variations in weather and lighting conditions.


### InScope: A New Real-world 3D Infrastructure-side Collaborative Perception Dataset for Open Traffic Scenarios  [[paper](https://arxiv.org/abs/2407.21581)] [[code](https://github.com/xf-zh/InScope)] [~~project~~]

- **Background and Motivation**  
The paper introduces the InScope dataset, aiming to address the issue of occlusion in vehicle-centric perception systems. Infrastructure-side perception systems (IPS) are suggested to complement autonomous vehicles, providing broader coverage. However, the lack of real-world 3D infrastructure-side datasets limits the progress in V2X technologies. InScope aims to bridge this gap by capturing occlusion challenges and providing collaborative perception data.

- **Key Contributions**  
  - **First Large-Scale Infrastructure-Side Collaborative Dataset**: InScope is the first 3D infrastructure-side dataset designed to handle occlusion challenges with multi-position LiDARs.
  - **Comprehensive Data Collection**: Includes 303 tracking trajectories and 187,787 3D bounding box annotations captured over 20 days in open traffic scenarios.
  - **Four Key Benchmarks**: The dataset provides benchmarks for 3D object detection, multi-source data fusion, data domain transfer, and 3D multi-object tracking tasks.
  - **Anti-Occlusion Evaluation Metric**: Introduces a new metric (𝜉𝐷) to evaluate the anti-occlusion capabilities of detection methods, quantifying detection degradation ratios between single and multi-LiDAR setups.
  - **Enhanced Perception**: The dataset significantly enhances performance in detecting and tracking occluded, small, and distant objects, which are critical for real-world traffic safety.

## 其他协同感知数据集论文
- **Griffin** (Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark) [[paper](https://arxiv.org/abs/2503.06983)] [[code](https://github.com/wang-jh18-SVM/Griffin)] [[project](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g)]
- **RLS** (Analyzing Infrastructure LiDAR Placement with Realistic LiDAR Simulation Library) [[paper](https://arxiv.org/abs/2211.15975)] [[code](https://github.com/PJLab-ADG/LiDARSimLib-and-Placement-Evaluation)] [~~project~~]
- **Roadside-Opt** (Optimizing the Placement of Roadside LiDARs for Autonomous Driving) [[paper](https://arxiv.org/abs/2310.07247)] [~~code~~] [~~project~~]
- {Real} **DAIR-V2X-C Complemented** (Robust Collaborative 3D Object Detection in Presence of Pose Errors) [[paper](https://arxiv.org/abs/2211.07214)] [[code](https://github.com/yifanlu0227/CoAlign)] [[project](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented)]
- **V2XP-ASG** (V2XP-ASG: Generating Adversarial Scenes for Vehicle-to-Everything Perception) [[paper](https://arxiv.org/abs/2209.13679)] [[code](https://github.com/XHwind/V2XP-ASG)] [~~project~~]
- **AutoCastSim** (COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles) [[paper](https://arxiv.org/abs/2205.02222)] [[code](https://github.com/hangqiu/AutoCastSim)] [[project](https://utexas.app.box.com/v/coopernaut-dataset)]
- **Mono3DVLT-V2X** (Mono3DVLT: Monocular-Video-Based 3D Visual Language Tracking) [~~paper~~] [~~code~~] [~~project~~]
- **RCP-Bench** (RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions) [~~paper~~] [~~code~~] [~~project~~]





## Methods





