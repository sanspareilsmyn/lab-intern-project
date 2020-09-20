# lab-intern-project
Undergraduate Research Internship - Intern project (2018.12 - 2019.03)

## Introduction of Project 
1. Title :  Implementation of Deep-Learning-Based Active User Detection(AUD) Layer

2. Goal of project
- Understanding MTC System Model and Spreading sequence
- Implementation of AUD Layer

3. Reference
- E. Dahlman et al., <5G wireless access: Requirements and realization" in IEEE Communication Magazine, vol. 52, no.12, Dec. 2014, pp.42-47 
- B. Xin et al., <Maximal sparsity with deep networks?> in Proc. Nerual Inf. Process. Syst., Dec. 2016, pp 1-9

## Project Detail
1. Understanding MTC System Model and Spreading sequence
N개의 Machine-type Communication device가 있다고 가정하자. 이 장치들이 동기화되어 one signal at a time으로 Transmitter에서 송출된다면 이는 N차원의 single signal sparse vector를 형성한다. Transmitter에서 이 signal vector를 user-specific spreading sequence s(m차원)과 합성해서 송출한다. 이 때 모든 Device에 대한 spreading sequence를 모은 spreading matrix S는 MxN차원이다.
System Model은 다음과 같이 모델링된다.

                                                              Y=SX+W
이 때 W는 Channel에서 발생하는 Gaussian Noise이다.

2. Implementation of Spreading Layer / Channel Layer / AUD Layer

![image](https://user-images.githubusercontent.com/52681837/93707988-ce567100-fb6d-11ea-831b-a15d4527537d.png)

다음과 같은 Transmitter - Channel - Receiver 구조에서 아래와 같이 AUD Layer를 모델링한다. 
사용한 모델은 IHT(Iterative Hard Thresholding)-net을 간략화한 형태이다. 원 모델의 IHT를 IHT-Net으로 개조했는데, Thresholding은 ReLU로 대체했고 Iteration에서의 weight는 Training 가능하게 바꾼 형태이다. 

![image](https://user-images.githubusercontent.com/52681837/93708977-a8cd6580-fb75-11ea-9f44-f00bf531b600.png)

