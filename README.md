# 시옷(SeeO-K) : 시각장애인을 위한 옷 종류, 색, 패턴 판별 및 코디 추천 시스템 

<img width="619" alt="title" src="https://user-images.githubusercontent.com/28584275/132673765-dd23f735-b06f-44d8-9cba-ac8441f7ff54.png">

## 개발 환경

* **S/W 개발환경** : Python, Ubuntu 18.04 LTS

* **S/W 개발환경도구** : PyCharm

* **H/W 개발환경** : Jetson Nano

* **H/W** : 웹캠, 스피커, 마이크

  

## 시옷(SeeO-K) 초기 설정

### 1. H/W 초기 설정

1.1 NVIDIA Jetson Nano, 웹캠, 스피커, 마이크를 준비한다. 

1.2 웹캠, 스피커, 마이크를 Jetson Nano에 연결 후, 카메라를 적절한 위치에 세팅.

1.3 Jetson Nano 인터넷 연결 필요



### 2.S/W 초기 설정

#### 2.1 코드 다운받기

```python
git clone https://github.com/2do1/SeeO-K.git
```

#### 2.2 Jetson Nano OS, JetPack 설치

2.2.1 https://developer.nvidia.com/embedded/downloads에서 **Jetson Nano Developer Kit SD Card Image** 다운

2.2.2 Micro SD 메모리카드에 다운받은 이미지를 Write 한 후, Jetson Nano 구동.

#### 2.3 필요한 라이브러리 설치

* Python3 pip3

  ```python
  sudo apt install python3-pip
  ```

* Numpy

  ```python
  pip3 install numpy
  ```

* Tensorflow

  ```	python
  sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
  ```

  제일 최신 버전 설치하는 명령어

  주의할점 ! : Jetson Nano OS, JetPack 버전에 맞는 tensorflow 버전을 설치 해야함

  https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel 링크에서 Jetpack 버전에 맞는 tensorflow 버전 확인.

* OpenCV

  https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html 링크 참조

* PyTorch, TorchVision

  https://qengineering.eu/install-pytorch-on-jetson-nano.html 링크 참조

* PyMySQL

  ```python
  pip3 install pymysql
  ```

* Scikit-Learn

  ```python
  sudo -H pip3 install scikit-learn
  ```

* Gtts

  ```python
  pip3 install gtts
  ```

* Pillow

  ```python
  pip3 install pillow
  ```

* PyAudio

  ``` python
  pip3 install pyaudio
  ```

* PlaySound

  ``` python
  pip3 install playsound
  ```

* SpeechRecognition

  ``` python
  pip3 install SpeechRecognition
  ```

* Requests

  ```python
  pip3 install requests
  ```

* BeautifulSoup4

  ``` python
  pip3 install beautifulsoup4
  ```

#### 2.4 자신만의 DB(옷장) 생성

2.4.1 사용자가 데이터베이스 관리자에게 IP 주소를 준다.

2.4.2 구글  클라우드 플랫폼을 활용하여,  관리자는 사용자에게 받은 IP 주소를 등록하여, 사용자마다의 데이터베이스를 생성해주고 관리한다.



## 프로젝트 구조, 작동 방식

### 1. FlowChart

![flow](https://user-images.githubusercontent.com/28584275/132673835-c019fbd3-ed7c-4409-807b-1e1439c64252.png)


### 2. 서비스 아키텍처

![arch](https://user-images.githubusercontent.com/28584275/132673894-21262e8a-b256-4c7b-8c52-e7e4fe70d95f.png)

## 시옷 시연 영상

https://www.youtube.com/watch?v=HJoyewyEHHY

## 팀원

이도원, 노성환, 이한정, 김하연, 송수인
