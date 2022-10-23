# ADHD Children CARE Solution

### Purpose of Project
가정환경에서 아동을 모니터링하며 ADHD의 징후로 의심되는 행동의 빈도를 체크하고 
보호자에게 알림을 주는 ‘아동 ADHD 모니터링 로봇’ 제작 프로젝트.

### Sequence
1. 아동으로부터 거부감이 들지 않고 안전한 형태의 로봇 하드웨어 설계.
2. 로봇의 카메라로 아동을 촬영한 영상을 Pose Estimation으로 행동을 분류.
3. 분류된 행동이 ADHD 진단 기준인 DSM-5 항목에 해당되는 행동 특성인지 확인.
    - 대표적인 ADHD 증상으로는 과잉행동, 반복행동, 행동 전환성 등이 있음.
    - 해당 프로젝트에서는 Pose Estimation을 통해 갑작스런 달리기(과잉행동), 의미없는 손 움직임 반복(반복행동), 가만히 있지 못하고 앉았다 일어서기를 반복하는 행동(행동 전환성)을 확인하고자 하였음.
4. 진단기준의 징후들을 바탕으로 행동 시작 시간, 빈도, 지속 시간 등을 기록.
5. ADHD 의심 행동으로 기록된 로그를 이용하여 보호자에게 행동분석 리포트를 제공.

### Data Preprocessing
1. Features Engineering
- Pose Estimation으로 추출된 Human 2d Skeleton X and Y axis coordinate 정보를 Face/Left arm/Right arm/body 로 나누어
1662개의 feature를 생성
- X and Y axis coordinate 정보로 X Y 사이의 angle feature를 생성할 예정

### Modeling
1. RNN - Linear 계층 테스트 진행중

### Referece
1. https://github.com/felixchenfy/Realtime-Action-Recognition
2. https://github.com/shreyas-jk/Baby-Action-Detection-for-Safety-System-Prototype
3. https://github.com/nicknochnack/ActionDetectionforSignLanguage
4. Lots of references are left, I'm going to add soon.