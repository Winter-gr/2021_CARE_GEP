# 2021_CARE_GEP_EDUBOT
한양대학교ERICA 교내대회 GEP에 참가하는 stayc 팀 레포지토리 입니다.





## git 사용법 간략하게 정리

> !!필수적인 내용 요약!!
>
> `git pull origin master` > 코드 수정 >`git add *`>`git commit -m “무엇이 바뀌었는지 설명작성”`>`git push origin master`

### 개인 노트북에서 사용할 디렉토리 안에서 초기화, 셋팅할 때 
1. 먼저 git 로그인. Git 없다면 터미널에서 Git 다운로드 하고 로컬에서 로그인 하기
2. `git init` 레포지토리를 연결할 로컬 저장소에서
3. `git remote add origin (연결할 레포지포리 HTTP주소)` 레포지토리 연결하기

### 상태 확인하기
현재 브랜치, 저장된 커밋(버전), add 된 파일, 변경된 파일 확인 `git status`

### 로컬에서 코드 수정하고 업데이트 할 때 
1. `git pull origin master` 레포지토리에 업데이트 된 코드를 로컬에도 업데이트 
2.  `git add .` 수정후 원하는 파일을 선택 
    * `.`은 수정된 전체 파일인 경우이고, `.`가 아닌 파일명이나 폴더명을 적을 수 있다. 
    * 띄어쓰기로 여러 파일을 구분해서 한 번에 add 할 수 있다. `Ex)$ git add README test.rb LICENSE`

3.  `git commit -m “무엇이 바뀌었는지 설명작성”` 로컬 저장소에 커밋하기 (버전 기록하기)
    * (-m 은 messege의 m)
4. `git push origin master` 원격 저장소에도 버전 저장하기 
    * ‘Master’ = 로컬저장소 브랜치 이름,  ‘origin’= 원격저장소 브랜치 이름
