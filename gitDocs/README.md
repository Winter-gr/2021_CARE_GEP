# 2021_CARE_GEP_EDUBOT
한양대학교ERICA 교내대회 GEP에 참가하는 stayc 팀 레포지토리 입니다.


# git 사용법 간략하게 정리

-origin  : 원격 저장소 브랜치  
-master  : 로컬 저장소 브랜치(main 일 때도 있으니 주의)  
-(new)   : 이슈별로 업데이트 하면서 만드는 브랜치  

# git 에 업데이트하는 방법
* 원격저장소의 변경내용을 로컬에 가져온다.  
`git pull origin (로컬브랜치 이름 master나 main)`
* 로컬에서 새로운 브랜치를 만든다.  
`git branch (새 브랜치 이름)`  
    삭제 하는 방법 : `git branch -d (삭제할 브랜치 이름)`  
    브랜치 확인하는 방법 : `git branch -a`
* 새로운 브랜치에 업데이트 할 내용을 add, commit 한다.  
`git checkout (옮겨갈 브랜치 이름)`
    `git add (업데이트 할 파일)`
    `git commit -m "(커밋 내용)"`
* push 하고 githib 원격저장소에서 PR을 남긴다.  
`git push origin (커밋 올려둔 브랜치)`  
그리고 github 홈페이지에 들어가면 새로 생긴 PR 버튼을 누른다.  
자잘한 설정이나 설명등을 추가할 수도 있는데, 충돌이 나면 일단 @Winter-gr을 부르세욧  
**Merge 까지 끝내면 다음으로 넘어갈 수 있다**
* 다시 원격저장소의 변경내용을 로컬에 가져온다.
`git pull origin (로컬브랜치 이름 master나 main)`
* 사용했던 브랜치를 삭제한다.  
`git checkout (로컬브랜치 이름 master나 main)`
`git branch -D (다 사용한 브랜치)`
