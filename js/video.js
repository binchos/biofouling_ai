let isExtracting = false;         // 프레임 추출 중인지 여부
let extractIntervalId = null;     // setInterval ID
let lastCapturedTime = 0;         // 마지막 추출된 시점
let framesData = [];

const framesContainer = document.querySelector('.frames');
const frames = [];

// ✅ 프레임 하나를 추가하는 함수
function addFrameSet(originalSrc, mSegSrc, sSegSrc) {
  const frameDiv = document.createElement('div');
  frameDiv.classList.add('frame');

  // 1️⃣ 원본 이미지 행
  const originalRow = document.createElement('div');
  originalRow.classList.add('frame-original-row');
  const originalImg = document.createElement('img');
  originalImg.src = originalSrc;
  originalImg.classList.add('frame-original');
  originalRow.appendChild(originalImg);

  // 2️⃣ M seg / S seg 행
  const segRow = document.createElement('div');
  segRow.classList.add('frame-seg-row');

  // M seg
  const mBox = document.createElement('div');
  mBox.classList.add('seg-box');
  const mImg = document.createElement('img');
  mImg.src = mSegSrc;
  mImg.classList.add('frame-seg');
  const mLabel = document.createElement('div');
  mLabel.classList.add('frame-label');
  mLabel.textContent = 'M seg';
  mBox.appendChild(mImg);
  mBox.appendChild(mLabel);

  // S seg
  const sBox = document.createElement('div');
  sBox.classList.add('seg-box');
  const sImg = document.createElement('img');
  sImg.src = sSegSrc;
  sImg.classList.add('frame-seg');
  const sLabel = document.createElement('div');
  sLabel.classList.add('frame-label');
  sLabel.textContent = 'S seg';
  sBox.appendChild(sImg);
  sBox.appendChild(sLabel);

  // 행 구성
  segRow.appendChild(mBox);
  segRow.appendChild(sBox);

  // 프레임 완성
  frameDiv.appendChild(originalRow);
  frameDiv.appendChild(segRow);

  // ✅ 클릭 시 선택/해제 토글
  frameDiv.addEventListener('click', () => {
    frameDiv.classList.toggle('selected');
  });

  framesContainer.appendChild(frameDiv);
}

//영상 업로드 함수
function handleVideoUpload(event) {
  const file = event.target.files[0];
  if(!file) return;

  const videoPreview = document.getElementById('preview');
  const sourceTag = videoPreview.querySelector('source');
  if (sourceTag.src) {
    URL.revokeObjectURL(sourceTag.src);
  }

  const videoURL = URL.createObjectURL(file);

  // source에 경로 지정 + 타입 설정
  sourceTag.src = videoURL;
  sourceTag.type = file.type;
  videoPreview.load(); // 새 소스 로드
}

//영상 재생 함수
function playVideo(event){
  const video = document.getElementById('preview');
  // ✅ src가 없으면 재생 안 시도
  if (!video || !video.currentSrc) {
    console.warn("영상이 없습니다.");
    return;
  }
  // ✅ 재생 중이면 중복 재생 시도 방지
  if (video.paused || video.ended) {
    video.play().then(() => {
      startExtractFrames(video);// 영상 재생 시작 시 프레임 추출 시작
    }).catch(err => {
      console.warn("재생 실패:", err);
    });
  } else {
    console.log("이미 재생 중입니다.");
  }
}

//영상 정지 함수
function stopVideo(event){
  const video = document.getElementById('preview');
  if(video){
    video.pause();
    stopExtractFrames();
  }
}

//이미지 프레임 추가
function addFrame(blobUrl, metadata){
  const framesContainer = document.querySelector('.frames');
  const frameDiv = document.createElement('div');
  frameDiv.className = 'frame';

  frameDiv.innerHTML = `
    <button class="frame-close">×</button>
    <div class="frame-original-row">
        <img src="${blobUrl}" class="frame-original" alt="원본 프레임">
    </div>
    <div class="frame-seg-row">
    <div class="seg-box">
        <img src="mseg/frame_m.png" class="frame-seg" alt="M seg">
        <div class="frame-label">M seg</div>
    </div>
    <div class="seg-box">
        <img src="sseg/frame$_s.png" class="frame-seg" alt="S seg">
        <div class="frame-label">S seg</div>
    </div>
    </div>
  `;

  frameDiv.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    openframeModal(frameDiv);
  });

  framesContainer.appendChild(frameDiv);
}
//삭제 모드(삭제 클릭 시 x버튼 나오게)
function toggleDeleteMode() {
  const deleteButton = document.querySelector('.delete');
  const frames = document.querySelectorAll('.frame');
  if (frames.length === 0) return;

  frames.forEach(frame => {
    frame.classList.toggle('show-close');
  });

  if (deleteButton) {
    const isActive = deleteButton.classList.toggle('active');
    deleteButton.textContent = isActive ? '취소' : '삭제';
  }
}
//프레임 삭제 함수
function deleteFrame(event) {
  // ✅ 삭제 버튼 내부 어떤 걸 클릭하든 상관없이 버튼 자체로 인식되도록
  const closeBtn = event.target.closest('.frame-close');
  if (closeBtn) {
    event.stopPropagation();   // ✅ 프레임 클릭 이벤트 막기 (이게 핵심)
    const frame = closeBtn.closest('.frame');
    if (frame) frame.remove();

    // ✅ 모든 프레임 삭제 시 자동으로 버튼/모드 초기화
    const remainingFrames = document.querySelectorAll('.frame').length;
    const deleteButton = document.querySelector('.delete');

    if (remainingFrames === 0 && deleteButton) {
      deleteButton.classList.remove('active');
      deleteButton.textContent = '삭제';

      // 모든 show-close 제거 (혹시 남아있을 수도 있음)
      document.querySelectorAll('.frame').forEach(f => {
        f.classList.remove('show-close');
      });
    }
  }
}
//삭제 버튼 이벤트 연결 함수
function setupDeleteFeature() {
  const deleteButton = document.querySelector('.delete');
  if (deleteButton) {
    deleteButton.addEventListener('click', toggleDeleteMode);
  }
  document.addEventListener('click', deleteFrame);
}

//프레임 추출 함수
function startExtractFrames(video, interval = 1000){
  if (isExtracting) return; // 중복 실행 방지
  isExtracting = true;

  extractIntervalId = setInterval(() => {
    if (video.paused || video.ended) {
      stopExtractFrames();  // 영상 멈추면 추출도 멈춤
      return;
    }
    const currentTime = video.currentTime;
    // 중복 추출 방지 (0.5초 이상 차이 날 때만)
    if (Math.abs(currentTime - lastCapturedTime) > 0.5) {
      captureCurrentFrame(video, currentTime);
      lastCapturedTime = currentTime;
    }

  }, interval); // e.g. 1000ms마다 추출 시도
}

//프레임 추출 정지 함수
function stopExtractFrames(){
  if (extractIntervalId){
    clearInterval(extractIntervalId);
    extractIntervalId = null;
    isExtracting = false;
  }
}

//현재 프레임을 캡처하는 로직(blob + 메타 데이터 저장)
function captureCurrentFrame(videoElement, time) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    const frameURL = URL.createObjectURL(blob);
    addFrame(frameURL); // DOM에 렌더링
    framesData.push({
      id: `frame_${framesData.length}`,
      timestamp: time.toFixed(2),
      blob: blob
    });
  }, 'image/jpeg');
}

//전체 삭제 버튼 클릭시 모달 창 활성화 함수
function setupAllDeleteModal({
  buttonSelector,
  modalSelector,
  confirmSelector,
  cancelSelector,
  targetSelector
}){
  const allDeleteBtn = document.querySelector(buttonSelector);
  const modal = document.querySelector(modalSelector);
  const confirmBtn = document.querySelector(confirmSelector);
  const cancelBtn = document.querySelector(cancelSelector);
  const targetContainer = document.querySelector(targetSelector);

  if (!allDeleteBtn || !modal || !confirmBtn || !cancelBtn || !targetContainer) {
    console.error("❌ 모달 삭제 설정 실패: 선택자 오류");
    return;
  }

  //전체 삭제 버튼 클릭 이벤트
  allDeleteBtn.addEventListener('click', () => {
    const frameCount = targetContainer.querySelectorAll('.frame').length;
    if (frameCount == 0){
      console.log("삭제할 프레임이 없습니다. 모달을 열지 않음.");
      return;
    }
    modal.style.display = 'flex';
  });

  confirmBtn.addEventListener('click', () => {
    targetContainer.innerHTML = '';
    modal.style.display = 'none';
  });

  // 취소 버튼: 모달 닫기
  cancelBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
}

//frame 모달 창 열기
function openframeModal(frameElement){
  const modal = document.getElementById('frameModal');
  const modalImg = document.getElementById('frame-modalPreviewImage');
  const sValue = document.getElementById('modalSValue');
  const mValue = document.getElementById('modalMValue');

  //모달 이미지에 frame의 원본 이미지 넣기
  const originalImg = frameElement.querySelector('.frame-original');
  if (originalImg) {
    modalImg.src = originalImg.src;
  }
  //임시 설정
  sValue.textContent = '23.5';
  mValue.textContent = '12.7';

  modal.style.display = 'flex';
}
//frame 모달 닫기 버튼 함수
function closeframeModal(){
  const closeBtn = document.getElementById('modalClose');
  const modal = document.getElementById('frameModal');

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
}
function setupFinalAnalyzeModal() {
  // 1️⃣ 열기 함수
  function openFinalAnalyzeModal() {
    const modal = document.getElementById('finalanalyzeModal');
    if (modal) {
      modal.style.display = 'flex';
    }
  }
  // 2️⃣ 닫기 함수
  function closeFinalAnalyzeModal() {
    const modal = document.getElementById('finalanalyzeModal');
    if (modal) {
      modal.style.display = 'none';
    }
  }
  const analyzeBtn = document.querySelector('.analyze');
  const closeBtn = document.getElementById('finalanalyzeModalClose');

  if (analyzeBtn) analyzeBtn.addEventListener('click', openFinalAnalyzeModal);
  if (closeBtn) closeBtn.addEventListener('click', closeFinalAnalyzeModal);
}

function navigateTo(page) {
  const currentPath = window.location.pathname;
  if (!currentPath.endsWith(page)) {
    window.location.href = page;
  }
}




// ✅ 페이지 로드 시 자동 생성
window.addEventListener('load', () => {
  frames.forEach(f => addFrameSet(f.original, f.mseg, f.sseg));
  //이미지 영상 버튼
   document.getElementById('img-mode').addEventListener('click', () => {
    navigateTo('image.html');
  });
  document.getElementById('video-mode').addEventListener('click', () => {
    navigateTo('video.html');
  });

  // 파일 업로드 이벤트 등록
  const fileInput = document.getElementById("fileInput");
  fileInput.addEventListener("change", handleVideoUpload);

  //시작 정지 버튼 이벤트 등록
  const startButton = document.querySelector(".start");
  const stopButton = document.querySelector(".stop");

  startButton.addEventListener("click", playVideo);
  stopButton.addEventListener("click", stopVideo);

  //삭제 버튼 기능
  setupDeleteFeature();
  setupAllDeleteModal({
    buttonSelector: '.all_delete',
    modalSelector: '#deleteModal',
    confirmSelector: '#confirmDelete',
    cancelSelector: '#cancelDelete',
    targetSelector: '.frames'
  });

  //frame modal 닫기 및 열기
  closeframeModal();
  //최종 분석 모달 닫고 열기
  setupFinalAnalyzeModal();
});
