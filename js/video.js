let isExtracting = false;         // 프레임 추출 중인지 여부
let extractIntervalId = null;     // setInterval ID
let lastCapturedTime = 0;         // 마지막 추출된 시점
let framesData = [];

const framesContainer = document.querySelector('.frames');
const frames = [];

let modalChart = null;
function addFrameSet(originalSrc, mSegSrc, sSegSrc) {
  const frameDiv = document.createElement('div');
  frameDiv.classList.add('frame');
 const frameIndex = framesContainer.querySelectorAll('.frame').length + 1;
 const frameNum = document.createElement('div');
  frameNum.classList.add('frame-number');
  frameNum.textContent = `Frame ${frameIndex}`;
  frameDiv.appendChild(frameNum);

 const closeBtn = document.createElement('button');
  closeBtn.classList.add('frame-close');
  closeBtn.textContent = '×';
  frameDiv.appendChild(closeBtn);

  // 원본 이미지 행
  const originalRow = document.createElement('div');
  originalRow.classList.add('frame-original-row');
  const originalImg = document.createElement('img');
  originalImg.src = originalSrc;
  originalImg.classList.add('frame-original');
  originalRow.appendChild(originalImg);

  // M seg / S seg 행
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


  frameDiv.addEventListener('click', (e) => {
    const deleteActive = document.querySelector('.delete')?.classList.contains('active');
    if (deleteActive) return; // 삭제 모드일 때는 무시
    if (e.target.closest('.frame-close')) return; // X버튼 클릭시 무시
    openframeModal(frameDiv);
  });

  framesContainer.appendChild(frameDiv);
}

function updateFrameNumbers() {
  const frames = document.querySelectorAll('.frame');
  frames.forEach((frame, index) => {
    const numberEl = frame.querySelector('.frame-number');
    if (numberEl) {
      numberEl.textContent = `Frame ${index + 1}`;
    }
  });
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
  videoPreview.load();
}

//영상 재생 함수
function playVideo(event){
  const video = document.getElementById('preview');
  const framesContainer = document.querySelector('.frames');
  const deleteButton = document.querySelector('.delete');
  if (deleteButton) {
    deleteButton.classList.remove('active');
    deleteButton.textContent = '삭제';
  }
  document.querySelectorAll('.frame').forEach(f => f.classList.remove('show-close'));
  if (!video || !video.currentSrc) {
    console.warn("영상이 없습니다.");
    return;
  }
  if (video.paused || video.ended) {
    video.play().then(() => {
      startExtractFrames(video); // 영상 재생 시작 시 프레임 추출 시작
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

function deleteFrame(event) {

  const closeBtn = event.target.closest('.frame-close');
  if (closeBtn) {
    event.stopPropagation();
    const frame = closeBtn.closest('.frame');

    if (frame) {

      const allFrames = Array.from(document.querySelectorAll('.frame'));
      const index = allFrames.indexOf(frame);


      if (index !== -1 && framesData.length > index) {
        framesData.splice(index, 1);
      }


      frame.remove();
      updateFrameNumbers();
    }


    const remainingFrames = document.querySelectorAll('.frame').length;
    const deleteButton = document.querySelector('.delete');

    if (remainingFrames === 0 && deleteButton) {
      deleteButton.classList.remove('active');
      deleteButton.textContent = '삭제';


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
function startExtractFrames(video, interval = 500){
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
// 현재 프레임을 캡처 + 서버에 보내서 세그 결과 추가
function captureCurrentFrame(videoElement, time) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async blob => {
    const frameURL = URL.createObjectURL(blob);

    // ✅ 서버로 프레임 전송
    const formData = new FormData();
    formData.append("file", blob);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error("서버 예측 실패");
      const result = await response.json();

      // ✅ 세그 이미지(Base64)
      const mSegSrc = result.M_mask;
      const sSegSrc = result.S_mask;

      // ✅ 프레임을 DOM에 추가
      addFrameSet(frameURL, mSegSrc, sSegSrc);

      framesData.push({
        id: `frame_${framesData.length}`,
        timestamp: time.toFixed(2),
        blob: blob,
        S_area: result.S_area,
        M_area: result.M_area
      });
    } catch (err) {
      console.error("❌ 프레임 분석 실패:", err);
    }
  }, "image/jpeg");
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
  framesData = [];
  updateFrameNumbers();

  const deleteButton = document.querySelector('.delete');
  if (deleteButton) {
    deleteButton.classList.remove('active');
    deleteButton.textContent = '삭제';
  }

  document.querySelectorAll('.frame').forEach(f => f.classList.remove('show-close'));

  modal.style.display = 'none';
  console.log("✅ 모든 프레임 및 데이터 삭제 완료 (삭제 버튼 리셋)");
});



  cancelBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
}

//frame 모달 창 열기
function openframeModal(frameElement) {
  const modal = document.getElementById('frameModal');
  const modalImg = document.getElementById('frame-modalPreviewImage');
  const sValue = document.getElementById('modalSValue');
  const mValue = document.getElementById('modalMValue');


  const index = Array.from(document.querySelectorAll('.frame')).indexOf(frameElement);
  const frameData = framesData[index];
const frameNumberLabel = document.getElementById('frameNumberLabel');
if (frameNumberLabel) {
  frameNumberLabel.textContent = `Frame ${index + 1}`;
}
  // 원본 이미지
  const originalImg = frameElement.querySelector('.frame-original');
  if (originalImg) {
    modalImg.src = originalImg.src;
  }


  const S_area = frameData?.S_area ?? 0;
  const M_area = frameData?.M_area ?? 0;
  const structureRatio = S_area > 0 ? Math.round((M_area / S_area) * 100) : 0;


const ctx = document.getElementById('modalPieChart').getContext('2d');

// 기존 차트 있으면 제거
if (modalChart) modalChart.destroy();


const centerTextPlugin = {
  id: 'centerText',
  beforeDraw: (chart) => {
    const { width, height, ctx } = chart;
    ctx.save();
    const value = `${structureRatio}%`; // 중앙 퍼센트 텍스트
    const fontSize = (height / 4).toFixed(2);
    ctx.font = `${fontSize}px Arial`;
    ctx.fillStyle = '#333';
    ctx.textBaseline = 'middle';
    const textX = (width - ctx.measureText(value).width) / 2;
    const textY = height / 2;
    ctx.fillText(value, textX, textY);
    ctx.restore();
  }
};

//  도넛 차트 생성
modalChart = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ['부착생물', '남은영역'],
    datasets: [{
      data: [structureRatio, 100 - structureRatio],
      backgroundColor: ['#007bff', '#e9ecef'],
      borderWidth: 0
    }]
  },
  options: {
    cutout: '70%',
    plugins: { legend: { display: false } },
    responsive: false
  },
  plugins: [centerTextPlugin]
});


  //  막대그래프 업데이트
  const sFill = modal.querySelector('.s-fill');
  const mFill = modal.querySelector('.m-fill');
  sFill.style.width = `${S_area}%`;
  sFill.querySelector('.percent-text').textContent = `${Math.round(S_area)}%`;
  mFill.style.width = `${M_area}%`;
  mFill.querySelector('.percent-text').textContent = `${Math.round(M_area)}%`;

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
  const modal = document.getElementById('finalanalyzeModal');
  const closeBtn = document.getElementById('finalanalyzeModalClose');
  const analyzeBtn = document.querySelector('.analyze');
  let avgChart = null;
  let lineChart = null;

  analyzeBtn.addEventListener('click', () => {
    if (framesData.length === 0) {
      alert("아직 분석된 프레임이 없습니다.");
      return;
    }

    modal.style.display = 'flex';


    const thumbContainer = document.getElementById('frameThumbnails');
    thumbContainer.innerHTML = '';
 framesData.forEach((f, idx) => {
  // ✅ 프레임 컨테이너
  const thumbWrapper = document.createElement('div');
  thumbWrapper.classList.add('thumb-wrapper');

  // ✅ 번호 텍스트
  const label = document.createElement('div');
  label.classList.add('thumb-number');
  label.textContent = `Frame ${idx + 1}`;

  // ✅ 이미지
  const img = document.createElement('img');
  img.src = URL.createObjectURL(f.blob);
  img.alt = `Frame ${idx + 1}`;
  img.title = `Frame ${idx + 1}`;

  // ✅ 조립
  thumbWrapper.appendChild(label);
  thumbWrapper.appendChild(img);
  thumbContainer.appendChild(thumbWrapper);
});


    //  평균 계산
    const avgStructure = framesData
      .map(f => (f.S_area > 0 ? f.M_area / f.S_area : 0))
      .reduce((a, b) => a + b, 0) / framesData.length;

    const avgPercent = Math.round(avgStructure * 100);

    //  도넛 차트 (평균)
    const ctxPie = document.getElementById('modalPieChart2').getContext('2d');
    if (avgChart) avgChart.destroy();

    const centerTextPlugin = {
      id: 'centerText',
      beforeDraw: (chart) => {
        const { width, height, ctx } = chart;
        ctx.save();
        const text = `${avgPercent}%`;
        const fontSize = (height / 4).toFixed(2);
        ctx.font = `${fontSize}px Arial`;
        ctx.fillStyle = '#333';
        ctx.textBaseline = 'middle';
        const textX = (width - ctx.measureText(text).width) / 2;
        const textY = height / 2;
        ctx.fillText(text, textX, textY);
        ctx.restore();
      }
    };

    avgChart = new Chart(ctxPie, {
      type: 'doughnut',
      data: {
        labels: ['평균 부착', '남은 영역'],
        datasets: [{
          data: [avgPercent, 100 - avgPercent],
          backgroundColor: ['#007bff', '#e9ecef'],
          borderWidth: 0
        }]
      },
      options: {
        cutout: '70%',
        plugins: { legend: { display: false } },
        responsive: false
      },
      plugins: [centerTextPlugin]
    });

    //  꺾은선 그래프
    const ctxLine = document.getElementById('modalGraph').getContext('2d');
    if (lineChart) lineChart.destroy();

    const frameLabels = framesData.map((_, i) => `Frame ${i + 1}`);
    const frameRatios = framesData.map(f =>
      f.S_area > 0 ? Math.round((f.M_area / f.S_area) * 100) : 0
    );

    lineChart = new Chart(ctxLine, {
      type: 'line',
      data: {
        labels: frameLabels,
        datasets: [{
          label: '구조물 대비 부착생물 비율(%)',
          data: frameRatios,
          borderColor: '#007bff',
          backgroundColor: 'rgba(0,123,255,0.2)',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 4,
          fill: true
        }]
      },
    options: {
    responsive: false,
    plugins: { legend: { display: false } },
    layout: {
      padding: {
        top: 20,
        bottom: 10
      }
    },
    scales: {
      y: {
        min: 50,
        max: 100,
        ticks: {
          stepSize: 10,
          color: '#444',
          font: { size: 12 }
        },
        grid: {
          color: 'rgba(200,200,200,0.3)'
        }
      },
      x: {
        ticks: {
          color: '#555',
          font: { size: 12 }
        },
        grid: {
          color: 'rgba(230,230,230,0.3)'
        }
      }
    },
      clip: false
  }
});
  });

  // 닫기 버튼
  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });
}


function navigateTo(page) {
  const currentPath = window.location.pathname;
  if (!currentPath.endsWith(page)) {
    window.location.href = page;
  }
}




//  페이지 로드 시 자동 생성
window.addEventListener('load', () => {
  const currentPage = window.location.pathname;
const imgBtn = document.getElementById('img-mode');
const vidBtn = document.getElementById('video-mode');

if (currentPage.includes('image')) {
  imgBtn.classList.add('active');
  vidBtn.classList.remove('active');
} else if (currentPage.includes('video')) {
  vidBtn.classList.add('active');
  imgBtn.classList.remove('active');
}
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
