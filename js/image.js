let donutChart = null;



function drawDonutChart(percent, elementId = 'doughnutChart') {
  const ctx = document.getElementById(elementId).getContext('2d');
  if (donutChart) donutChart.destroy();
  const centerTextPlugin = {
    id: 'centerText',
    beforeDraw: (chart) => {
      const { width, height } = chart;
      const ctx = chart.ctx;
      const text = `${chart.config.data.datasets[0].data[0]}%`;
      const fontSize = (height / 4).toFixed(2);

      ctx.restore();
      ctx.font = `${fontSize}px Arial`;
      ctx.fillStyle = '#333';
      ctx.textBaseline = 'middle';
      const textX = (width - ctx.measureText(text).width) / 2;
      const textY = height / 2;
      ctx.fillText(text, textX, textY);
      ctx.save();
    }
  };

  donutChart=new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['채워진 영역', '남은 영역'],
      datasets: [{
        data: [percent, 100 - percent],
        backgroundColor: ['#007bff', '#e9ecef'],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '70%',
      responsive: false,
      plugins: { legend: { display: false } }
    },
    plugins: [centerTextPlugin]
  });
}


async function sendImageToServer(file) {
  const formData = new FormData();
  formData.append("file", file);
formData.append("mode", "image");

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("서버 요청 실패");
    const result = await response.json();
    console.log("✅ 서버 응답:", result);

    // 1. S_area, M_area: 전체 이미지 대비 (%)
    const sPercent = result.S_area;
        const mPercent = Math.min(result.M_area ?? 0, result.S_area??0);

    // 2. 구조물 대비 부착생물 비율
    const structureRatio = sPercent > 0 ? Math.round(Math.min((mPercent / sPercent),1) * 100) : 0;

    // 3. 도넛 (구조물 대비 부착비율)
    drawDonutChart(structureRatio);

    // 4. 막대 그래프 (전체 대비)
   document.querySelector(".s-fill").style.width = `${sPercent}%`;
document.querySelector(".s-fill .percent-text").textContent = `${Math.round(sPercent)}%`;

document.querySelector(".m-fill").style.width = `${mPercent}%`;
document.querySelector(".m-fill .percent-text").textContent = `${Math.round(mPercent)}%`;
const segBoxes = document.querySelectorAll(".seg-box");
segBoxes[0].innerHTML = `<img src="${result.M_mask}" alt="M mask" style="width:100%;height:100%;object-fit:contain;">`;
segBoxes[1].innerHTML = `<img src="${result.S_mask}" alt="S mask" style="width:100%;height:100%;object-fit:contain;">`;
  } catch (error) {
    console.error("🚨 예측 요청 실패:", error);
  }
}


function handleImageUpload(inputId, previewBoxId) {
  const fileInput = document.getElementById(inputId);
  const previewBox = document.getElementById(previewBoxId);

  fileInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      previewBox.innerHTML = '';  // 기존 내용 제거
      const img = document.createElement('img');
      img.src = e.target.result;
      img.style.width = '100%';     // 박스 크기에 맞게
      img.style.height = '100%';    // 박스 크기에 맞게
      img.style.objectFit = 'contain'; // 비율 유지하면서 채우기
      previewBox.appendChild(img);
    };
    reader.readAsDataURL(file);

  });
}

function navigateTo(page) {
  const currentPath = window.location.pathname;
  if (!currentPath.endsWith(page)) {
    window.location.href = page;
  }
}

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

  drawDonutChart(0);
  document.querySelector(".s-fill").style.width = "0%";
  document.querySelector(".s-fill .percent-text").textContent = "0%";
  document.querySelector(".m-fill").style.width = "0%";
  document.querySelector(".m-fill .percent-text").textContent = "0%";


  handleImageUpload('fileInput', 'previewBox');


  document.getElementById('img-mode').addEventListener('click', () => navigateTo('image.html'));
  document.getElementById('video-mode').addEventListener('click', () => navigateTo('video.html'));


  document.querySelector(".analyze").addEventListener("click", async () => {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
      alert("이미지를 먼저 업로드해주세요!");
      return;
    }
    console.log("🔍 분석 시작: 서버에 요청 중...");
    await sendImageToServer(file);
  });
});


