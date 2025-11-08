function drawDonutChart(percent, elementId = 'doughnutChart') {
  const ctx = document.getElementById(elementId).getContext('2d');

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

  new Chart(ctx, {
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
  drawDonutChart(80);
  handleImageUpload('fileInput', 'previewBox');
  document.getElementById('img-mode').addEventListener('click', () => {
    navigateTo('image.html');
  });
  document.getElementById('video-mode').addEventListener('click', () => {
    navigateTo('video.html');
  });
});

