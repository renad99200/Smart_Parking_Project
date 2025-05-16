const videoUpload = document.getElementById('videoUpload');
const videoPlayer = document.getElementById('videoPlayer');
videoUpload.addEventListener('change', async function() {
  const file = this.files[0];
  if (file) {
    const videoURL = URL.createObjectURL(file);
    videoPlayer.src = videoURL;

    // إرسال الفيديو للـ API مع اسم الماسك الثابت
    const formData = new FormData();
    formData.append('video', file);
    // اسم الماسك الثابت (غير المسار حسب مكانه على السيرفر)
    formData.append('mask', 'mask_1920_1080.png');

    // نرسل فقط اسم الملف، السيرفر سيقرأه من مكان ثابت
    try {
      const res = await fetch('/api/parking-detect-video', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.error) {
        alert('Error: ' + data.error);
      } else {
        // تحديث الأرقام في الصفحة وتخزين النتائج
        window.parkingData = data;
        document.getElementById('available-count').innerText = 'Available spaces: ' + data.available;
        document.getElementById('occupied-count').innerText = 'Occupied spaces: ' + data.occupied;
      }
    } catch (err) {
      alert('Error connecting to server.');
    }
  }
});

// لتتبع آخر عنصر تم فتحه
let lastOpened = null;

function showInfo(type) {
  const boxes = {
    available: document.getElementById('available-details'),
    occupied: document.getElementById('occupied-details'),
    cost: document.getElementById('cost-box')
  };

  // استخدم البيانات الحقيقية لو موجودة
  const data = window.parkingData || {
    available: 12,
    occupied: 8
  };

  if (lastOpened === type) {
    boxes[type].style.display = 'none';
    lastOpened = null;
    return;
  }

  for (let key in boxes) {
    boxes[key].style.display = 'none';
  }

  if (type === 'available') {
    document.getElementById('available-count').innerText = "Available spaces: " + data.available;
  } else if (type === 'occupied') {
    document.getElementById('occupied-count').innerText = "Occupied spaces: " + data.occupied;
  } else if (type === 'cost') {
    let cost = data.cost !== undefined ? data.cost : '...';
    document.getElementById('cost-value').innerText = "Cost: " + cost + " EGP";
  }

  boxes[type].style.display = 'block';
  lastOpened = type;
}