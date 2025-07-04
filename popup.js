// popup.js (tooltip akıllı yönlendirme dahil)

// 1. Skora göre etiket, renk ve emoji belirleyen yardımcı fonksiyon
function labelAndColor(score) {
  if (score >= 70) return { label: "Spam", color: "red", emoji: "🔴" };
  if (score >= 40) return { label: "Suspicious", color: "orange", emoji: "🟡" };
  return { label: "Safe", color: "green", emoji: "🟢" };
}

// 2. Yüzdelik skora göre renkli progress bar oluşturan fonksiyon (HTML string olarak)
function createScoreBar(score, color) {
  return `
    <div class="score-bar">
      <div class="score-bar-fill" style="width:${score}%; background-color:${color};"></div>
    </div>`;
}

// 3. Email analizini yapan ana fonksiyon
async function analyzeEmail() {
    // Sonuç kısmını sıfırla
  document.querySelector("#result").textContent = "Analyzing...";
  document.querySelector("#details").textContent = "";

  // Kullanıcının seçtiği metin ve URL modeli alınır
  const selectedModel = document.querySelector("#modelSelect").value;
  const selectedUrlModel = document.querySelector("#urlModelSelect").value;

  // Thunderbird API ile mevcut mail sekmesini ve seçili mesajları al
  let currentTab = await browser.mailTabs.getCurrent();
  let { messages: selected } = await browser.mailTabs.getSelectedMessages(currentTab.id);

  // Hiç email seçilmemişse uyarı ver
  if (!selected || selected.length === 0) {
    document.querySelector("#result").textContent = "⚠️ Please select an email.";
    return;
  }

  // Seçilen emailin içeriğini al
  let full = await browser.messages.getFull(selected[0].id);
  let bodyPart = full.parts.find(p => p.contentType === "text/plain");
  let emailBody = bodyPart ? bodyPart.body : "";

  // 4. Text analizi

  // Flask API'ye POST isteği göndererek spam skoru alınır
  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email_text: emailBody,
      model: selectedModel
    })
  });

  const data = await response.json();

  // Ortalama skor ve seçilen modele ait skor
  const avgStatus = labelAndColor(data.spam_score);
  const selectedScore = data.model_scores[selectedModel];
  const selectedStatus = labelAndColor(selectedScore);

  // Model adını düzgün göster
  let label = selectedModel.replaceAll("_", " ").replace(/\b\w/g, l => l.toUpperCase());
  if (label === "Svm") label = "SVM";

  // HTML sonuç içeriği oluştur (text modeli için)
  let resultHtml = `${avgStatus.emoji} <strong style="color:${avgStatus.color}">Text Spam Score (Average): ${data.spam_score}% – ${avgStatus.label}</strong>` +
                   createScoreBar(data.spam_score, avgStatus.color);

  resultHtml += `<br>${selectedStatus.emoji} <span style="color:${selectedStatus.color}">${label} Score: ${selectedScore}% – ${selectedStatus.label}</span>` +
                createScoreBar(selectedScore, selectedStatus.color);

  // 5. Url analizi

  // Email içindeki tüm URL’leri ayıkla
  const urlsInText = emailBody.match(/https?:\/\/\S+|www\.\S+/g);
  let detailsHtml = "";

  if (urlsInText && urlsInText.length > 0) {
    // Kullanılan URL modelleri
    const urlModels = ["naive_bayes", "svm", "decision_tree"];
    const urlScores = {};

    // Her bir model için API'den skor alınır
    for (const model of urlModels) {
      try {
        const urlResponse = await fetch("http://localhost:5000/predict_url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email_text: emailBody,
            model: model
          })
        });

        const urlData = await urlResponse.json();
        const score = urlData.selected_score;
        urlScores[model] = score;
      } catch (error) {
        console.error(`URL prediction failed for ${model}:`, error);
      }
    }

    // Sadece Naive Bayes ve SVM skorlarının ortalaması alınır (Decision Tree dışlanabilir)
    const averageScore = Math.round((urlScores["naive_bayes"] + urlScores["svm"]) / 2);
    const avgLabel = labelAndColor(averageScore);

    detailsHtml += `${avgLabel.emoji} <span style="color:${avgLabel.color}">URL Spam Score (Average): ${averageScore}% – ${avgLabel.label}</span>` +
                   createScoreBar(averageScore, avgLabel.color);

    // Kullanıcının seçtiği URL modeline ait skor ve tooltip kontrolü
    if (selectedUrlModel in urlScores) {
      const score = urlScores[selectedUrlModel];
      const label = labelAndColor(score);
      let displayName = selectedUrlModel.replaceAll("_", " ").replace(/\b\w/g, l => l.toUpperCase());
      if (displayName === "Svm") displayName = "SVM";

      // Decision Tree modeli 0 veya 100 verirse tooltip göster
      const extraInfo = (selectedUrlModel === "decision_tree" && (score === 0 || score === 100))
        ? `<span class="tooltip"> ℹ<span class="tooltiptext">Decision Tree makes hard binary decisions, not included in Average.</span></span>`
        : "";

      detailsHtml += `<br>${label.emoji} <span style="color:${label.color}">${displayName} Score: ${score}% – ${label.label}${extraInfo}</span>` +
                     createScoreBar(score, label.color);
    }
  } else {
    // Email içinde URL yoksa bilgi mesajı göster
    detailsHtml = `<span style="color: gray; font-weight: bold;">⚠️ No URLs found in this email.</span>`;
  }

  // Sonuçları popup içine yazdır
  document.querySelector("#result").innerHTML = resultHtml;
  document.querySelector("#details").innerHTML = detailsHtml;

  // Tooltip yönünü akıllı şekilde ayarla (sol/sağ)
  document.querySelectorAll(".tooltip").forEach(tip => {
    const rect = tip.getBoundingClientRect();
    const screenWidth = window.innerWidth;

    if (rect.left < screenWidth / 2) {
      tip.classList.remove("tooltip-left");
      tip.classList.add("tooltip-right");
    } else {
      tip.classList.remove("tooltip-right");
      tip.classList.add("tooltip-left");
    }
  });
}

// 6. Model dropdownları değişince yeniden analiz yap
document.getElementById("modelSelect").addEventListener("change", analyzeEmail);
document.getElementById("urlModelSelect").addEventListener("change", analyzeEmail);

// 7. Popup ilk açıldığında analiz otomatik başlasın
analyzeEmail();
