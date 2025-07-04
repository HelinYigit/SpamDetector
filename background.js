// Eklenti ilk yüklendiğinde (kurulum veya güncelleme sırasında) tetiklenen olay dinleyici
browser.runtime.onInstalled.addListener(() => {
  console.log("Spam Detector extension installed and running.");
});
