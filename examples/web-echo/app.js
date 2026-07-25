// web-echo: on Enter, POST the text to the local tts_eu_pt server, play the returned WAV.
(function () {
  const input = document.getElementById("text");
  const status = document.getElementById("status");
  let busy = false;

  async function speak(text) {
    if (!text.trim() || busy) return;
    busy = true;
    status.textContent = "A sintetizar…";
    try {
      const res = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      // The Enter keypress is a user gesture, so playback is allowed by autoplay policy.
      await audio.play();
      status.textContent = "";
    } catch (e) {
      status.textContent = "Erro: " + e.message;
    } finally {
      busy = false;
    }
  }

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      speak(input.value);
    }
  });
})();
