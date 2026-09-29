let audioContext;
let processor; 
let input; 
let globalStream;
let audioData = []; 
let sampleRate = 44100; 

function setActiveTab(isText){
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-btn")[isText ? 0 : 1].classList.add("active");
}

function showText(){
    document.getElementById("textSection").classList.remove("hidden"); 
    document.getElementById("voiceSection").classList.add("hidden"); 
    setActiveTab(true);
}

function showVoice(){
    document.getElementById("voiceSection").classList.remove("hidden"); 
    document.getElementById("textSection").classList.add("hidden"); 
    setActiveTab(false);
}

async function analyzeText(){
    const btn = document.querySelector("#textSection .primary-btn"); 
    btn.classList.add("loading"); 

    const text = document.getElementById("textInput").value;

    try {
        const response = await fetch("/analyze_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();
        if (!response.ok){
            alert(result.error || "Text analysis failed.");
            return;
        }

        document.getElementById("textEmotion").innerText = result.emotion;
        document.getElementById("textConfidence").innerText = result.confidence + "%";

    } catch (err){
        console.error(err);
        alert("Text analysis failed.");
    } finally {
        btn.classList.remove("loading");
    }
}

async function analyzeVoice(){
    const btn = document.querySelector("#voiceSection .primary-btn"); 

    const fileInput = document.getElementById("audioFile");

    if (fileInput.files.length === 0){
        alert("Please upload or record audio first. ");
        return;
    }

    btn.classList.add("loading"); 
    btn.disabled = true;

    const formData = new FormData(); 
    formData.append("audio", fileInput.files[0]);    

    try {
        const response = await fetch("/analyze_speech", {
            method: "POST", 
            body: formData
        });

        const result = await response.json();
        if (!response.ok){
            alert(result.error || "Speech analysis failed.");
            return;
        }

        document.getElementById("voiceEmotion").innerText = result.emotion;
        document.getElementById("voiceConfidence").innerText = result.confidence + "%";
    
    } catch (err){
        console.error(err); 
        alert("Speech analysis failed. ");
    } finally {
        btn.classList.remove("loading");
        btn.disabled = false;
    }   
}

async function startRecording(){
    globalStream = await navigator.mediaDevices.getUserMedia({ audio: true});

    audioContext = new AudioContext();
    sampleRate = audioContext.sampleRate;

    input = audioContext.createMediaStreamSource(globalStream);
    // Deprecated but still supported in all browsers
    processor = audioContext.createScriptProcessor(4096, 1, 1); 

    audioData = []; 

    processor.onaudioprocess = event => {
        audioData.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    }; 

    input.connect(processor);
    processor.connect(audioContext.destination);

    document.getElementById("startBtn").classList.add("recording");
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
}

function stopRecording(){
    try {
        if (processor) processor.disconnect(); 
        if (input) input.disconnect(); 

        if (globalStream){
            globalStream.getTracks().forEach(track => track.stop()); 
        }

        const wavBlob = encodeWAV(audioData, sampleRate); 

        const file = new File([wavBlob], "recorded.wav", { type: "audio/wav" }); 
        const dt = new DataTransfer(); 
        dt.items.add(file); 
        document.getElementById("audioFile").files = dt.files; 

        const preview = document.getElementById("audioPreview");

        if (preview){
            preview.src = "";
            preview.style.display = "none";
        }

    } catch (err){
        console.error("Stop recording failed: ", err); 
        alert("Stop failed. Check console error. ");
    } finally {
        document.getElementById("startBtn").classList.remove("recording");
        document.getElementById("startBtn").disabled = false;
        document.getElementById("stopBtn").disabled = true;
    }
}

function encodeWAV(buffers, sampleRate){
    const buffer = flattenArray(buffers); 

    const arrayBuffer = new ArrayBuffer(44 + buffer.length * 2); 
    const view = new DataView(arrayBuffer); 

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + buffer.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); 
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, buffer.length * 2, true);

    let offset = 44;
    for (let i = 0; i < buffer.length; i++){
            let s = Math.max(-1, Math.min(1, buffer[i])); 
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            offset += 2;
    }

    return new Blob([arrayBuffer], { type: "audio/wav" });
}

function flattenArray(buffers){
    let length = buffers.reduce((a, b) => a + b.length, 0);
    let result = new Float32Array(length);
    let offset = 0;

    buffers.forEach(b => {
        result.set(b, offset);
        offset += b.length;
    }); 

    return result;
}

function writeString(view, offset, string){
    for (let i = 0; i < string.length; i++){
        view.setUint8(offset + i, string.charCodeAt(i)); 
    }
}

document.getElementById("audioFile").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const audioPreview = document.getElementById("audioPreview");
    audioPreview.src = URL.createObjectURL(file);
    audioPreview.style.display = "block";
});