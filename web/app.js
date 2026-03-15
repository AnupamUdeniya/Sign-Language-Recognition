const HF_API =
"https://anupam090-asl-sign-language-recognition.hf.space/run/predict";

const classes = [
"A","B","C","D","E","F","G","H","I","J","K","L","M","N",
"O","P","Q","R","S","T","U","V","W","X","Y","Z","del","nothing","space"
];

const predictionLetter = document.getElementById("predictionLetter");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const phraseBox = document.getElementById("phraseBox");
const statusMessage = document.getElementById("statusMessage");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const liveDetectBtn = document.getElementById("liveDetectBtn");
const analyzeFrameBtn = document.getElementById("analyzeFrameBtn");

const imageUpload = document.getElementById("imageUpload");
const cameraFeed = document.getElementById("cameraFeed");
const captureCanvas = document.getElementById("captureCanvas");

let currentPrediction = { label: "nothing", confidence: 0 };
let mediaStream = null;
let liveDetectionTimer = null;
let requestInFlight = false;

function prettyLabel(label){
if(label==="del") return "Delete";
if(label==="nothing") return "No sign";
if(label==="space") return "Space";
return `Letter ${label}`;
}

function setStatus(msg){
statusMessage.textContent = msg;
}

function blobToBase64(blob){
return new Promise(resolve=>{
const reader = new FileReader();
reader.onloadend = ()=>resolve(reader.result);
reader.readAsDataURL(blob);
});
}

function updatePrediction(result){

currentPrediction = result;

predictionLetter.textContent =
result.label==="space"?"_" :
result.label==="nothing"?"?" :
result.label;

predictionLabel.textContent = prettyLabel(result.label);
predictionConfidence.textContent = `${result.confidence}%`;
}

function appendPrediction(){

const txt = phraseBox.textContent;

if(currentPrediction.label==="space"){
phraseBox.textContent = txt+" ";
return;
}

if(currentPrediction.label==="del"){
phraseBox.textContent = txt.slice(0,-1);
return;
}

if(currentPrediction.label==="nothing") return;

phraseBox.textContent = txt+currentPrediction.label;
}

async function startCamera(){

try{

mediaStream = await navigator.mediaDevices.getUserMedia({
video:true,
audio:false
});

cameraFeed.srcObject = mediaStream;
await cameraFeed.play();

setStatus("Camera started");

}catch(err){

setStatus("Camera error: "+err.message);

}
}

function stopCamera(){

if(mediaStream){
mediaStream.getTracks().forEach(t=>t.stop());
}

mediaStream = null;
setStatus("Camera stopped");
}

function captureCurrentFrameBlob(){

captureCanvas.width = cameraFeed.videoWidth;
captureCanvas.height = cameraFeed.videoHeight;

const ctx = captureCanvas.getContext("2d");

ctx.drawImage(
cameraFeed,
0,
0,
captureCanvas.width,
captureCanvas.height
);

return new Promise(resolve=>{
captureCanvas.toBlob(resolve,"image/jpeg");
});
}

async function sendImageForPrediction(blob){

if(requestInFlight) return;

requestInFlight=true;

setStatus("Running AI model...");

try{

const base64 = await blobToBase64(blob);

const response = await fetch(HF_API,{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
data:[base64]
})
});

const result = await response.json();

let label="nothing";

if(result && result.data){

label=result.data[0];

if(Array.isArray(label)){
label=label[0];
}

}

updatePrediction({
label:label,
confidence:100
});

setStatus(`Detected ${prettyLabel(label)}`);

}catch(err){

setStatus("Prediction error: "+err.message);

}

requestInFlight=false;
}

async function analyzeCurrentFrame(){

if(!mediaStream){
setStatus("Start camera first");
return;
}

const blob = await captureCurrentFrameBlob();

await sendImageForPrediction(blob);
}

function toggleLiveDetection(){

if(liveDetectionTimer){

clearInterval(liveDetectionTimer);
liveDetectionTimer=null;
setStatus("Live detection stopped");
return;

}

liveDetectionTimer=setInterval(()=>{
analyzeCurrentFrame();
},1500);

setStatus("Live detection running");
}

function handleUpload(e){

const file = e.target.files[0];

if(!file) return;

sendImageForPrediction(file);
}

startCameraBtn.addEventListener("click",startCamera);
stopCameraBtn.addEventListener("click",stopCamera);
liveDetectBtn.addEventListener("click",toggleLiveDetection);
analyzeFrameBtn.addEventListener("click",analyzeCurrentFrame);
imageUpload.addEventListener("change",handleUpload);

setStatus("Connected to HuggingFace model");