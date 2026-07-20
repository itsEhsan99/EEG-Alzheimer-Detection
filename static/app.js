const CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
];
const BANDS = [
    { key: "delta", name: "Delta", freq: "0.5–4 Hz", waves: 1.5 },
    { key: "theta", name: "Theta", freq: "4–8 Hz", waves: 3 },
    { key: "alpha", name: "Alpha", freq: "8–13 Hz", waves: 5 },
    { key: "beta",  name: "Beta",  freq: "13–30 Hz", waves: 8 },
    { key: "gamma", name: "Gamma", freq: "30–45 Hz", waves: 12 },
];
let selectedBand = "alpha";

// Central preprocessing state — applied to every analysis request.
const preprocessing = {
    resample_freq: 0,
    low_freq: 0,
    high_freq: 0,
    notch_freq: 0,
    drop_channels: "",
};

const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const loadBtn = document.getElementById("loadBtn");
const statusDiv = document.getElementById("status");

document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    });
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    fileName.textContent = file ? file.name : "Choose a .set file…";
});

function requireFile() {
    const file = fileInput.files[0];
    if (!file) { statusDiv.textContent = "Please choose a .set file first."; return null; }
    return file;
}

// Append the file + preprocessing settings + any extra fields to a request.
async function fetchPlot(endpoint, img, container, btn, statusMsg, extra = {}) {
    const file = requireFile();
    if (!file) return;
    statusDiv.textContent = statusMsg;
    btn.disabled = true;
    try {
        const formData = new FormData();
        formData.append("file", file);
        for (const [k, v] of Object.entries(preprocessing)) formData.append(k, v);
        for (const [k, v] of Object.entries(extra)) formData.append(k, v);
        const response = await fetch(endpoint, { method: "POST", body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Request failed.");
        }
        const blob = await response.blob();
        img.src = URL.createObjectURL(blob);
        container.classList.remove("hidden");
        statusDiv.textContent = "";
    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
    } finally {
        btn.disabled = false;
    }
}

// Brain SVG
function brainSvg(size = 120) {
    return `
    <svg width="${size}" height="${size * 0.92}" viewBox="0 0 120 110">
      <g fill="none" stroke="#22d3ee" stroke-width="1" opacity="0.55">
        <path d="M4 30 Q11 20 18 30 T32 30"/><path d="M4 55 Q11 45 18 55 T32 55"/><path d="M4 80 Q11 70 18 80 T32 80"/>
      </g>
      <g fill="none" stroke="#a78bfa" stroke-width="1" opacity="0.55">
        <path d="M88 30 Q95 20 102 30 T116 30"/><path d="M88 55 Q95 45 102 55 T116 55"/><path d="M88 80 Q95 70 102 80 T116 80"/>
      </g>
      <path d="M60 16 C43 16 37 29 39 40 C28 42 28 59 39 63 C37 75 50 84 60 79
               C70 84 83 75 81 63 C92 59 92 42 81 40 C83 29 77 16 60 16 Z"
            fill="#14203a" stroke="#f59e0b" stroke-width="1.6"/>
      <path d="M60 18 L60 79" stroke="#f59e0b" stroke-width="1" opacity="0.5"/>
      <path d="M49 28 Q56 40 49 50 Q42 60 51 71" stroke="#f59e0b" stroke-width="0.9" fill="none" opacity="0.45"/>
      <path d="M71 28 Q64 40 71 50 Q78 60 69 71" stroke="#f59e0b" stroke-width="0.9" fill="none" opacity="0.45"/>
    </svg>`;
}
document.getElementById("introBrain").innerHTML = brainSvg(150);
document.getElementById("heroBrain").innerHTML = brainSvg(120);

// Model architecture SVG
function modelArchSvg() {
    return `
    <svg width="260" height="120" viewBox="0 0 260 120">
      <!-- graph input: a few connected nodes -->
      <circle cx="18" cy="35" r="5" fill="#22d3ee"/>
      <circle cx="40" cy="20" r="5" fill="#22d3ee"/>
      <circle cx="45" cy="55" r="5" fill="#22d3ee"/>
      <circle cx="20" cy="72" r="5" fill="#22d3ee"/>
      <g stroke="#2a3a52" stroke-width="1">
        <line x1="18" y1="35" x2="40" y2="20"/>
        <line x1="18" y1="35" x2="45" y2="55"/>
        <line x1="45" y1="55" x2="20" y2="72"/>
        <line x1="40" y1="20" x2="45" y2="55"/>
      </g>
      <text x="32" y="92" fill="#94a3b8" font-size="8" text-anchor="middle">PLV graph</text>
      <!-- GCN layers -->
      <rect x="78" y="30" width="30" height="50" rx="4" fill="#1a2234" stroke="#f59e0b"/>
      <text x="93" y="93" fill="#94a3b8" font-size="8" text-anchor="middle">GCN 1</text>
      <rect x="120" y="30" width="30" height="50" rx="4" fill="#1a2234" stroke="#f59e0b"/>
      <text x="135" y="93" fill="#94a3b8" font-size="8" text-anchor="middle">GCN 2</text>
      <!-- MLP head -->
      <rect x="162" y="42" width="28" height="26" rx="4" fill="#7c3aed22" stroke="#a78bfa"/>
      <text x="176" y="82" fill="#94a3b8" font-size="8" text-anchor="middle">MLP</text>
      <!-- output -->
      <text x="228" y="51" fill="#22d3ee" font-size="9" text-anchor="middle">AD</text>
      <text x="228" y="66" fill="#a78bfa" font-size="9" text-anchor="middle">HC</text>
      <g stroke="#2a3a52" stroke-width="1.2">
        <line x1="52" y1="50" x2="78" y2="50"/>
        <line x1="108" y1="55" x2="120" y2="55"/>
        <line x1="150" y1="55" x2="162" y2="55"/>
        <line x1="190" y1="55" x2="214" y2="55"/>
      </g>
    </svg>`;
}
document.getElementById("modelArch").innerHTML = modelArchSvg();

// Load recording -> metadata
loadBtn.addEventListener("click", async () => {
    const file = requireFile();
    if (!file) return;
    statusDiv.textContent = "Loading recording…";
    loadBtn.disabled = true;
    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/metadata", { method: "POST", body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to load.");
        }
        const m = await response.json();
        document.getElementById("ovFilename").textContent = m.filename;
        document.getElementById("ovDuration").textContent = `${m.duration_minutes} min`;
        document.getElementById("ovChannels").textContent = `${m.n_channels} channels`;
        document.getElementById("mSfreq").textContent = `${m.sampling_frequency} Hz`;
        document.getElementById("mChannels").textContent = `${m.n_channels} EEG`;
        document.getElementById("mDuration").textContent = `${m.duration_seconds} s (${m.duration_minutes} min)`;
        document.getElementById("mFilter").textContent = `${m.highpass}–${m.lowpass} Hz`;
        document.getElementById("mBads").textContent = m.n_bad_channels > 0 ? m.bad_channels.join(", ") : "None";
        document.getElementById("mSamples").textContent = m.n_samples.toLocaleString();
        document.getElementById("overviewPlaceholder").classList.add("hidden");
        document.getElementById("overviewContent").classList.remove("hidden");
        statusDiv.textContent = "";
    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
    } finally {
        loadBtn.disabled = false;
    }
});

// Prediction
const analyzeBtn = document.getElementById("analyzeBtn");
analyzeBtn.addEventListener("click", async () => {
    const file = requireFile();
    if (!file) return;
    statusDiv.textContent = "Analyzing… this may take a moment.";
    analyzeBtn.disabled = true;
    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/predict", { method: "POST", body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Something went wrong.");
        }
        const data = await response.json();
        document.getElementById("prediction").textContent = data.prediction;
        document.getElementById("adProb").textContent = data.ad_probability;
        document.getElementById("nWindows").textContent = data.n_windows;
        document.getElementById("predPlaceholder").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");
        statusDiv.textContent = "";
    } catch (err) {
        statusDiv.textContent = "Error: " + err.message;
    } finally {
        analyzeBtn.disabled = false;
    }
});

// Signal-tab channel checkboxes
const channelList = document.getElementById("channelList");
CHANNELS.forEach(ch => {
    const label = document.createElement("label");
    label.className = "channel-chip";
    label.innerHTML = `<input type="checkbox" value="${ch}" checked> ${ch}`;
    channelList.appendChild(label);
});
function getSelectedChannels() {
    return Array.from(channelList.querySelectorAll("input[type=checkbox]"))
        .filter(b => b.checked).map(b => b.value);
}
document.getElementById("selectAllCh").addEventListener("click", () => {
    channelList.querySelectorAll("input").forEach(b => b.checked = true);
});
document.getElementById("clearCh").addEventListener("click", () => {
    channelList.querySelectorAll("input").forEach(b => b.checked = false);
});

// Preprocessing-tab drop-channel checkboxes
const ppChannelList = document.getElementById("ppChannelList");
CHANNELS.forEach(ch => {
    const label = document.createElement("label");
    label.className = "channel-chip";
    label.innerHTML = `<input type="checkbox" value="${ch}" checked> ${ch}`;
    ppChannelList.appendChild(label);
});

// Apply / reset preprocessing
const applyPpBtn = document.getElementById("applyPpBtn");
const resetPpBtn = document.getElementById("resetPpBtn");
const ppStatus = document.getElementById("ppStatus");
const ppSummary = document.getElementById("ppSummary");
const ppSummaryText = document.getElementById("ppSummaryText");

function updatePreprocessingSummary() {
    const parts = [];
    if (preprocessing.resample_freq > 0) parts.push(`Resample ${preprocessing.resample_freq} Hz`);
    if (preprocessing.low_freq > 0) parts.push(`HP ${preprocessing.low_freq} Hz`);
    if (preprocessing.high_freq > 0) parts.push(`LP ${preprocessing.high_freq} Hz`);
    if (preprocessing.notch_freq > 0) parts.push(`Notch ${preprocessing.notch_freq} Hz`);
    if (preprocessing.drop_channels) parts.push(`Drop: ${preprocessing.drop_channels}`);

    if (parts.length === 0) {
        ppSummary.classList.add("hidden");
    } else {
        ppSummaryText.textContent = parts.join(" · ");
        ppSummary.classList.remove("hidden");
    }
}

applyPpBtn.addEventListener("click", () => {
    preprocessing.resample_freq = parseFloat(document.getElementById("ppResample").value) || 0;
    preprocessing.low_freq = parseFloat(document.getElementById("ppLow").value) || 0;
    preprocessing.high_freq = parseFloat(document.getElementById("ppHigh").value) || 0;
    preprocessing.notch_freq = parseFloat(document.getElementById("ppNotch").value) || 0;

    // Dropped = the channels that are UNCHECKED
    const dropped = Array.from(ppChannelList.querySelectorAll("input[type=checkbox]"))
        .filter(b => !b.checked).map(b => b.value);
    preprocessing.drop_channels = dropped.join(",");

    updatePreprocessingSummary();
    ppStatus.textContent = "Applied ✓ — analyses will use these settings.";
    setTimeout(() => { ppStatus.textContent = ""; }, 3000);
});

resetPpBtn.addEventListener("click", () => {
    document.getElementById("ppResample").value = "0";
    document.getElementById("ppLow").value = "0";
    document.getElementById("ppHigh").value = "0";
    document.getElementById("ppNotch").value = "0";
    ppChannelList.querySelectorAll("input").forEach(b => b.checked = true);
    preprocessing.resample_freq = 0;
    preprocessing.low_freq = 0;
    preprocessing.high_freq = 0;
    preprocessing.notch_freq = 0;
    preprocessing.drop_channels = "";
    updatePreprocessingSummary();
    ppStatus.textContent = "Reset to original.";
    setTimeout(() => { ppStatus.textContent = ""; }, 3000);
});

// Band picker
function waveSvg(waves) {
    const w = 60, h = 24, mid = h / 2;
    let d = `M 0 ${mid}`;
    const steps = 60;
    for (let i = 1; i <= steps; i++) {
        const x = (i / steps) * w;
        const y = mid - Math.sin((i / steps) * waves * 2 * Math.PI) * (mid - 3);
        d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    return `<svg width="${w}" height="${h}"><path d="${d}" fill="none" stroke="#f59e0b" stroke-width="1.5"/></svg>`;
}
const bandPicker = document.getElementById("bandPicker");
BANDS.forEach(b => {
    const div = document.createElement("div");
    div.className = "band-option" + (b.key === selectedBand ? " active" : "");
    div.innerHTML = `${waveSvg(b.waves)}<span class="band-name">${b.name}</span><span class="band-freq">${b.freq}</span>`;
    div.addEventListener("click", () => {
        selectedBand = b.key;
        document.querySelectorAll(".band-option").forEach(o => o.classList.remove("active"));
        div.classList.add("active");
    });
    bandPicker.appendChild(div);
});

// Signal
const signalBtn = document.getElementById("signalBtn");
signalBtn.addEventListener("click", () => {
    fetchPlot("/signal",
        document.getElementById("signalImg"),
        document.getElementById("signalContainer"),
        signalBtn, "Rendering signal…", {
            start_time: document.getElementById("startTime").value,
            n_seconds: document.getElementById("duration").value,
            channels: getSelectedChannels().join(","),
        });
});

// PSD
const psdBtn = document.getElementById("psdBtn");
psdBtn.addEventListener("click", () => {
    fetchPlot("/psd",
        document.getElementById("psdImg"),
        document.getElementById("psdContainer"),
        psdBtn, "Computing band powers…");
});

// Connectivity
const heatmapBtn = document.getElementById("heatmapBtn");
const topomapBtn = document.getElementById("topomapBtn");
const connImg = document.getElementById("connImg");
const connContainer = document.getElementById("connContainer");
const connDesc = document.getElementById("connDesc");
function bandLabel() { return BANDS.find(b => b.key === selectedBand).name; }

heatmapBtn.addEventListener("click", async () => {
    await fetchPlot("/plv-heatmap", connImg, connContainer, heatmapBtn, "Computing PLV…", { band: selectedBand });
    connDesc.textContent = `${bandLabel()}-band PLV between all channel pairs (0 = none, 1 = perfect phase locking).`;
});
topomapBtn.addEventListener("click", async () => {
    await fetchPlot("/plv-topomap", connImg, connContainer, topomapBtn, "Mapping connections…", { band: selectedBand });
    connDesc.textContent = `Strongest 15% of ${bandLabel()}-band PLV connections on a 10–20 head layout.`;
});