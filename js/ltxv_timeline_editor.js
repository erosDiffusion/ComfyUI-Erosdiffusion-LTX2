import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- STYLES INJECTED DYNAMICALLY ---
const STYLE = `
:root {
    --bg-body-overlay: #121212;
    --bg-overlay-panel: #252525;
    --bg-timeline-track: #333;
    --bg-scene: #2a2a2a;
    --scene-selected: #3a3a3a;
    --accent-blue: #3b82f6;
    --accent-orange: #f97316;
    --border-color: #404040;
    --input-bg: #2d2d2d;
    --text-main: #e0e0e0;
    --text-dim: #a0a0a0;
    --link-color: #4ade80;
    --unlink-color: #f87171;
}

.ltxv-overlay-backdrop {
    display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0, 0, 0, 0.85); z-index: 9000;
    align-items: center; justify-content: center; backdrop-filter: blur(2px);
    font-family: 'Segoe UI', Roboto, sans-serif; color: var(--text-main);
}

.ltxv-editor-modal {
    background: var(--bg-overlay-panel); width: 95%; max-width: 1300px; height: 90vh;
    border-radius: 8px; box-shadow: 0 20px 25px rgba(0,0,0,0.5);
    border: 1px solid var(--border-color); display: flex; flex-direction: column; overflow: hidden;
}

.ltxv-modal-header {
    padding: 16px; border-bottom: 1px solid var(--border-color);
    display: flex; justify-content: space-between; align-items: center; background: #1f1f1f;
}

.ltxv-modal-body {
    flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px;
}

.ltxv-timeline-scroll {
    width: 100%; overflow-x: auto; background: var(--bg-timeline-track);
    border: 1px solid var(--border-color); border-radius: 4px; position: relative;
    height: 80px; cursor: grab; user-select: none;
}
.ltxv-timeline-scroll::-webkit-scrollbar { display: none; }
.ltxv-timeline-track { height: 100%; position: relative; min-width: 100%; }

.ltxv-scene-block {
    position: absolute; top: 2px; bottom: 2px;
    background: var(--bg-scene); border: 1px solid #444; border-radius: 4px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
    transition: background 0.1s; cursor: pointer;
}
.ltxv-scene-block:hover { background: #333; }
.ltxv-scene-block.selected { background: var(--scene-selected); border: 1px solid var(--accent-blue); z-index: 5; }
.ltxv-scene-label { font-size: 11px; color: #aaa; pointer-events: none; white-space: nowrap; overflow: hidden; padding: 0 5px; }

.ltxv-cut-handle {
    position: absolute; top: 0; bottom: 0; width: 6px; background: transparent;
    cursor: ew-resize; z-index: 20; transform: translateX(-50%);
}
.ltxv-cut-handle::after {
    content: ''; position: absolute; top: 0; bottom: 0; left: 2px; width: 2px; background: #555;
}
.ltxv-cut-handle:hover::after { background: var(--accent-blue); }

.ltxv-mid-marker {
    position: absolute; top: 0; bottom: 0; width: 4px; background: var(--accent-orange); opacity: 0.6;
    transform: translateX(-50%); z-index: 15; cursor: ew-resize;
}
.ltxv-mid-marker.selected, .ltxv-mid-marker:hover { opacity: 1; background: #fff; box-shadow: 0 0 5px var(--accent-orange); }

.ltxv-time-label {
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    font-size: 10px; background: rgba(0,0,0,0.8); padding: 2px 4px; border-radius: 3px; pointer-events: none; margin-top: 2px;
}

.ltxv-props-panel {
    background: #1a1a1a; border: 1px solid var(--border-color);
    padding: 16px; border-radius: 6px; flex: 1; display: flex; flex-direction: column; gap: 16px;
}

.ltxv-img-row {
    display: flex; align-items: center; gap: 15px; overflow-x: auto; padding-bottom: 10px; min-height: 140px;
}

.ltxv-uploader {
    display: flex; flex-direction: column; align-items: center; gap: 6px; width: 110px; flex-shrink: 0;
}
.ltxv-upload-box {
    width: 100px; height: 100px; border: 2px dashed #444; border-radius: 6px;
    display: flex; justify-content: center; align-items: center; cursor: pointer;
    position: relative; overflow: hidden; background: #222;
}
.ltxv-upload-box:hover { border-color: #666; }
.ltxv-upload-box img { width: 100%; height: 100%; object-fit: cover; }
.ltxv-upload-box span { font-size: 24px; color: #555; }
.ltxv-upload-box.highlighted { border-color: var(--accent-orange); box-shadow: 0 0 5px var(--accent-orange); }

.ltxv-link-toggle {
    display: flex; flex-direction: column; align-items: center; cursor: pointer; color: #555; width: 30px; margin-top: 30px;
}
.ltxv-link-toggle.linked { color: var(--link-color); }
.ltxv-link-toggle.unlinked { color: var(--unlink-color); }

.ltxv-textarea {
    background: var(--input-bg); border: 1px solid var(--border-color); color: var(--text-main);
    padding: 12px; border-radius: 4px; resize: none; height: 60px; width: 100%; box-sizing: border-box;
}

.ltxv-btn {
    background: #444; color: white; border: 1px solid #555; padding: 6px 16px; border-radius: 4px; cursor: pointer;
}
.ltxv-btn:hover { background: #555; }
`;

function injectStyle() {
    if (!document.getElementById("ltxv-style")) {
        const s = document.createElement("style");
        s.id = "ltxv-style";
        s.innerHTML = STYLE;
        document.head.appendChild(s);
    }
}

// --- CLASS IMPLEMENTATION ---

class LTXVTimelineEditor {
    constructor(widget, saveCallback) {
        this.widget = widget;
        this.saveCallback = saveCallback;

        // State
        this.scenes = [];
        this.duration = 5;
        this.pixelsPerSec = 30; // Zoom level
        this.selection = { sceneId: null, frameId: null };
        this.currentUploadTarget = null;
        this.lastParsedValue = null; // Track last synced value

        // Parse Widget Value
        this.syncFromWidget();

        // Build UI
        this.buildUI();
    }

    // Sync timeline state from widget value (called on open)
    syncFromWidget() {
        const currentValue = this.widget.value;
        // Only re-parse if value actually changed
        if (currentValue !== this.lastParsedValue) {
            this.parseScript(currentValue);
            this.lastParsedValue = currentValue;
            return true; // Changed
        }
        return false; // No change
    }

    // --- PARSING & SERIALIZATION ---

    parseScript(script) {
        // Regex for [MM:SS-MM:SS]
        // Example: [00:00-00:05] Prompt | first:img.png
        this.scenes = [];

        if (!script || !script.trim()) {
            // Default
            this.scenes.push(this.createScene(0, 5));
            this.duration = 5;
            return;
        }

        const lines = script.split('\n');
        let maxEnd = 0;

        const timeToSec = (str) => {
            const parts = str.split(':');
            if (parts.length === 2) return parseInt(parts[0]) * 60 + parseFloat(parts[1]);
            return 0;
        };

        lines.forEach(line => {
            const timeMatch = line.match(/^\[(\d+:\d+(?:\.\d+)?)-(\d+:\d+(?:\.\d+)?)\](.*)/);
            if (timeMatch) {
                const start = timeToSec(timeMatch[1]);
                const end = timeToSec(timeMatch[2]);
                const rest = timeMatch[3];

                const scene = this.createScene(start, end);
                // Parse directives
                const parts = rest.split('|').map(s => s.trim());
                scene.prompt = parts[0] || "";

                parts.slice(1).forEach(p => {
                    // Check directives
                    if (p.toLowerCase().startsWith('audio:')) {
                        // Parse audio spec - can be audio:silent, audio:ambient, or audio:"text"
                        let audioVal = p.substring(6).trim();
                        // Remove quotes if present
                        if (audioVal.startsWith('"') && audioVal.endsWith('"')) {
                            audioVal = audioVal.slice(1, -1);
                        }
                        scene.audioSpec = audioVal || 'silent';
                    } else if (p.startsWith('first:')) {
                        scene.startImg = p.substring(6).trim();
                    } else if (p.startsWith('end:')) {
                        scene.endImg = p.substring(4).trim();
                    } else if (p.startsWith('mid:')) {
                        // mid:MM:SS:img.png
                        const tMatch = p.match(/mid:(\d+:\d+(?:\.\d+)?):(.*)/);
                        if (tMatch) {
                            const timeVal = timeToSec(tMatch[1]);
                            const imgVal = tMatch[2];
                            scene.middleFrames.push({
                                id: 'mf_' + Math.random(),
                                time: timeVal,
                                img: imgVal
                            });
                        }
                    }
                });

                if (end > maxEnd) maxEnd = end;
                this.scenes.push(scene);
            }
        });

        if (this.scenes.length === 0) {
            this.scenes.push(this.createScene(0, 5));
            this.duration = 5;
        } else {
            this.duration = maxEnd;
        }
    }

    serializeScript() {
        // [MM:SS-MM:SS] Prompt | first:f | mid:MM:SS:f
        const secToTime = (s) => {
            const m = Math.floor(s / 60);
            const ss = (s % 60).toFixed(2);
            return `${m.toString().padStart(2, '0')}:${ss.toString().padStart(5, '0')}`; // 05.00
        };

        return this.scenes.map(s => {
            const tRange = `[${secToTime(s.start)}-${secToTime(s.end)}]`;
            let line = `${tRange} ${s.prompt}`;

            // Audio spec
            if (s.audioSpec && s.audioSpec !== 'silent') {
                // Quote if it contains spaces (dialogue)
                const audioVal = s.audioSpec.includes(' ') ? `"${s.audioSpec}"` : s.audioSpec;
                line += ` | audio:${audioVal}`;
            }

            if (s.startImg) line += ` | first:${s.startImg}`;
            // Sorted Mids
            s.middleFrames.sort((a, b) => a.time - b.time).forEach(mf => {
                if (mf.img) line += ` | mid:${secToTime(mf.time)}:${mf.img}`;
            });

            if (s.endImg) line += ` | end:${s.endImg}`;

            return line;
        }).join('\n');
    }

    createScene(start, end) {
        return {
            id: 'sc_' + Math.random().toString(36).substr(2, 6),
            start: start,
            end: end,
            prompt: '',
            audioSpec: 'silent',
            startImg: null,
            endImg: null,
            middleFrames: [],
            hardCutStart: false,
            hardCutEnd: false
        };
    }

    // --- UI BUILDING ---

    buildUI() {
        this.overlay = document.createElement("div");
        this.overlay.className = "ltxv-overlay-backdrop";

        const modal = document.createElement("div");
        modal.className = "ltxv-editor-modal";

        // Header
        const header = document.createElement("div");
        header.className = "ltxv-modal-header";
        header.innerHTML = `
            <div style="display:flex; align-items:center; gap:15px;">
                <h3 style="color:white; margin:0;">Timeline Editor</h3>
                <div style="display:flex; align-items:center; gap:8px; font-size:0.9rem; color:#aaa;">
                    <label>Duration (s):</label>
                    <input type="number" id="ltxv-dur" style="background:#333; color:white; border:1px solid #555; width:60px;" value="${this.duration}">
                </div>
            </div>
        `;
        const closeBtn = document.createElement("button");
        closeBtn.className = "ltxv-btn";
        closeBtn.innerText = "Save & Close";
        closeBtn.onclick = () => this.close();
        header.appendChild(closeBtn);
        modal.appendChild(header);

        // Body
        const body = document.createElement("div");
        body.className = "ltxv-modal-body";

        // Instructions
        const instr = document.createElement("div");
        instr.style = "background:#2a2a2a; padding:10px; border-radius:4px; font-size:0.85rem; color:#aaa; border-left:3px solid var(--accent-blue);";
        instr.innerHTML = `
            <strong>Controls:</strong> Drag background to pan. Click Scene to edit. 
            <strong>Ctrl+Click</strong> to Split/Cut. <strong>Alt+Click</strong> to Add Keyframe.
            <br>Upload images by clicking the boxes properties panel.
        `;
        body.appendChild(instr);

        // Track
        this.scroll = document.createElement("div");
        this.scroll.className = "ltxv-timeline-scroll";
        this.track = document.createElement("div");
        this.track.className = "ltxv-timeline-track";

        this.setupPanning(this.scroll);

        this.scroll.appendChild(this.track);
        body.appendChild(this.scroll);

        // Props
        this.props = document.createElement("div");
        this.props.className = "ltxv-props-panel";
        this.props.innerHTML = `<div style="text-align:center; color:#555; padding:40px;">Select a scene to edit</div>`;
        body.appendChild(this.props);

        modal.appendChild(body);
        this.overlay.appendChild(modal);
        document.body.appendChild(this.overlay);

        // File Input
        this.fileInput = document.createElement("input");
        this.fileInput.type = "file";
        this.fileInput.accept = "image/*";
        this.fileInput.hidden = true;
        this.fileInput.onchange = (e) => this.handleFileUpload(e.target.files[0]);
        document.body.appendChild(this.fileInput);

        // Event Listeners for Duration
        const durInp = header.querySelector("#ltxv-dur");
        durInp.onchange = (e) => this.updateDuration(parseFloat(e.target.value));
    }

    // --- ACTIONS ---

    open() {
        // Sync from widget in case it was edited externally
        this.syncFromWidget();
        this.overlay.style.display = "flex";
        this.render();
    }

    close() {
        const val = this.serializeScript();
        this.widget.value = val;
        if (this.saveCallback) this.saveCallback(val);
        this.overlay.style.display = "none";
    }

    updateDuration(val) {
        if (!val || val < 1) return;
        this.duration = val;
        // Adjust last scene
        if (this.scenes.length > 0) {
            const last = this.scenes[this.scenes.length - 1];
            if (last.end < val) last.end = val; // Extend
            else {
                // Shrink?
                // Simple logic: just set last.end = val.
                // Filter out scenes that are now out of bounds
                this.scenes = this.scenes.filter(s => s.start < val);
                const newLast = this.scenes[this.scenes.length - 1];
                newLast.end = val;
            }
        } else {
            this.scenes.push(this.createScene(0, val));
        }
        this.render();
    }

    // --- RENDERING ---

    render() {
        this.track.innerHTML = "";
        this.track.style.width = (this.duration * this.pixelsPerSec) + "px";

        this.scenes.forEach((s, idx) => {
            // Scene Block
            const el = document.createElement("div");
            el.className = "ltxv-scene-block";
            const left = (s.start / this.duration) * 100;
            const w = ((s.end - s.start) / this.duration) * 100;
            el.style.left = left + "%";
            el.style.width = w + "%";
            if (this.selection.sceneId === s.id) el.classList.add("selected");

            el.innerHTML = `<span class="ltxv-scene-label">Scene ${idx + 1}</span>`;

            el.onclick = (e) => this.handleSceneClick(e, s);

            this.track.appendChild(el);

            // Middle Frames
            s.middleFrames.forEach(mf => {
                const mel = document.createElement("div");
                mel.className = "ltxv-mid-marker";
                mel.style.left = (mf.time / this.duration) * 100 + "%";
                if (this.selection.frameId === mf.id) mel.classList.add("selected");

                // Add drag handler
                mel.onmousedown = (e) => this.startDragMid(e, s, mf);

                // Tooltip time
                const tt = document.createElement("div");
                tt.className = "ltxv-time-label";
                tt.innerText = mf.time.toFixed(1) + "s";
                mel.appendChild(tt);

                mel.onmousedown = (e) => this.startDragMid(e, s, mf);
                mel.onclick = (e) => {
                    this.selection = { sceneId: s.id, frameId: mf.id };
                    this.render(); this.renderProps();
                    e.stopPropagation();
                };
                this.track.appendChild(mel);
            });

            // Cut Handle
            if (idx < this.scenes.length - 1) {
                const h = document.createElement("div");
                h.className = "ltxv-cut-handle";
                h.style.left = (s.end / this.duration) * 100 + "%";
                h.dataset.cutIdx = idx; // Store index for direct updates

                // Time Label
                const tl = document.createElement("div");
                tl.className = "ltxv-time-label";
                tl.innerText = s.end.toFixed(1) + "s";
                h.appendChild(tl);

                h.onmousedown = (e) => this.startDragCut(e, idx, h);
                this.track.appendChild(h);
            }
        });

        this.renderProps();
    }

    // --- DRAGGING HANDLERS ---

    startDragCut(e, idx, handleEl) {
        e.preventDefault();
        e.stopPropagation();
        if (e.altKey) {
            this.mergeScenes(idx);
            return;
        }
        // Store references for direct DOM updates
        const leftScene = this.scenes[idx];
        const rightScene = this.scenes[idx + 1];
        const leftEl = this.track.querySelectorAll('.ltxv-scene-block')[idx];
        const rightEl = this.track.querySelectorAll('.ltxv-scene-block')[idx + 1];
        const timeLabel = handleEl.querySelector('.ltxv-time-label');

        this.dragState = { type: 'cut', idx, handleEl, leftScene, rightScene, leftEl, rightEl, timeLabel };
        this.setupDragListeners();
    }

    startDragMid(e, scene, mf) {
        e.preventDefault();
        e.stopPropagation();
        // Find the marker element
        const markers = this.track.querySelectorAll('.ltxv-mid-marker');
        let markerEl = null;
        markers.forEach(m => {
            const tl = m.querySelector('.ltxv-time-label');
            if (tl && Math.abs(parseFloat(tl.innerText) - mf.time) < 0.01) markerEl = m;
        });
        const timeLabel = markerEl ? markerEl.querySelector('.ltxv-time-label') : null;
        this.dragState = { type: 'mid', scene, mf, markerEl, timeLabel };
        this.setupDragListeners();
    }

    setupDragListeners() {
        document.body.style.cursor = 'ew-resize';
        const move = (e) => {
            const rect = this.track.getBoundingClientRect();
            let x = e.clientX - rect.left;
            let time = (x / rect.width) * this.duration;

            if (this.dragState.type === 'cut') {
                const { leftScene, rightScene, handleEl, leftEl, rightEl, timeLabel } = this.dragState;
                const min = leftScene.start + 0.2;
                const max = rightScene.end - 0.2;
                if (time < min) time = min;
                if (time > max) time = max;

                // Update data model
                leftScene.end = time;
                rightScene.start = time;

                // Update DOM directly (no full render)
                const pct = (time / this.duration) * 100;
                handleEl.style.left = pct + '%';
                if (timeLabel) timeLabel.innerText = time.toFixed(1) + 's';

                // Update scene block widths
                if (leftEl) {
                    leftEl.style.width = ((leftScene.end - leftScene.start) / this.duration) * 100 + '%';
                }
                if (rightEl) {
                    rightEl.style.left = pct + '%';
                    rightEl.style.width = ((rightScene.end - rightScene.start) / this.duration) * 100 + '%';
                }
            } else if (this.dragState.type === 'mid') {
                const { scene, mf, markerEl, timeLabel } = this.dragState;
                if (time < scene.start + 0.1) time = scene.start + 0.1;
                if (time > scene.end - 0.1) time = scene.end - 0.1;
                mf.time = time;

                // Update DOM directly
                if (markerEl) markerEl.style.left = (time / this.duration) * 100 + '%';
                if (timeLabel) timeLabel.innerText = time.toFixed(1) + 's';
            }
        };
        const up = () => {
            document.body.style.cursor = '';
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
            this.dragState = null;
            // Final render to sync everything
            this.render();
        };
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
    }

    mergeScenes(idx) {
        const left = this.scenes[idx];
        const right = this.scenes[idx + 1];

        left.end = right.end;
        left.endImg = right.endImg;
        left.hardCutEnd = right.hardCutEnd;
        left.middleFrames = [...left.middleFrames, ...right.middleFrames];

        this.scenes.splice(idx + 1, 1);
        this.render();
    }

    handleSceneClick(e, s) {
        if (e.ctrlKey) {
            // Split
            const rect = this.track.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const t = (x / rect.width) * this.duration;
            if (t > s.start + 0.1 && t < s.end - 0.1) this.splitScene(s, t);
            e.stopPropagation();
            return;
        }
        if (e.altKey) {
            // Add Mid
            const rect = this.track.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const t = (x / rect.width) * this.duration;
            s.middleFrames.push({ id: 'mf_' + Math.random(), time: t, img: null });
            this.render();
            e.stopPropagation();
            return;
        }

        this.selection = { sceneId: s.id, frameId: null };
        this.render();
        this.renderProps();
        e.stopPropagation();
    }

    splitScene(s, time) {
        const idx = this.scenes.indexOf(s);
        const newS = this.createScene(time, s.end);
        newS.endImg = s.endImg; newS.hardCutEnd = s.hardCutEnd;

        s.end = time; s.endImg = null; s.hardCutEnd = false; // Linked by default

        // Move frames
        newS.middleFrames = s.middleFrames.filter(f => f.time > time);
        s.middleFrames = s.middleFrames.filter(f => f.time <= time);

        this.scenes.splice(idx + 1, 0, newS);
        this.render();
    }

    // --- PROPS ---

    renderProps() {
        this.props.innerHTML = "";
        const s = this.scenes.find(x => x.id === this.selection.sceneId);
        if (!s) {
            this.props.innerHTML = `<div style="text-align:center; color:#555; padding:40px;">Select a scene</div>`;
            return;
        }

        // Header
        const h = document.createElement("div");
        h.style = "color:#aaa; border-bottom:1px solid #333; margin-bottom:10px;";
        h.innerText = "SCENE PROPERTIES";
        this.props.appendChild(h);

        // Image Row
        const row = document.createElement("div");
        row.className = "ltxv-img-row";

        // Start Check
        const idx = this.scenes.indexOf(s);
        const hasPrev = idx > 0;

        // Toggle Start
        if (hasPrev) {
            row.appendChild(this.createToggle(s.hardCutStart, () => {
                s.hardCutStart = !s.hardCutStart;
                this.scenes[idx - 1].hardCutEnd = s.hardCutStart;
                this.renderProps();
            }));
        }

        // Start Img
        row.appendChild(this.createUploader("Start", s.startImg, 'start', s));

        // Mids
        s.middleFrames.sort((a, b) => a.time - b.time).forEach((mf, i) => {
            const u = this.createUploader(`Frame ${i + 1}`, mf.img, 'mid', s, mf);
            if (mf.id === this.selection.frameId) u.querySelector('.ltxv-upload-box').classList.add('highlighted');
            row.appendChild(u);
        });

        // End Img
        row.appendChild(this.createUploader("End", s.endImg, 'end', s));

        // Toggle End
        if (idx < this.scenes.length - 1) {
            row.appendChild(this.createToggle(s.hardCutEnd, () => {
                s.hardCutEnd = !s.hardCutEnd;
                this.scenes[idx + 1].hardCutStart = s.hardCutEnd;
                this.renderProps();
            }));
        }

        this.props.appendChild(row);

        // Prompt
        const pb = document.createElement("div");
        pb.innerHTML = `<label style="color:#aaa; display:block; margin-bottom:5px;">Prompt</label>`;
        const ta = document.createElement("textarea");
        ta.className = "ltxv-textarea";
        ta.value = s.prompt;
        ta.oninput = (e) => { s.prompt = e.target.value; }; // No render needed
        ta.addEventListener('keydown', (e) => e.stopPropagation()); // Prevent ComfyUI shortcuts
        pb.appendChild(ta);
        this.props.appendChild(pb);
    }

    createUploader(lbl, url, type, scene, mf) {
        const d = document.createElement("div");
        d.className = "ltxv-uploader";
        d.innerHTML = `<span class="ltxv-scene-label">${lbl}</span>`;

        const box = document.createElement("div");
        box.className = "ltxv-upload-box";
        if (url) {
            // Check if full path or filename
            // If just filename, we might need /view?filename=...
            // Or assumes it's in input
            let src = url;
            if (!url.startsWith('http') && !url.startsWith('blob')) {
                src = `/view?filename=${url}&type=input`;
            }
            box.innerHTML = `<img src="${src}">`;
        } else {
            box.innerHTML = `<span>+</span>`;
        }

        box.onclick = () => {
            this.currentUploadTarget = { type, scene, mf };
            this.fileInput.click();
        };

        d.appendChild(box);
        return d;
    }

    createToggle(isHard, onClick) {
        const d = document.createElement("div");
        d.className = `ltxv-link-toggle ${isHard ? 'unlinked' : 'linked'}`;
        d.innerHTML = `<div>${isHard ? '⍁' : '∞'}</div><div style="font-size:8px;">${isHard ? 'HARD' : 'LINK'}</div>`;
        d.onclick = onClick;
        return d;
    }

    async handleFileUpload(file) {
        if (!file || !this.currentUploadTarget) return;

        // 1. Upload to ComfyUI
        const body = new FormData();
        body.append("image", file);
        body.append("overwrite", "true");

        try {
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            const data = await resp.json();
            // Expected: { name: "filename.png", subfolder: "", type: "input" }

            const filename = data.name;
            const { type, scene, mf } = this.currentUploadTarget;

            if (type === 'start') {
                scene.startImg = filename;
                // Sync prev?
                const idx = this.scenes.indexOf(scene);
                if (!scene.hardCutStart && idx > 0) {
                    this.scenes[idx - 1].endImg = filename;
                }
            } else if (type === 'end') {
                scene.endImg = filename;
                const idx = this.scenes.indexOf(scene);
                if (!scene.hardCutEnd && idx < this.scenes.length - 1) {
                    this.scenes[idx + 1].startImg = filename;
                }
            } else if (type === 'mid') {
                mf.img = filename;
            }

            this.renderProps();

        } catch (e) {
            console.error("Upload failed", e);
            alert("Upload failed: " + e.message);
        }

        this.fileInput.value = "";
    }

    setupPanning(container) {
        let isDown = false;
        let startX, scrollLeft;

        container.onmousedown = (e) => {
            if (e.target.closest(".ltxv-scene-block") || e.target.closest(".ltxv-mid-marker") || e.target.closest(".ltxv-cut-handle")) return;
            isDown = true;
            startX = e.pageX - container.offsetLeft;
            scrollLeft = container.scrollLeft;
            container.style.cursor = 'grabbing';
        };
        container.onmouseleave = () => { isDown = false; container.style.cursor = 'grab'; };
        container.onmouseup = () => { isDown = false; container.style.cursor = 'grab'; };
        container.onmousemove = (e) => {
            if (!isDown) return;
            e.preventDefault();
            const x = e.pageX - container.offsetLeft;
            const walk = (x - startX);
            container.scrollLeft = scrollLeft - walk;
        };
    }
}

// --- EXTENSION REGISTRATION ---

app.registerExtension({
    name: "ErosDiffusion.LTXVTimelineEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "LTXVTimelineEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                injectStyle();
                const node = this;
                const widget = this.widgets.find(w => w.name === "script");

                if (!widget) return;

                this.addWidget("button", "Open Visual Editor", null, () => {
                    if (!node.editor) {
                        node.editor = new LTXVTimelineEditor(widget, (val) => {
                            widget.value = val;
                        });
                    }
                    node.editor.open();
                });
            };
        }
    }
});
