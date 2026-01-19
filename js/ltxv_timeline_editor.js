import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- STYLES ---
const STYLE = `
.ltxv-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.8); z-index: 10000; display: none;
    align-items: center; justify-content: center;
}
.ltxv-modal {
    width: 90%; height: 80%; background: #1e1e1e; border: 1px solid #444;
    border-radius: 8px; display: flex; flex-direction: column;
    box-shadow: 0 0 20px rgba(0,0,0,0.5); font-family: sans-serif; color: #ddd;
}
.ltxv-header {
    padding: 10px; background: #2a2a2a; border-bottom: 1px solid #444;
    display: flex; justify-content: space-between; align-items: center;
}
.ltxv-body {
    flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative;
}
.ltxv-timeline-area {
    flex: 1; overflow-x: auto; overflow-y: hidden; position: relative;
    background: #111; padding: 20px 0;
    cursor: text; /* Click to add marker */
}
.ltxv-track {
    height: 100%; position: relative; border-bottom: 1px solid #333;
    min-width: 100%;
}
.ltxv-chunk-block {
    position: absolute; top: 10px; height: 60px;
    background: #2b3a42; border: 1px solid #4a6fa5;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: #fff; overflow: hidden;
    white-space: nowrap; cursor: pointer;
}
.ltxv-chunk-block.selected { background: #3a4b55; border-color: #61afef; }

.ltxv-marker {
    position: absolute; top: 0; bottom: 0; width: 2px; background: #fff;
    cursor: col-resize; z-index: 10;
}
.ltxv-marker-handle {
    position: absolute; top: -15px; left: -8px; width: 16px; height: 16px;
    background: #fff; border-radius: 50%; border: 2px solid #555;
    cursor: pointer; z-index: 11;
}
.ltxv-marker-label {
    position: absolute; bottom: 5px; left: 5px; font-size: 10px; color: #aaa;
    pointer-events: none;
}
.ltxv-marker-img-slot {
    position: absolute; top: 80px; left: -20px; width: 40px; height: 40px;
    background: #000; border: 1px dashed #555; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #555; cursor: pointer; z-index: 12;
    overflow: hidden;
}
.ltxv-marker-img-slot:hover { border-color: #fff; color: #fff; }
.ltxv-marker-img-slot img { width: 100%; height: 100%; object-fit: cover; }
.ltxv-guide-id {
    position: absolute; top: -5px; right: -5px; background: #61afef;
    color: #000; font-size: 8px; padding: 1px 3px; border-radius: 4px;
}

.ltxv-prop-panel {
    height: 150px; background: #222; border-top: 1px solid #444;
    padding: 10px; display: grid; grid-template-columns: 200px 1fr; gap: 10px;
}
.ltxv-prop-group { display: flex; flex-direction: column; gap: 5px; }
.ltxv-input { background: #111; border: 1px solid #555; color: #eee; padding: 4px; }
.ltxv-textarea { flex: 1; resize: none; background: #111; border: 1px solid #555; color: #eee; }
`;

function injectStyle() {
    if (!document.getElementById("ltxv-v2-style")) {
        const s = document.createElement("style");
        s.id = "ltxv-v2-style";
        s.innerHTML = STYLE;
        document.head.appendChild(s);
    }
}

// --- LOGIC ---

class TimelineEditorV2 {
    constructor(widget, onChange) {
        this.widget = widget;
        this.onChange = onChange;
        this.chunks = []; // { duration, prompt, guides: [] }
        this.zoom = 50; // pixels per second
        this.selectedChunkIdx = -1;

        this.parse(widget.value);
        this.buildUI();
    }

    parse(str) {
        // Basic parser. If empty, create default.
        if (!str || str.trim().length < 5) {
            this.chunks = [{ duration: 2.0, prompt: "Scene 1", guides: [] }];
            return;
        }
        // TODO: Implement robust parser. For now, reset if invalid or try very basic split.
        // We will assume state is cleared for this "Prototype" rewrite.
        // Or keep existing string if parse fails.
    }

    serialize() {
        let t = 0.0;
        return this.chunks.map(c => {
            let guideStr = "";
            if (c.guides && c.guides.length) {
                guideStr = " { " + c.guides.map(g => {
                    // Determine relative time
                    // Simplified: Just output image refs for now, assume Start/End logic
                    // Or full syntax: 0.0s: $0:1.0
                    return `${g.time.toFixed(1)}s: $${g.imgIdx}:${g.strength}`;
                }).join(", ") + " }";
            }
            const s = `(${c.duration.toFixed(1)}s: "${c.prompt.replace(/"/g, '\\"')}"${guideStr})`;
            t += c.duration;
            return s;
        }).join(" + ");
    }

    buildUI() {
        this.overlay = document.createElement("div");
        this.overlay.className = "ltxv-overlay";

        const modal = document.createElement("div");
        modal.className = "ltxv-modal";

        // Header
        const header = document.createElement("div");
        header.className = "ltxv-header";
        header.innerHTML = `<span>LTXV Timeline Editor</span>`;
        const closeBtn = document.createElement("button");
        closeBtn.innerText = "Close & Save";
        closeBtn.onclick = () => {
            this.onChange(this.serialize());
            this.overlay.style.display = "none";
        };
        header.appendChild(closeBtn);
        modal.appendChild(header);

        // Body
        const body = document.createElement("div");
        body.className = "ltxv-body";

        // Timeline
        const area = document.createElement("div");
        area.className = "ltxv-timeline-area";
        this.track = document.createElement("div");
        this.track.className = "ltxv-track";
        area.appendChild(this.track);
        body.appendChild(area);

        // Properties
        this.props = document.createElement("div");
        this.props.className = "ltxv-prop-panel";
        body.appendChild(this.props);

        modal.appendChild(body);
        this.overlay.appendChild(modal);
        document.body.appendChild(this.overlay);

        this.renderTimeline();
        this.renderProps();
    }

    open() {
        this.overlay.style.display = "flex";
        this.renderTimeline();
    }

    renderTimeline() {
        this.track.innerHTML = "";
        let x = 0;

        this.chunks.forEach((chunk, i) => {
            const width = chunk.duration * this.zoom;

            // Block
            const el = document.createElement("div");
            el.className = "ltxv-chunk-block";
            if (i === this.selectedChunkIdx) el.classList.add("selected");
            el.style.left = x + "px";
            el.style.width = width + "px";
            el.innerText = chunk.prompt.substring(0, 20) + "...";
            el.onclick = (e) => {
                e.stopPropagation();
                this.selectedChunkIdx = i;
                this.renderTimeline();
                this.renderProps();
            };

            // Drag Handle (Right Edge)
            const handle = document.createElement("div");
            handle.className = "ltxv-marker-handle";
            handle.style.left = (width - 8) + "px";
            handle.style.top = "50%";
            handle.onmousedown = (e) => this.startDrag(e, i);
            el.appendChild(handle);

            this.track.appendChild(el);
            x += width;
        });

        this.track.style.width = (x + 200) + "px"; // Extra space
    }

    startDrag(e, idx) {
        e.stopPropagation();
        const startX = e.clientX;
        const startDur = this.chunks[idx].duration;

        const onMove = (mv) => {
            const diffPx = mv.clientX - startX;
            const diffSec = diffPx / this.zoom;
            let newDur = Math.max(0.5, startDur + diffSec); // Min 0.5s
            this.chunks[idx].duration = newDur;
            this.renderTimeline();
        };

        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
            this.renderProps(); // Update duration input
        };

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
    }

    renderProps() {
        this.props.innerHTML = "";
        if (this.selectedChunkIdx < 0) {
            this.props.innerText = "Select a chunk to edit.";
            return;
        }

        const c = this.chunks[this.selectedChunkIdx];

        // Duration
        const grp1 = document.createElement("div");
        grp1.className = "ltxv-prop-group";
        grp1.innerHTML = `<label>Duration (s)</label>`;
        const durIn = document.createElement("input");
        durIn.type = "number"; durIn.step = "0.1"; durIn.className = "ltxv-input";
        durIn.value = c.duration.toFixed(1);
        durIn.onchange = (e) => {
            c.duration = parseFloat(e.target.value);
            this.renderTimeline();
        };
        grp1.appendChild(durIn);
        this.props.appendChild(grp1);

        // Prompt
        const grp2 = document.createElement("div");
        grp2.className = "ltxv-prop-group";
        grp2.style.flex = "1";
        grp2.innerHTML = `<label>Prompt</label>`;
        const pIn = document.createElement("textarea");
        pIn.className = "ltxv-textarea";
        pIn.value = c.prompt;
        pIn.onchange = (e) => {
            c.prompt = e.target.value;
            this.renderTimeline();
        };
        grp2.appendChild(pIn);
        this.props.appendChild(grp2);

        // Add Chunk Btn
        const grp3 = document.createElement("div");
        grp3.className = "ltxv-prop-group";
        const addBtn = document.createElement("button");
        addBtn.innerText = "Add After";
        addBtn.onclick = () => {
            this.chunks.splice(this.selectedChunkIdx + 1, 0, {
                duration: 2.0, prompt: "New Scene", guides: []
            });
            this.renderTimeline();
        };
        grp3.appendChild(addBtn);
        this.props.appendChild(grp3);
    }
}

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

                // Create Button to Open Editor
                this.addWidget("button", "Open Visual Editor", null, () => {
                    if (!node.editor) {
                        node.editor = new TimelineEditorV2(widget, (v) => {
                            widget.value = v;
                        });
                    }
                    node.editor.open();
                });
            };
        }
    }
});
