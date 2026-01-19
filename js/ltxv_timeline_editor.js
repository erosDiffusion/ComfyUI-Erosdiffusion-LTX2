import { app } from "../../scripts/app.js";

// Styles
const STYLE = `
.ltxv-timeline-editor {
    background: #222;
    padding: 10px;
    border-radius: 8px;
    color: #ddd;
    font-family: sans-serif;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 600px;
    overflow-y: auto;
}
.ltxv-chunk {
    background: #333;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    position: relative;
}
.ltxv-chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ltxv-chunk-title {
    font-weight: bold;
    color: #aaddff;
}
.ltxv-row {
    display: flex;
    gap: 8px;
    align-items: center;
}
.ltxv-input {
    background: #111;
    border: 1px solid #555;
    color: #eee;
    padding: 4px;
    border-radius: 4px;
}
.ltxv-input-sm {
    width: 60px;
}
.ltxv-btn {
    background: #444;
    border: none;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
}
.ltxv-btn:hover { background: #555; }
.ltxv-btn-danger { background: #844; }
.ltxv-btn-danger:hover { background: #a55; }
.ltxv-btn-primary { background: #268; width: 100%; padding: 8px;}
.ltxv-btn-primary:hover { background: #37a; }

.ltxv-guides {
    border-top: 1px solid #444;
    padding-top: 5px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.ltxv-guide-item {
    background: #2a2a2a;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    gap: 6px;
    align-items: center;
}
.ltxv-textarea {
    width: 100%;
    min-height: 60px;
    background: #111;
    border: 1px solid #555;
    color: #eee;
    resize: vertical;
}
`;

function injectStyle() {
    if (!document.getElementById("ltxv-editor-style")) {
        const style = document.createElement("style");
        style.id = "ltxv-editor-style";
        style.innerHTML = STYLE;
        document.head.appendChild(style);
    }
}

// State Management
class TimelineState {
    constructor(initialString, onChange) {
        this.chunks = [];
        this.onChange = onChange;
        this.parse(initialString);
    }

    parse(str) {
        this.chunks = [];
        // Very basic parsing for demo. Robust parsing would be complex.
        // We will assume if it starts with '(', we try to match regex.
        // Or we just default to one empty chunk if empty.

        // Regex to match chunks: (duration: "prompt" { ... })
        // This is hard. We'll start fresh if parsing fails, or try simple split.
        // For MVP, we might overwrite existing value if structure is too complex?
        // Let's try to be non-destructive: If empty, add default.
        if (!str || str.trim() === "") {
            this.addChunk();
            return;
        }

        // Simple Regex for chunks? (?: *\()([^)]+)(?:\))
        // This breaks on nested parens in prompt.
        // We will just initialize with one chunk if parse invalid.
        // Or just let user build from scratch.
    }

    addChunk() {
        this.chunks.push({
            id: Date.now(),
            duration: 2.0,
            transition: "blend",
            prompt: "",
            guides: []
        });
        this.notify();
    }

    removeChunk(index) {
        this.chunks.splice(index, 1);
        this.notify();
    }

    updateChunk(index, key, value) {
        this.chunks[index][key] = value;
        this.notify();
    }

    addGuide(chunkIndex) {
        this.chunks[chunkIndex].guides.push({
            timeType: "start", // start, end, custom
            timeVal: 0.0,
            imageIdx: 0,
            strength: 1.0
        });
        this.notify();
    }

    removeGuide(chunkIndex, guideIndex) {
        this.chunks[chunkIndex].guides.splice(guideIndex, 1);
        this.notify();
    }

    updateGuide(chunkIndex, guideIndex, key, value) {
        this.chunks[chunkIndex].guides[guideIndex][key] = value;
        this.notify();
    }

    notify() {
        this.onChange(this.serialize());
    }

    serialize() {
        // (2.0s: "Prompt" { guides }) + ( ... )
        return this.chunks.map(c => {
            let guideStr = "";
            if (c.guides.length > 0) {
                const gParts = c.guides.map(g => {
                    let time = "0.0s";
                    if (g.timeType === "start") time = "0.0s";
                    else if (g.timeType === "end") time = "end";
                    else time = `${g.timeVal}s`;
                    return `${time}: $${g.imageIdx}:${g.strength}`;
                });
                guideStr = ` { ${gParts.join(", ")} }`;
            }
            // Transition format: 2.0s -> "cut" syntax supported?
            // Existing parser: (duration: "prompt")
            // Transition is separate? 
            // My Parser: "+ (2.0s ...)" means Blend. 
            // "|" means Cut.
            // So we need to control the joining character.

            // Wait, my parser expects joining characters BETWEEN chunks.
            // But chunks array is linear.
            // First chunk joins nothing.
            // Second chunk joins First?
            // We should store 'transitionToNext' or 'transitionFromPrev'?
            // Parser: `Chunk1 + Chunk2`. The '+' belongs to the join.
            // Let's store `transition` on the chunk as "how it joins PREVIOUS".
            // Chunk 0 has no transition.

            const content = `(${c.duration}s: "${c.prompt.replace(/"/g, '\\"')}"${guideStr})`;
            return { content, transition: c.transition };
        }).reduce((acc, curr, idx) => {
            if (idx === 0) return curr.content;
            const op = curr.transition === "cut" ? " | " : " + ";
            return acc + op + curr.content;
        }, "");
    }
}

app.registerExtension({
    name: "ErosDiffusion.LTXVTimelineEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LTXVTimelineEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                injectStyle();

                // Find script widget
                const scriptWidget = this.widgets.find(w => w.name === "script");
                if (scriptWidget) {
                    scriptWidget.inputEl.style.display = "none"; // Hide default textarea
                    // Keep it for serialization
                }

                // Create Container
                const container = document.createElement("div");
                container.className = "ltxv-timeline-editor";

                // Add to DOMWidget
                const domWidget = this.addDOMWidget("timeline_ui", "ui", container, {
                    getValue() { return scriptWidget.value; },
                    setValue(v) { scriptWidget.value = v; },
                });

                // Logic
                const state = new TimelineState(scriptWidget.value, (newStr) => {
                    scriptWidget.value = newStr;
                    // Trigger graph update if needed?
                });

                // Render Function
                function render() {
                    container.innerHTML = "";

                    state.chunks.forEach((chunk, cIdx) => {
                        const card = document.createElement("div");
                        card.className = "ltxv-chunk";

                        // Header
                        const header = document.createElement("div");
                        header.className = "ltxv-chunk-header";

                        const title = document.createElement("span");
                        title.className = "ltxv-chunk-title";
                        title.innerText = `Chunk ${cIdx + 1}`;
                        header.appendChild(title);

                        // Delete Button
                        const delBtn = document.createElement("button");
                        delBtn.className = "ltxv-btn ltxv-btn-danger";
                        delBtn.innerText = "X";
                        delBtn.onclick = () => { state.removeChunk(cIdx); render(); };
                        header.appendChild(delBtn);

                        card.appendChild(header);

                        // Controls Row
                        const row = document.createElement("div");
                        row.className = "ltxv-row";

                        // Duration
                        const durLabel = document.createElement("label");
                        durLabel.innerText = "Dur(s):";
                        const durIn = document.createElement("input");
                        durIn.type = "number";
                        durIn.step = "0.1";
                        durIn.className = "ltxv-input ltxv-input-sm";
                        durIn.value = chunk.duration;
                        durIn.onchange = (e) => state.updateChunk(cIdx, "duration", parseFloat(e.target.value));
                        row.appendChild(durLabel);
                        row.appendChild(durIn);

                        // Transition (If not first)
                        if (cIdx > 0) {
                            const transLabel = document.createElement("label");
                            transLabel.innerText = "Trans:";
                            const transSel = document.createElement("select");
                            transSel.className = "ltxv-input";
                            ["blend", "cut"].forEach(opt => {
                                const o = document.createElement("option");
                                o.value = opt;
                                o.innerText = opt;
                                if (opt === chunk.transition) o.selected = true;
                                transSel.appendChild(o);
                            });
                            transSel.onchange = (e) => state.updateChunk(cIdx, "transition", e.target.value);
                            row.appendChild(transLabel);
                            row.appendChild(transSel);
                        }

                        card.appendChild(row);

                        // Prompt
                        const promptArea = document.createElement("textarea");
                        promptArea.className = "ltxv-textarea";
                        promptArea.placeholder = "Prompt...";
                        promptArea.value = chunk.prompt;
                        promptArea.onchange = (e) => state.updateChunk(cIdx, "prompt", e.target.value);
                        card.appendChild(promptArea);

                        // Guides
                        const guidesDiv = document.createElement("div");
                        guidesDiv.className = "ltxv-guides";

                        chunk.guides.forEach((g, gIdx) => {
                            const gRow = document.createElement("div");
                            gRow.className = "ltxv-guide-item";

                            // Type
                            const tSel = document.createElement("select");
                            tSel.className = "ltxv-input";
                            ["start", "end", "time"].forEach(o => {
                                const op = document.createElement("option");
                                op.value = o;
                                op.innerText = o;
                                if (o === g.timeType) op.selected = true;
                                tSel.appendChild(op);
                            });
                            tSel.onchange = (e) => {
                                state.updateGuide(cIdx, gIdx, "timeType", e.target.value);
                                render(); // Re-render to show/hide timeVal
                            };
                            gRow.appendChild(tSel);

                            // Time Value
                            if (g.timeType === "time") {
                                const tVal = document.createElement("input");
                                tVal.type = "number";
                                tVal.step = "0.1";
                                tVal.className = "ltxv-input ltxv-input-sm";
                                tVal.value = g.timeVal;
                                tVal.onchange = (e) => state.updateGuide(cIdx, gIdx, "timeVal", parseFloat(e.target.value));
                                gRow.appendChild(tVal);
                            }

                            // Image Index
                            const imgLabel = document.createElement("span");
                            imgLabel.innerText = "Img $";
                            gRow.appendChild(imgLabel);

                            const imgIn = document.createElement("input");
                            imgIn.type = "number";
                            imgIn.min = "0";
                            imgIn.className = "ltxv-input ltxv-input-sm";
                            imgIn.value = g.imageIdx;
                            imgIn.style.width = "40px";
                            imgIn.onchange = (e) => state.updateGuide(cIdx, gIdx, "imageIdx", parseInt(e.target.value));
                            gRow.appendChild(imgIn);

                            // Strength
                            const strIn = document.createElement("input");
                            strIn.type = "range";
                            strIn.min = "0";
                            strIn.max = "1";
                            strIn.step = "0.05";
                            strIn.value = g.strength;
                            strIn.onchange = (e) => state.updateGuide(cIdx, gIdx, "strength", parseFloat(e.target.value));
                            gRow.appendChild(strIn);

                            // Remove Guide
                            const remG = document.createElement("button");
                            remG.className = "ltxv-btn ltxv-btn-danger";
                            remG.innerText = "x";
                            remG.onclick = () => { state.removeGuide(cIdx, gIdx); render(); };
                            gRow.appendChild(remG);

                            guidesDiv.appendChild(gRow);
                        });

                        // Add Guide Btn
                        const addGBtn = document.createElement("button");
                        addGBtn.className = "ltxv-btn";
                        addGBtn.innerText = "+ Add Guide";
                        addGBtn.onclick = () => { state.addGuide(cIdx); render(); };
                        guidesDiv.appendChild(addGBtn);

                        card.appendChild(guidesDiv);
                        container.appendChild(card);
                    });

                    // Add Chunk Btn
                    const addCBtn = document.createElement("button");
                    addCBtn.className = "ltxv-btn ltxv-btn-primary";
                    addCBtn.innerText = "+ Add Scene Chunk";
                    addCBtn.onclick = () => { state.addChunk(); render(); };
                    container.appendChild(addCBtn);
                }

                render();
            };
        }
    }
});
