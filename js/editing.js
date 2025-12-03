const MANIPULATION_NUM_EXAMPLES = 15;

const MANIPULATION_PROMPTS = [
    "Change the character's T-pose to a waving gesture with one arm raised.",
    "Change the character's pose so she is sitting down with her hands in her lap.",
    "Change the character's pose from a T-pose to a fighting stance with fists raised.",
    "Change the robot's action so its claw hands are waving hello.",
    "Add a single blueberry placed on the peak of the white whipped cream swirl.",
    "Add a small wicker basket to the front of the scooter, below the handlebars.",
    "Add a window box full of red flowers beneath one of the front windows.",
    "Add a vintage-style luggage rack with a suitcase strapped to it on the cream roof.",
    "Add a small watering can on the ground next to the striped pot.",
    "Remove the skewer topped with a green olive from the sandwich.",
    "Remove the black handles from the top of the container.",
    "Remove the red ladder that is leaning against the structure.",
    "Replace the smooth, light gray cylindrical pot with a terracotta pot.",
    "Remove the two horizontal handles from the wicker basket.",
    "Replace the chocolate icing with strawberry-pink icing.",
];



var manipulation_items = [];
for (let i = 0; i < MANIPULATION_NUM_EXAMPLES; i++) {
    manipulation_items.push({
        index: i,
        name: `Example ${i}`,
        editedPrompt: MANIPULATION_PROMPTS[i] || ""
    });
}


function manipulation_carousel_item_template(item) {
    var i = item.index;

    return `
    <div class="x-card" style="padding: 16px; display: flex; flex-direction: column; gap: 12px;">
        <div class="x-handwriting">
            ${item.name}
        </div>

        <div style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 12px;">

            <!-- Source -->
            <div style="flex: 1 1 180px; display: flex; flex-direction: column; gap: 4px;">
                <div class="x-label">Source</div>
                <model-viewer
                    src="assets/${i}/src.glb"
                    camera-controls
                    auto-rotate
                    autoplay
                    exposure="1"
                    style="width: 100%; height: 220px; border-radius: 8px; background: #f5f5f5;">
                </model-viewer>
            </div>

            <!-- Baseline -->
            <div style="flex: 1 1 180px; display: flex; flex-direction: column; gap: 4px;">
                <div class="x-label">Baseline</div>
                <model-viewer
                    src="assets/${i}/baseline.glb"
                    camera-controls
                    auto-rotate
                    autoplay
                    exposure="1"
                    style="width: 100%; height: 220px; border-radius: 8px; background: #f5f5f5;">
                </model-viewer>
            </div>

            <!-- Ours -->
            <div style="flex: 1 1 180px; display: flex; flex-direction: column; gap: 4px;">
                <div class="x-label">Ours</div>
                <model-viewer
                    src="assets/${i}/ours.glb"
                    camera-controls
                    auto-rotate
                    autoplay
                    exposure="1"
                    style="width: 100%; height: 220px; border-radius: 8px; background: #f5f5f5;">
                </model-viewer>
            </div>

            <!-- Edited Image -->
            <div style="flex: 1 1 180px; display: flex; flex-direction: column; gap: 4px;">
                <div class="x-label">Edited Image</div>
                <img
                    src="assets/${i}/edited.png"
                    alt="Edited result ${i}"
                    style="width: 100%; height: 220px; border-radius: 8px; object-fit: cover; background: #f5f5f5;">
            </div>

            <!-- Edited Prompt -->
            <div class="edited-prompt-card">
                <span class="prompt-title"> Editing Prompt:</span>
                <span class="prompt-text">${item.editedPrompt}</span>
            </div>

        </div>
    </div>
    `;
}

function stripTextures(modelViewer) {
    if (!modelViewer || !modelViewer.model || !modelViewer.model.materials) return;

    const materials = modelViewer.model.materials;

    for (let i = 0; i < materials.length; i++) {
        const mat = materials[i];
        const pmr = mat.pbrMetallicRoughness;

        if (pmr.baseColorTexture) {
            pmr.baseColorTexture.setTexture(null);
        }

        pmr.setBaseColorFactor([1, 1, 1, 1]);


        if (pmr.metallicRoughnessTexture) {
            pmr.setMetallicRoughnessTexture(null);
        }
        pmr.setMetallicFactor(0.0);
        pmr.setRoughnessFactor(1.0);


        if (mat.normalTexture) {
            mat.normalTexture.setTexture(null);
        }
    }

    modelViewer.environmentImage = 'assets/env_maps/gradient.jpg';
    modelViewer.exposure = 5;
}

