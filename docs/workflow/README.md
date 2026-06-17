# NeSy RL Workflow Diagrams

This folder contains visual workflow assets for the data flow of this repository.

## Files

- `nesy_data_flow.mmd`: full repository data flow from config/env/specs to datasets, TT/DT, evaluation, artifacts, sweeps, suites, and figures.
- `tt_train_eval_flow.mmd`: detailed Trajectory Transformer training and evaluation flow.
- `dt_train_eval_flow.mmd`: detailed Decision Transformer training and evaluation flow.
- `tt_architecture.mmd`: exact TT model architecture, inputs, internal transformations, outputs, and loss.
- `dt_architecture.mmd`: exact DT model architecture, inputs, internal transformations, outputs, and loss.
- `logic_dfa_activity.puml`: PlantUML activity diagram for the LTLf/DFA/DeepDFA path.
- `MODEL_ARCHITECTURES.md`: prose explanation of TT vs DT input/output behavior.

Rendered outputs are generated next to each source:

- Mermaid: `.svg`, `.png`, `.pdf`
- PlantUML: `.svg`, `.png`

## Render With Mermaid CLI

From repo root:

```bash
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/nesy_data_flow.mmd -o docs/workflow/nesy_data_flow.svg
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/nesy_data_flow.mmd -o docs/workflow/nesy_data_flow.png -s 3
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/nesy_data_flow.mmd -o docs/workflow/nesy_data_flow.pdf

PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/tt_train_eval_flow.mmd -o docs/workflow/tt_train_eval_flow.svg
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/dt_train_eval_flow.mmd -o docs/workflow/dt_train_eval_flow.svg
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/tt_architecture.mmd -o docs/workflow/tt_architecture.svg
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome mmdc -i docs/workflow/dt_architecture.mmd -o docs/workflow/dt_architecture.svg
```

## Mermaid Live Editor

Use this for quick preview/export:

1. Open the Mermaid Live Editor in a browser.
2. Open one `.mmd` file from this folder.
3. Paste the text into the editor.
4. Export as SVG or PNG.

Best starting file: `nesy_data_flow.mmd`.

## diagrams.net / draw.io

Use this when you want to manually style the workflow for slides:

1. Open diagrams.net / draw.io.
2. Choose `Insert -> Advanced -> Mermaid`.
3. Paste the contents of any `.mmd` file.
4. Arrange, recolor, add swimlanes, or split the diagram into slide-sized chunks.
5. Export as SVG or PNG.

Recommended slide split:

- Slide 1: `nesy_data_flow.mmd`
- Slide 2: `tt_train_eval_flow.mmd`
- Slide 3: `dt_train_eval_flow.mmd`
- Slide 4: `tt_architecture.mmd`
- Slide 5: `dt_architecture.mmd`
- Slide 6: `logic_dfa_activity.puml` rendered SVG

## Render With PlantUML

From repo root:

```bash
plantuml -tsvg docs/workflow/logic_dfa_activity.puml
plantuml -tpng docs/workflow/logic_dfa_activity.puml
```

## How To Read The Flow

The central data path is:

```text
config/env/spec -> offline episodes -> token schema -> model dataset
spec/formula -> adapter -> DFA -> DeepDFA
model + logic -> training/evaluation -> metrics/artifacts -> analysis/figures
```

Important distinction:

- TT logic uses DeepDFA over sampled token sequences.
- DT logic currently uses action probabilities plus transition dynamics to estimate short-horizon hazard risk.
