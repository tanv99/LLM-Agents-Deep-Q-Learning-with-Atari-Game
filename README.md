# Deep Q-Learning with Atari Boxing

Implementation of Deep Q-Network (DQN) agent for Atari Boxing using Stable-Baselines3. Five experiments testing different hyperparameters with complete analysis and visualization.

## Quick Info

- **Environment:** ALE/Boxing-v5
- **Algorithm:** Deep Q-Learning with CNN policy
- **Training:** 500,000 steps per experiment
- **Best Result:** 26.25 ± 10.81 test reward (Experiment 4)

## Video Demonstration

Watch the trained agent in action:
**[🎥 Watch Trained Agent Gameplay](https://github.com/tanv99/LLM-Agents-Deep-Q-Learning-with-Atari-Game/blob/main/boxing_agent.mp4)**

The video showcases the best-performing agent (Experiment 4: Moderate Epsilon Decay) demonstrating learned boxing strategies including defensive positioning, counter-punching, and ring control.

> **Note:** To include the video in your repository:
> 1. Upload `boxing_agent.mp4` to your repository root or a `videos/` folder
> 2. Update the link above to match the file location
> 3. Alternatively, upload to YouTube/Google Drive and replace with that link

## Requirements
```
Python 3.7+
gymnasium[atari,accept-rom-license]
stable-baselines3[extra]
ale-py
opencv-python
matplotlib
numpy
```

## Setup Steps

### 1. Open in Google Colab

Upload the notebook to Google Colab or open directly from GitHub.

### 2. Mount Google Drive (Required)

Run the first code cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

This creates the directory structure:
- `boxing_dqn_assignment/weights/` - Trained models
- `boxing_dqn_assignment/results/` - JSON results
- `boxing_dqn_assignment/plots/` - Visualizations

### 3. Install Dependencies

Run the installation cell (installs automatically):
```bash
!pip install -q gymnasium "gymnasium[atari,accept-rom-license]" ale-py stable-baselines3[extra] opencv-python
```

### 4. Run Experiments

**Option A: Load Existing Results**

Skip training cells, run only the analysis section to view saved results.

**Option B: Re-run Training**

Execute experiment cells sequentially:
- Experiment 1: Baseline
- Experiment 2: Higher Learning Rate
- Experiment 3: Lower Gamma
- Experiment 4: Moderate Epsilon (best performer)
- Experiment 5: Boltzmann Policy

**Option C: Run Selected Experiments**

Execute only specific experiment cells of interest.

### 5. Generate Visualizations

Run the "Load and Analyze Results" section to create:
- Comparative performance plots
- Training curves
- Results summary table

### 6. Create Video Demo (Optional)

Run the "Video Demonstration" cell to generate gameplay video.

## File Structure
```
boxing_dqn_assignment/
├── weights/
│   ├── 01_baseline.zip
│   ├── 02_moderate_lr.zip
│   ├── 03_gamma_0.8.zip
│   ├── 04_moderate_epsilon.zip
│   └── 05_boltzmann_policy.pth
├── results/
│   ├── 01_baseline.json
│   ├── 02_moderate_lr.json
│   ├── 03_gamma_0.8.json
│   ├── 04_moderate_epsilon.json
│   └── 05_boltzmann_policy.json
└── plots/
    ├── analysis_[timestamp].png
    └── boxing_demo.mp4
```

## Key Results

| Experiment | Test Reward | Episodes |
|-----------|-------------|----------|
| 1. Baseline | 22.60 ± 8.48 | 1,118 |
| 2. Higher LR | 1.15 ± 5.96 | 1,118 |
| 3. Lower Gamma | 15.75 ± 14.19 | 1,124 |
| 4. Moderate ε | **26.25 ± 10.81** | 1,118 |
| 5. Boltzmann | -3.50 ± 12.27 | 1,118 |

**Finding:** Moderate epsilon decay (60% duration, 5% final ε) outperforms standard schedule.

## Conclusion

This Deep Q-Learning implementation successfully demonstrates reinforcement learning fundamentals through systematic hyperparameter experimentation. The baseline agent achieved competent Boxing performance (22.60 test reward) through environment interaction, learning effective strategies from sparse rewards (+1/-1 for punches landed/received).

**Key Findings:**

- **Learning Rate Sensitivity:** Increasing α from 1e-4 to 2.5e-4 degraded performance (1.15 reward), demonstrating neural network Q-learning requires conservative learning rates for stability
- **Discount Factor Impact:** Lowering γ from 0.99 to 0.8 reduced performance (15.75 reward), proving that myopic planning (5-step lookahead) is insufficient for multi-step boxing strategy
- **Exploration Optimization:** Moderate epsilon decay (60% duration, 5% final) achieved best results (26.25 reward), outperforming standard schedule by maintaining higher residual exploration
- **Policy Comparison:** Boltzmann exploration failed catastrophically (-3.50 reward), validating that epsilon-greedy's discrete explore/exploit is superior for sparse reward environments

**Validated Predictions:** Results align with Mnih et al. (2015) recommendations, confirming α=1e-4 and γ=0.99 are well-calibrated for Atari while revealing opportunities for exploration schedule refinement.

**Broader Impact:** This work illustrates complementary strengths of traditional RL and modern LLM agents. DQN excels at precise reactive control through environmental interaction, while LLM agents provide linguistic reasoning and knowledge generalization. Future AI systems may synthesize these approaches: LLM strategic planning guiding DQN tactical execution.

## Hardware Requirements

- **GPU:** Recommended (T4 or better in Colab)
- **RAM:** 12GB+ for training
- **Storage:** ~500MB Google Drive space

## Troubleshooting

**Issue:** "Module not found" errors  
**Fix:** Re-run installation cell

**Issue:** Training too slow  
**Fix:** Ensure GPU runtime (Runtime → Change runtime type → GPU)

**Issue:** Session timeout during training  
**Fix:** Results auto-save to Google Drive; resume from next experiment

## Citation

If you use this work, please cite:
```bibtex
@misc{boxing_dqn_2025,
  author = {Tanvi Inchanalkar},
  title = {Deep Q-Learning with Atari Boxing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tanv99/LLM-Agents-Deep-Q-Learning-with-Atari-Game}
}
```

## License

MIT License - See notebook for full text.

## References

- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium

---

**Quick Start:** Mount Drive → Install dependencies → Run analysis cells to view results.
