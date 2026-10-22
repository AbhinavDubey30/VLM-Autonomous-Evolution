# Improving VLMs on Scientific Chart Analysis via Autonomous Evolution: code and evolution logs

Code and evolution logs for the paper
*Improving VLMs on Scientific Chart Analysis via Autonomous Evolution*.

Authors: Abhinav Sudhakar Dubey, Vidushi Goel, Kiran Rana (University of California, Santa Cruz),
Shikhar Mathur (Airbnb Inc.) and Vaggos Chatziafratis (University of California, Santa Cruz).

This repository contains the seed programs, the evolved programs, and the per-generation logs for
all twelve evolution runs in the paper (three VLMs × four chart question-answering benchmarks), one
Colab notebook per run. Nothing here has been edited after the runs finished; the only changes from
the working folder are the directory layout and the removal of notebook-execution metadata.

## Layout

```
notebooks/
  chartqa/     evolve_chartqa_{qwen3vl2b,internvl3,smolvlm2}.ipynb
  charxiv/     evolve_charxiv_{qwen3vl2b,internvl3,smolvlm2}.ipynb
  evochartqa/  evolve_evochartqa_{qwen3vl2b,internvl3,smolvlm2}.ipynb
  plotqa/      evolve_plotqa_{qwen3vl2b,internvl3,smolvlm2}.ipynb
  superseded/  two earlier Qwen3-VL-2B runs (ChartQA, CharXiv) that were re-run; kept for completeness
```

### `notebooks/`: one notebook per evolution run

Each notebook is a self-contained Colab run of the evolution loop described in Sec. 3.3 of the
paper for one model–benchmark pair. It contains, in order: the seed program (generation 0: a plain
prompt, a generous token budget, no post-processing), the candidate contract (only the code between
`# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`, i.e. the function
`vlm_inference(image_path, question, qid) -> str`, is edited), the validator with its five
invariants (Table 2), the evaluator (descriptive exact match on the training window as fitness,
holdout window for selection, reasoning exact match and latency recorded but never used for
selection), the failure-aware mutation and crossover operators with Gemini 2.5 Flash as the mutator,
and the population manager. Every notebook ran 25 generations with a pool of at most ten programs,
crossover every five generations, and up to six failure examples per mutation prompt, on a Google
Colab A100. The cell outputs are the evolution logs: for every generation they print the parent, the
child's train and holdout accuracy, and the current best-by-holdout program, and the JSON history
described in Appendix B is written alongside.

Models: `qwen3vl2b` is Qwen3-VL-2B-Instruct, `internvl3` is InternVL3, `smolvlm2` is SmolVLM2.
All are run in float16 through HuggingFace Transformers with greedy decoding, with the model id
fixed by the validator.

## Running

The notebooks need a GPU runtime, a Hugging Face token for the model checkpoints, and a Gemini API
key for the mutator; both are requested interactively (Colab secrets or a prompt) and neither is
stored anywhere in this repository. Each notebook downloads its benchmark from the Hugging Face
Hub. All runs use greedy decoding with the model id fixed by the validator, so re-running a saved
program reproduces its outputs exactly. The evolved program of each run is printed in full in the
notebook's final cells and is also written to disk by the population manager as the best-by-holdout
candidate.

## Citation

```bibtex
@article{dubey2026improvingvlms,
  title   = {Improving VLMs on Scientific Chart Analysis via Autonomous Evolution},
  author  = {Dubey, Abhinav Sudhakar and Goel, Vidushi and Rana, Kiran and Mathur, Shikhar and Chatziafratis, Vaggos},
  journal = {arXiv preprint},
  year    = {2026}
}
```
