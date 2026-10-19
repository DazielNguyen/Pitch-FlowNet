"""Camera-ready training protocol for the SOICT 2026 PitchFlowNet experiments.

This package extends the historical candidate source (EdgeTTS badf381) without changing how the
historical checkpoints load. It provides:
- the split read by stable utterance ID from a manifest (independent of the training seed);
- separate model-initialisation, data-order and validation seeds;
- a deterministic validation metric aggregated exactly across DDP ranks;
- counters of generator/discriminator updates and samples, with a budget in generator updates;
- a ledger of every checkpoint decision.
"""
