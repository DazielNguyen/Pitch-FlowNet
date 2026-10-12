"""Count checkpoint parameters with explicit, source-checked exclusions."""
import collections
import json
from pathlib import Path

source = Path(__file__).with_name('question2_checkpoint_shapes_2026-09-23.json')
result = []
for item in json.loads(source.read_text(encoding='utf-16')):
    tensors = item['tensors']
    # WN._n_channels is a registered integer buffer, not an nn.Parameter.
    buffers = [t for t in tensors if t['name'].endswith('._n_channels')]
    params = [t for t in tensors if not t['name'].endswith('._n_channels')]
    groups = collections.Counter()
    for t in params:
        groups[t['name'].split('.')[1]] += t['numel']
    retained = [t for t in params if not t['name'].startswith('model_g.enc_q.')]
    # Reverse SDP does not run posterior branches or its first ConvFlow.
    unused_dp = [t for t in retained if t['name'].startswith(('model_g.dp.post_', 'model_g.dp.flows.1.'))]
    dec_g = [t for t in retained if t['name'].startswith('model_g.dec.') and t['name'].endswith('.weight_g')]
    result.append({
        'checkpoint': item['checkpoint'], 'parameters_by_generator_module': dict(groups),
        'excluded_buffer_elements': sum(t['numel'] for t in buffers),
        'generator_total': sum(groups.values()),
        'generator_excluding_enc_q': sum(t['numel'] for t in retained),
        'decoder_weight_norm_g_elements': sum(t['numel'] for t in dec_g),
        'unused_reverse_sdp_parameters': sum(t['numel'] for t in unused_dp),
        'inference_path_after_decoder_weight_norm_removal':
            sum(t['numel'] for t in retained) - sum(t['numel'] for t in unused_dp) - sum(t['numel'] for t in dec_g),
        'unused_reverse_sdp_keys': [t['name'] for t in unused_dp],
    })
print(json.dumps(result, indent=2))
