## Plan: LSTM Temporal Aggregation (1 layer)

Introduce a new LSTM-based temporal aggregator (latent_dim in/out, 1 layer) in a new module so teacher/student keep interfaces but preserve segment order instead of mean pooling.

### Steps
1. Copy current slim classifier module to autoencoder_benchmark/models_slim_classifier_only_lstm.py to keep the original intact.
2. In both TeacherClassifier and StudentClassifier constructors, add `self.temporal_aggregator = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, num_layers=1, batch_first=True)`.
3. Update encode to stack segment latents to `[batch, segments, latent_dim]`, run through `self.temporal_aggregator`, and return the last time-step output (or hidden state) instead of `.mean(dim=1)`.
4. Leave forward shape/usage unchanged: use the new `z` for logits via the existing MLP, ensuring distillation alignment still uses `z`.
5. Adjust any training/eval scripts to import from the new LSTM module to avoid clashing with existing checkpoints that lack LSTM weights.

### Further Considerations
1. Check checkpoint loading: existing state_dicts will miss LSTM keys; plan to train new weights or load with `strict=False` if needed.
