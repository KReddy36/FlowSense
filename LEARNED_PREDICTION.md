# Learned hybrid motion prediction

FlowSense now combines its reliable constant-velocity forecast with a small
learned correction. The learned component does not replace tracking or invent
a new direction. It estimates a bounded scale for the existing prediction
arrow, then applies only 35% of that adjustment:

```text
baseline displacement = recent average velocity × prediction horizon
hybrid displacement = baseline displacement × applied learned scale
```

The corrector uses recent speed, acceleration, velocity variation, motion
consistency, class, confidence, bounding-box geometry, image position, and
track-history length. It is a regularized Ridge regression model trained from
the canonical trajectories. Future observations of the same track provide the
training targets automatically, so no additional manual labels were needed.

## Honest evaluation

The model-selection experiment used leave-one-video-out evaluation. Four models
were trained: each was fitted on three videos and evaluated only on the fourth.
This produced 61,388 predictions whose test video was never used for training.

| Metric across all held-out predictions | Constant velocity | Learned hybrid |
| --- | ---: | ---: |
| Median error | 10.012 px | **9.149 px** |
| Mean error | 21.621 px | **18.897 px** |
| 90th-percentile error | 55.414 px | **46.238 px** |

The hybrid had lower median, mean, and 90th-percentile error on every held-out
video and beat the constant-velocity forecast on 66.287% of eligible examples.
Video 1 is dominated by nearly stationary motion, so a stationary baseline
still has the lowest median error there.

The complete per-video table is stored in
`easy_results/learned_prediction_results.csv`.

After validating the method, the portable production model was refitted using
all four available videos. Its in-sample performance must not be presented as
held-out evidence; the table above is the appropriate generalization result.

## Production behavior

The exported model is JSON rather than a scikit-learn `joblib` file. FlowSense
evaluates its coefficients using Python's standard library, avoiding
scikit-learn version incompatibilities at runtime.

The learned correction is used automatically for the trained settings:

- 15-frame prediction horizon
- Five recent velocity segments
- Supported classes: bus, car, motorcycle, person, and truck

FlowSense safely uses the original constant-velocity predictor when:

- The model file cannot be loaded or validated.
- The requested horizon or velocity window differs from training.
- A track does not yet have enough observed history.
- Learned prediction is disabled with
  `--disable-learned-prediction`.

The correction is deliberately limited: the raw learned scale is clipped to
0.25–1.75, and only 35% of the distance from 1.0 is applied. Consequently, the
production arrow remains between 73.75% and 126.25% of the mathematical
baseline's length.

## Limitations

- Predictions remain in image-pixel coordinates, not real-world meters.
- The model was trained on only four videos from this project.
- It adjusts the arrow length but does not yet learn a new turning direction.
- Camera perspective and scene differences may affect generalization.
- A larger and more varied held-out dataset would be needed for stronger
  claims.
