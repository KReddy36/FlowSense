# FlowSense Prediction Evaluation

This evaluation measures the short-term position predictor against later
observed canonical track centers. Each sample uses only information available
through its source frame. A sample is eligible when the track has at least
`velocity_window + 1` observed points and the same canonical ID is observed
exactly 15 frames later.

The prediction error is Euclidean pixel distance from the predicted center to
the future observed center. The stationary baseline assumes the object remains
at its current observed center. Lower error is better.

## Configuration

- Rolling history: 30 points
- Averaged recent velocities: 5
- Prediction horizon: 15 frames

## Results

| Dataset | Samples | Median prediction error (px) | Median stationary baseline error (px) | Median improvement vs baseline (%) | Prediction win rate (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Video 1 | 2144 | 3.314 | 1.529 | -116.74 | 41.6 |
| Video 2 | 13168 | 8.683 | 30.363 | 71.4 | 95.41 |
| Video 3 | 8271 | 8.643 | 27.289 | 68.33 | 85.2 |
| Video 4 | 37805 | 11.246 | 41.002 | 72.57 | 81.12 |
| All videos | 61388 | 10.012 | 34.79 | 71.22 | 83.36 |

Across all videos, median prediction error was
10.012 px versus
34.79 px for the stationary
baseline, a 71.22% reduction in
median error. The predictor beat the baseline on
83.36% of eligible samples. The stationary baseline was stronger for Video 1, so the predictor is not uniformly better on every scene.

## Limitations

Future canonical track centers are used as observed ground truth. This
evaluates the prediction algorithm consistently on the repository datasets,
but detector or tracker localization errors can affect both the prediction
inputs and the future reference centers. It is not an independent
human-annotated position benchmark.

The full table, including mean and 90th-percentile errors, is stored in
`easy_results/prediction_accuracy.csv`.
