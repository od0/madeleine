# VPT-small foreign gameplay scorecard

This is a mapped-foreign **development** scorecard. It does not replace or blend with the native-keyboard eligibility gates. The y4n vertical-axis sign is indeterminate, so up/down results are label-noise-sensitive.

| Family | Endpoint | Rows | Macro AP | Macro state F1 | Down AP | Down recall | Down PPR / prevalence | All-key nonzero recall |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| VPT-small | `tier_b_13p45h_epoch20` | 555,840 | 0.7381 | 0.6368 | 0.4591 | 0.2116 | 0.262 | yes |
| VPT-small | `unflagged92_103p4056h_epoch20` | 555,840 | 0.4177 | 0.1826 | 0.0441 | 0.0044 | 0.102 | yes |
| VPT-small | `unflagged92_down_ridge5pct_ft5e_epoch5` | 555,840 | 0.4121 | 0.1923 | 0.0429 | 0.0233 | 0.332 | yes |
| pixel GRU | `gru_unflagged92_103p4056h` | 554,304 | 0.2693 | 0.2888 | 0.0373 | 0.2013 | 4.375 | yes |
| pixel GRU | `gru_all_valid210` | 554,304 | 0.2723 | 0.2986 | 0.0342 | 0.2783 | 7.823 | yes |

Historical GRU rows are shown as contextual evidence on their native full-y4n support. Only the VPT-small rows are guaranteed identical across endpoints.

## Paired native and foreign view

The scorecards are intentionally not averaged. Native values use each model's
fixed val-A report; foreign values use the identical 555,840 active y4n rows
above. The y4n surface is development-only and its vertical-axis sign is
indeterminate, so the foreign up/down columns are diagnostic rather than final
test evidence.

| Endpoint | Native macro AP | Native down AP | Foreign macro AP | Foreign down AP |
|---|---:|---:|---:|---:|
| Tier-B/Wild 13.45h | 0.3603 | 0.0102 | **0.7381** | **0.4591** |
| NitroGen unflagged 103.41h | **0.4209** | 0.0109 | 0.4177 | 0.0441 |
| 103.41h + 5% Ridge fine-tune | 0.4030 | 0.0116 | 0.4121 | 0.0429 |

## Interpretation

- The earlier native-only framing was incomplete. The small Tier-B/Wild model
  has strong threshold-free action ranking on mapped foreign gameplay,
  including `down`; its native `down` failure is therefore not evidence that
  the architecture cannot represent concurrent down actions.
- More hours did not monotonically improve the foreign scorecard. The
  103-hour NitroGen model is worse than the 13.45-hour Tier-B/Wild model on
  every key's AP. This points to population and label quality as first-order
  variables, not merely a shortage of positive `down` examples.
- The Ridge fine-tune moves the 0.5-threshold down prediction rate from 0.24%
  to 0.79% on y4n, but down AP falls from 0.0441 to 0.0429, macro AP falls from
  0.4177 to 0.4121, and NLL worsens. It changes operating behavior without
  improving ranking quality.
- All three VPT endpoints have nonzero recall for every key on y4n, but none
  satisfies the 0.5--2.0 predicted-positive-rate/prevalence band for every
  key. This development scorecard describes capability; it is not a promotion
  gate pass.
- A full 161.97-hour VPT-small arm is scientifically useful because it tests
  whether the additional all-valid NitroGen and admitted-Wild data recover
  foreign performance or compound label noise. It must remain a distinct arm:
  it sees the promote reserve, while the 135-hour reserve-excluded model is
  still the candidate for reserve-based labeler-unseen work.

The practical model-selection rule is now role-specific: use the native
scorecard for native-keyboard eligibility and the foreign scorecard for
Internet-labeling development. A final foreign claim still requires a
prospectively frozen, multi-video advanced-gameplay test set.
