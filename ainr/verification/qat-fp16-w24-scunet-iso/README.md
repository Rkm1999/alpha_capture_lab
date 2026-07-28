# S25 model quality comparison

No-reference ISO comparison. PSNR/SSIM measure agreement with SCUNet, not restoration accuracy.

Open each ISO folder's `detail_comparison.jpg` or the `by_region` sheets for matched 100% crops.

```json
{
  "w24": {
    "mean_psnr_to_scunet": 41.99303127682578,
    "mean_preview_ssim_to_scunet": 0.9522417883078257
  },
  "w32_fp16": {
    "mean_psnr_to_scunet": 41.03007950293165,
    "mean_preview_ssim_to_scunet": 0.9290850857893626
  },
  "w32_qat": {
    "mean_psnr_to_scunet": 37.208367957757936,
    "mean_preview_ssim_to_scunet": 0.9231991271177927
  },
  "qat_to_fp16": {
    "mean_psnr": 41.124628847353485,
    "mean_mae": 0.006933545832540474
  }
}
```
