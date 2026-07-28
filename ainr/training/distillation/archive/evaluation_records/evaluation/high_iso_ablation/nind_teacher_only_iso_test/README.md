# ISO Test Set Visual Validation

This is a visual and teacher-agreement study. The Sony JPEGs do not have paired clean
targets, so these results must not be read as clean-reference PSNR/SSIM.

Student checkpoint: **LiteDenoise NIND teacher-only** (`nind_teacher_only`).

| ISO | Full comparison | Region map | Detailed crops | Student/teacher PSNR | Preview SSIM |
| ---: | --- | --- | --- | ---: | ---: |
| 1600 | [view](ISO1600_DSC00001/full_comparison.jpg) | [view](ISO1600_DSC00001/region_map.jpg) | [view](ISO1600_DSC00001/detail_contact_sheet.jpg) | 44.150 | 0.982231 |
| 3200 | [view](ISO3200_DSC00002/full_comparison.jpg) | [view](ISO3200_DSC00002/region_map.jpg) | [view](ISO3200_DSC00002/detail_contact_sheet.jpg) | 42.039 | 0.966290 |
| 6400 | [view](ISO6400_DSC00003/full_comparison.jpg) | [view](ISO6400_DSC00003/region_map.jpg) | [view](ISO6400_DSC00003/detail_contact_sheet.jpg) | 38.307 | 0.931349 |
| 12800 | [view](ISO12800_DSC00004/full_comparison.jpg) | [view](ISO12800_DSC00004/region_map.jpg) | [view](ISO12800_DSC00004/detail_contact_sheet.jpg) | 37.044 | 0.915501 |
| 25600 | [view](ISO25600_DSC00005/full_comparison.jpg) | [view](ISO25600_DSC00005/region_map.jpg) | [view](ISO25600_DSC00005/detail_contact_sheet.jpg) | 32.619 | 0.825815 |
| 51200 | [view](ISO51200_DSC00006/full_comparison.jpg) | [view](ISO51200_DSC00006/region_map.jpg) | [view](ISO51200_DSC00006/detail_contact_sheet.jpg) | 29.925 | 0.715112 |

Preview SSIM is measured after scaling the full frame to at most 1024 px. Native
crop PSNR/SSIM values are available in `report.json`.

## Findings

- Student/teacher agreement falls from 44.15 dB at ISO 1600 to 29.93 dB at ISO 51200.
- In input shadows, student correction is 79.1% of SCUNet at ISO 1600 and 46.5% at ISO 51200. The largest visible gap is high-ISO shadow and chroma noise.
- The maximum measured seam-gradient ratio is 1.047x versus nearby gradients; the deployment-style feathering did not introduce a systematic seam spike.
- These comparisons establish teacher imitation behavior only. Without a clean paired
  target, they cannot determine whether every SCUNet correction is desirable.

Each ISO folder also contains twelve native 768 px lossless crop sheets and the
lossless full-frame student and SCUNet outputs.

## Cross-ISO Detail Sheets

- [01 Dark shelf](by_region/01_dark_shelf.jpg)
- [02 Smooth blue background](by_region/02_blue_background.jpg)
- [03 Bright screen and text](by_region/03_screen.jpg)
- [04 Red cap and colored edge](by_region/04_red_cap.jpg)
- [05 White figure and face detail](by_region/05_white_figure.jpg)
- [06 Specular metal](by_region/06_metal_bit.jpg)
- [07 Color blocks and edges](by_region/07_color_blocks.jpg)
- [08 Dark tool body](by_region/08_dark_tool.jpg)
- [09 Fine dot pattern](by_region/09_dot_pattern.jpg)
- [10 Black table and edge](by_region/10_black_table.jpg)
- [11 Fine printed label](by_region/11_fine_label.jpg)
- [12 Transparent bottle and desk](by_region/12_bottle_desk.jpg)
