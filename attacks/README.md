# Attacks: threat model and protocol

All attacks are **digital and offline**, applied to copies of the authorised COCO val2017 subset. Nothing is tested on real vehicles or physical objects.

| Set name | Attack | Attacker knowledge | Budget |
|---|---|---|---|
| `pgd_eps4`, `pgd_eps8` | PGD, vanishing objective | White-box: YOLOv8n gradients | L∞ ≤ 4/255 or 8/255, 10 steps, step size eps/4, random start |
| `transfer_eps8` | PGD on surrogate `yolov5nu.pt`, applied to YOLOv8n | Black-box: no access to YOLOv8n | L∞ ≤ 8/255 |
| `patch` | Universal DPatch-style patch (80×80 at 640 scale), trained on dev images | White-box during training | Patch covers about 1.6% of the input |
| `patch_random` | Random-pixel patch, same size and positions | None (occlusion control) | Same as `patch` |

- **Space:** every attack works in the 640×640 letterboxed model input and changes only the content region, never the padding. Images are saved as **PNG crops of the content region**, so loading and letterboxing them gives exactly the attacked input.
- **Objective:** a vanishing (disappearance) attack. The loss is the sum of the best class score of every anchor above 0.1; minimising it removes detections.
- **Images:** the same 200 test images for every set (`splits/attack_ids.csv`), each with at least one clean detection. The patch is trained only on development images.
- **Attack success** (measured on YOLOv8n, confidence 0.25): an image counts as attacked successfully when at least 50% of its clean detections are lost, meaning they have no same-class match with IoU ≥ 0.5.
- **Screening** is evaluated on all attack attempts and separately on successful attacks only, because an attack being generated does not mean it worked (Section 17.4).
- **Not covered:** adaptive attacks that know about the JPEG/blur check. These are listed as a limitation and future work.
