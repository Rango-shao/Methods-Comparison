import cv2, numpy as np, time, os, csv, pywt


def wavelet_fusion_single(ir_path, vis_path, wavelet='haar', levels=2, output_path=None):
    start_time = time.perf_counter()

    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    if ir_img is None or vis_img is None:
        raise FileNotFoundError(f'Cannot read images: {ir_path} or {vis_path}')

    if ir_img.shape != vis_img.shape:
        vis_img = cv2.resize(vis_img, (ir_img.shape[1], ir_img.shape[0]))

    ir_img = ir_img.astype(np.float32) / 255.0
    vis_img = vis_img.astype(np.float32) / 255.0

    coeffs_ir = pywt.wavedec2(ir_img, wavelet, level=levels)
    coeffs_vis = pywt.wavedec2(vis_img, wavelet, level=levels)

    coeffs_fused = []

    coeffs_fused.append((coeffs_ir[0] + coeffs_vis[0]) / 2)

    for i in range(1, len(coeffs_ir)):
        cH_ir, cV_ir, cD_ir = coeffs_ir[i]
        cH_vis, cV_vis, cD_vis = coeffs_vis[i]

        mask_H = np.abs(cH_ir) > np.abs(cH_vis)
        cH_fused = mask_H * cH_ir + (~mask_H) * cH_vis

        mask_V = np.abs(cV_ir) > np.abs(cV_vis)
        cV_fused = mask_V * cV_ir + (~mask_V) * cV_vis

        mask_D = np.abs(cD_ir) > np.abs(cD_vis)
        cD_fused = mask_D * cD_ir + (~mask_D) * cD_vis

        coeffs_fused.append((cH_fused, cV_fused, cD_fused))

    fused_img = pywt.waverec2(coeffs_fused, wavelet)

    fused_img = np.clip(fused_img, 0, 1)
    fused_uint8 = (fused_img * 255).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, fused_uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return elapsed_ms, fused_uint8


def batch_process(image_pairs, wavelet='haar', levels=2, output_dir='fusion_results_wavelet'):
    os.makedirs(output_dir, exist_ok=True)
    timing_results = []

    print(f"{'Index':<6} {'IR Path':<20} {'Visible Path':<20} {'Fusion Time(ms)':<12}")
    print("-" * 60)

    for idx, (ir_path, vis_path) in enumerate(image_pairs, 1):
        output_path = os.path.join(output_dir, f'fused_{idx:02d}.jpg')

        try:
            elapsed_ms, _ = wavelet_fusion_single(ir_path, vis_path, wavelet, levels, output_path)
            timing_results.append({
                'pair': idx,
                'ir_path': ir_path,
                'vis_path': vis_path,
                'time_ms': elapsed_ms,
                'output': output_path
            })

            print(f"{idx:<6d} {os.path.basename(ir_path):<20} {os.path.basename(vis_path):<20} {elapsed_ms:<12.2f}")

        except Exception as e:
            print(f"{idx:<6d} Processing error: {str(e)[:30]}")
            continue

    if timing_results:
        times = [r['time_ms'] for r in timing_results]
        print("\n" + "=" * 60)
        print("Fusion Time Statistics Summary")
        print(f"Total Pairs: {len(image_pairs)} (Success: {len(timing_results)})")
        print(f"Average Time: {np.mean(times):.2f} ms")
        print(f"Minimum Time: {np.min(times):.2f} ms")
        print(f"Maximum Time: {np.max(times):.2f} ms")
        print(f"Standard Deviation: {np.std(times):.2f} ms")
        print("=" * 60)

    return timing_results


if __name__ == '__main__':
    IMAGE_PAIRS = [
        ('ir_grey/IR1.jpg', 'vis_grey/VIS1.jpg'),
        ('ir_grey/IR4.jpg', 'vis_grey/VIS4.jpg'),
        ('ir_grey/IR18.jpg', 'vis_grey/VIS18.jpg'),
    ]

    WAVELET_TYPE = 'haar'
    DECOMP_LEVEL = 2
    OUTPUT_DIR = 'fusion_results_wavelet'

    results = batch_process(IMAGE_PAIRS, wavelet=WAVELET_TYPE, levels=DECOMP_LEVEL, output_dir=OUTPUT_DIR)

    if results:
        csv_path = os.path.join(OUTPUT_DIR, 'timing_report.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pair', 'ir_path', 'vis_path', 'time_ms', 'output'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed time record saved to: {csv_path}")