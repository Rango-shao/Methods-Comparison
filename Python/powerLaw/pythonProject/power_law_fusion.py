import cv2, numpy as np, time, os

def power_law_fusion(vis_path, ir_path, alpha=0.5, gamma=0.6, save_as='fused.jpg'):
    start_time = time.perf_counter()

    v = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    i = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if v is None or i is None:
        raise FileNotFoundError(f'Cannot read images: {vis_path} or {ir_path}')

    if v.shape != i.shape:
        i = cv2.resize(i, (v.shape[1], v.shape[0]))

    v, i = v.astype(np.float32) / 255.0, i.astype(np.float32) / 255.0

    fused = alpha * v + (1 - alpha) * i

    fused = np.power(fused, gamma)

    fused_uint8 = np.clip(fused * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(save_as, fused_uint8)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return elapsed_time, fused_uint8


def batch_process(image_pairs, alpha=0.5, gamma=0.6, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    timing_results = []

    print(f"{'Index':<6} {'IR Path':<20} {'Visible Path':<20} {'Fusion Time(ms)':<12}")
    print("-" * 60)

    for idx, (ir_path, vis_path) in enumerate(image_pairs, 1):
        output_path = os.path.join(output_dir, f'fused_{idx:02d}.jpg')

        try:
            elapsed, _ = power_law_fusion(vis_path, ir_path, alpha, gamma, output_path)
            elapsed_ms = elapsed * 1000

            timing_results.append({
                'pair': idx,
                'ir_path': ir_path,
                'vis_path': vis_path,
                'time_ms': elapsed_ms,
                'output': output_path
            })

            print(f"{idx:<6} {os.path.basename(ir_path):<20} {os.path.basename(vis_path):<20} {elapsed_ms:<12.2f}")

        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Fusion Time Statistics Summary")
    print(f"Total Pairs: {len(timing_results)}")
    print(f"Average Time: {np.mean([r['time_ms'] for r in timing_results]):.2f} ms")
    print(f"Minimum Time: {np.min([r['time_ms'] for r in timing_results]):.2f} ms")
    print(f"Maximum Time: {np.max([r['time_ms'] for r in timing_results]):.2f} ms")
    print("=" * 60)

    return timing_results


if __name__ == '__main__':
    IMAGE_PAIRS = [
        ('ir_grey/IR1.jpg', 'vis_grey/VIS1.jpg'),
        ('ir_grey/IR4.jpg', 'vis_grey/VIS4.jpg'),
        ('ir_grey/IR18.jpg', 'vis_grey/VIS18.jpg'),
    ]

    ALPHA = 0.5
    GAMMA = 0.6
    OUTPUT_DIR = 'fusion_results_powerlaw'

    results = batch_process(IMAGE_PAIRS, alpha=ALPHA, gamma=GAMMA, output_dir=OUTPUT_DIR)

    if results:
        import csv

        csv_path = os.path.join(OUTPUT_DIR, 'timing_report.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pair', 'ir_path', 'vis_path', 'time_ms', 'output'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed time record saved to: {csv_path}")