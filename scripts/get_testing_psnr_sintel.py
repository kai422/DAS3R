import os

root = 'results/sintel_rearranged'
exps = ['testing_pnsr_4000']
results = {}
for exp in exps:
    results[exp] = {}
for scene in sorted(os.listdir(root)):
    if os.path.isdir(os.path.join(root, scene)):
        for exp in exps:
            train_log = os.path.join(root, scene, exp, 'test_log.txt')
            if os.path.exists(train_log):
                with open(train_log, 'r') as file:
                    data = file.read()
                last_line = data.strip().split('\n')[-1]
                last_number = float(last_line.split()[-1])
                results[exp][scene] = last_number
                print(f'{scene},{exp[:10]}: {last_number}')

print("Scene & " + " & ".join(results[exps[0]].keys()).replace('_', '-') + "& average")
for exp in exps:
    avg_psnr = sum(results[exp].values()) / len(results[exp].values()) if results[exp].values() else 0
    print(f"PSNR & " + " & ".join(f"{results[exp].get(scene, 'N/A'):.2f}" for scene in results[exps[0]].keys()) + f" & {avg_psnr:.2f} ")

