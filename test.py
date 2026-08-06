import sys

def format_num(x):
    # 保留两位小数，同时去掉多余的 0，匹配示例中的 -0.3 格式
    s = f"{x:.2f}"
    s = s.rstrip('0').rstrip('.')
    if s == "-0":
        s = "0"
    return s

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    idx = 0
    N = int(data[idx])
    D = int(data[idx + 1])
    idx += 2

    noise = []
    for _ in range(N):
        row = []
        for _ in range(D):
            row.append(float(data[idx]))
            idx += 1
        noise.append(row)

    real_data = []
    for _ in range(N):
        row = []
        for _ in range(D):
            row.append(float(data[idx]))
            idx += 1
        real_data.append(row)

    result = []
    for i in range(N):
        row = []
        for j in range(D):
            fake_value = real_data[i][j] + noise[i][j]
            row.append(format_num(fake_value))
        result.append(" ".join(row))

    print("\n".join(result))

if __name__ == "__main__":
    main()