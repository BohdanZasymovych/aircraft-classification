import os

MAPPING = {
    "0": "0",
    "5": "1",
    "11": "2",
    "16": "3",
    "18": "4",
    "19": "5"
}

if __name__ == "__main__":
    os.chdir("/home/bohdan/code/aircraft-classification/data/russian-planes-yolov8-dataset/all-labels")
    for file in os.listdir():
        with open(file, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            corrected_lines = []
            for line in lines:
                line = line.strip().split()
                line[0] = MAPPING[line[0]]
                line = " ".join(line)
                print(lines)
                corrected_lines.append(line)
            f.seek(0)
            f.write("\n".join(corrected_lines))
            f.truncate()