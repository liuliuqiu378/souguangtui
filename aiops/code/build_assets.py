"""把仿真输出转成前端可直接加载的 JS 数据文件。"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
ASSETS = os.path.join(BASE, "..", "assets")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    a = load("chart_a.json")
    b = load("chart_b.json")
    with open(os.path.join(ASSETS, "data_a.js"), "w", encoding="utf-8") as f:
        f.write("window.DATA_A = ")
        json.dump(a["project_a"], f, ensure_ascii=False)
        f.write(";\n")
    with open(os.path.join(ASSETS, "data_b.js"), "w", encoding="utf-8") as f:
        f.write("window.DATA_B = ")
        json.dump(b["project_b"], f, ensure_ascii=False)
        f.write(";\n")
    print("data_a.js", os.path.getsize(os.path.join(ASSETS, "data_a.js")))
    print("data_b.js", os.path.getsize(os.path.join(ASSETS, "data_b.js")))


if __name__ == "__main__":
    main()
