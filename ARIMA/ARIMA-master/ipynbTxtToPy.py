import json

# 需要先将ipynb文件复制成txt文件
def fun1():
    sourceFilePath = "sourceFile.txt"
    # generateFilePath = sourceFilePath.split("/")[-1].split("\\")[-1].split("ipynb")[0]+".py"
    generateFilePath = "target.py"
    # true表示处于cell中，false表示处于md中
    flag = True
    with open(sourceFilePath,"r") as sourceFile:
        with open(generateFilePath,"w") as generateFile:
            line = sourceFile.readline()
            lineList = []
            while len(line)>0:
                # 为markdown注释或者cell的开始标志
                if "#%%" in line:
                    # 正常cell
                    if "md" not in line:
                        flag = True
                        # 先将原来存储到lineList中的行写入目标python文件
                        generateFile.writelines(lineList)
                        # 清空lineList，从新开始存储
                        lineList = []
                    # markdown注释
                    else:
                        flag = False
                if flag and "#%%" not in line:
                    lineList.append(line)
                line = sourceFile.readline()
            generateFile.writelines(lineList)

# 直接将ipynb文件转成python文件
def fun2():
    sourceFilePath = "time-series-analysis-ARIMA.ipynb"
    # generateFilePath = sourceFilePath.split("/")[-1].split("\\")[-1].split("ipynb")[0]+".py"
    generateFilePath = "target2.py"
    with open(sourceFilePath, "r") as sourceFile:
        dict = json.load(sourceFile)
        with open(generateFilePath, "w") as generateFile:
            for cell in dict.get("cells"):
                if cell.get("cell_type") == "code":
                    for line in cell.get("source"):
                        generateFile.write(line if line[-1] == "\n" else f"{line}\n")

fun2()

"""














"""
